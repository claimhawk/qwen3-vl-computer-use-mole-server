#!/usr/bin/env python3
"""Stacked inference: Routing LoRA → Task LoRA or Chandra.

Uses the trained routing LoRA to classify which adapter/model to use:
- calendar, claim-window, provider-select → Qwen + task LoRA
- chandra → Chandra OCR model → Qwen formats result as tool_call

Usage:
    modal run modal/stacked_inference.py --test
    modal run modal/stacked_inference.py --sample-idx 0
"""

import json
from pathlib import Path
from typing import Any

import modal

app = modal.App("stacked-lora-inference")

# Volumes
moe_volume = modal.Volume.from_name("moe-lora-data", create_if_missing=False)
checkpoints_volume = modal.Volume.from_name("moe-lora-checkpoints", create_if_missing=False)

# Image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        extra_options="--index-url https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers>=4.57.0",
        "accelerate>=0.26.0",
        "peft>=0.14.0",
        "qwen-vl-utils",
        "Pillow>=10.0.0",
    )
)

VALID_LORA_ADAPTERS = {"calendar", "claim-window", "provider-select"}
VALID_ADAPTERS = VALID_LORA_ADAPTERS | {"chandra"}

# Chandra OCR model
CHANDRA_MODEL = "datalab-to/Chandra-VLM-4B-Instruct"

# Routing LoRA checkpoint
DEFAULT_ROUTING_CHECKPOINT = "checkpoints/datasets/routing_20251125_065129/routing-20251125-065129/final"

# Task LoRA base paths (from config/loras.json) - on moe-lora-checkpoints volume
# Script will auto-detect if /final subfolder exists
TASK_LORA_BASE_PATHS = {
    "calendar": "/checkpoints/calendar-tasks",
    "claim-window": "/checkpoints/claim-window",
    "provider-select": "/checkpoints/provider-select",
}


def get_task_lora_path(adapter: str) -> str:
    """Get the correct task LoRA path, checking for /final subfolder."""
    base_path = TASK_LORA_BASE_PATHS.get(adapter)
    if not base_path:
        return None

    # Check if /final exists (training saves to subdirs)
    final_path = Path(base_path) / "final"
    if final_path.exists():
        return str(final_path)

    # Otherwise use base path (direct upload)
    return base_path


def format_ocr_tool_call(text: str) -> str:
    """Format OCR text as a tool_call response.

    Args:
        text: Raw text extracted by Chandra

    Returns:
        Formatted tool_call string for the ocr action
    """
    tool_call = {
        "name": "computer_use",
        "arguments": {
            "action": "ocr",
            "text": text,
        },
    }
    return f"Action: Return the extracted text.\n<tool_call>\n{json.dumps(tool_call)}\n</tool_call>"


def run_chandra_ocr(image_path: str, device: str = "cuda") -> str:
    """Run Chandra OCR model on an image.

    Args:
        image_path: Path to the image file
        device: Device to run on

    Returns:
        Extracted text from the image
    """
    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from PIL import Image

    print(f"Loading Chandra model: {CHANDRA_MODEL}")
    processor = AutoProcessor.from_pretrained(CHANDRA_MODEL, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        CHANDRA_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Load and process image
    image = Image.open(image_path).convert("RGB")

    # Chandra prompt for OCR
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Read and extract all text from this image."},
            ],
        }
    ]

    # Process inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[1]
    extracted_text = processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return extracted_text


@app.function(
    image=image,
    gpu="H100",
    timeout=600,
    volumes={
        "/moe-data": moe_volume,
        "/checkpoints": checkpoints_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def stacked_inference(
    sample: dict,
    routing_checkpoint: str = DEFAULT_ROUTING_CHECKPOINT,
) -> dict[str, Any]:
    """Run full stacked inference: route → task LoRA → output.

    Args:
        sample: Sample dict with conversations, image, metadata
        routing_checkpoint: Path to routing LoRA checkpoint

    Returns:
        Dict with routing result and task output
    """
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info

    moe_volume.reload()
    checkpoints_volume.reload()

    device = torch.device("cuda")
    model_name = "Qwen/Qwen3-VL-8B-Instruct"

    expected_adapter = sample["metadata"]["adapter"]

    print(f"\n{'='*60}")
    print("Stacked LoRA Inference")
    print(f"{'='*60}")
    print(f"Sample ID: {sample.get('id', 'unknown')}")
    print(f"Expected adapter: {expected_adapter}")

    # Load base model
    print("\nLoading base model...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # === STEP 1: Route with routing LoRA ===
    checkpoint_path = Path(f"/moe-data/{routing_checkpoint}")
    print(f"\nStep 1: Loading routing LoRA from {checkpoint_path}...")
    router_model = PeftModel.from_pretrained(base_model, checkpoint_path)
    router_model.eval()

    # Build messages from sample for routing
    messages = []
    image_path = None
    dataset_dir = Path("/moe-data/datasets/routing_20251125_065129")

    for conv in sample["conversations"]:
        if conv["from"] == "system":
            messages.append({"role": "system", "content": conv["value"]})
        elif conv["from"] == "human":
            content = []
            value = conv["value"]

            if "<image>" in value:
                img_name = Path(sample["image"]).name
                image_path = dataset_dir / "images" / img_name
                if image_path.exists():
                    content.append({"type": "image", "image": f"file://{image_path}"})
                value = value.replace("<image>", "").strip()

            if value:
                content.append({"type": "text", "text": value})

            messages.append({"role": "user", "content": content})

    # Generate routing decision
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print("Running routing inference...")
    with torch.no_grad():
        outputs = router_model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    routed_adapter = processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip().lower()

    print(f"Routed to: {routed_adapter}")
    routing_correct = routed_adapter == expected_adapter

    result = {
        "sample_id": sample.get("id", "unknown"),
        "expected_adapter": expected_adapter,
        "routed_adapter": routed_adapter,
        "routing_correct": routing_correct,
    }

    if routed_adapter not in VALID_ADAPTERS:
        print(f"ERROR: Invalid adapter '{routed_adapter}'")
        result["task_output"] = None
        result["error"] = f"Invalid adapter: {routed_adapter}"
        return result

    # Unload routing LoRA
    del router_model
    torch.cuda.empty_cache()

    # === STEP 2: Run task model ===
    if routed_adapter == "chandra":
        # === OCR path: Chandra → format as tool_call ===
        print(f"\nStep 2: Running Chandra OCR on {image_path}...")

        # Run Chandra to extract text
        extracted_text = run_chandra_ocr(str(image_path), device=str(device))
        print(f"Chandra extracted: {extracted_text[:100]}...")

        # Format as tool_call
        task_output = format_ocr_tool_call(extracted_text)

    else:
        # === LoRA path: Load task-specific LoRA ===
        task_lora_path = get_task_lora_path(routed_adapter)
        print(f"\nStep 2: Loading task LoRA from {task_lora_path}...")

        # Load task LoRA
        task_model = PeftModel.from_pretrained(base_model, task_lora_path)
        task_model.eval()

        # Build task messages (use original system prompt + user message, no routing modification)
        task_messages = []
        for conv in sample["conversations"]:
            if conv["from"] == "system":
                task_messages.append({"role": "system", "content": conv["value"]})
            elif conv["from"] == "human":
                content = []
                value = conv["value"]

                if "<image>" in value:
                    img_name = Path(sample["image"]).name
                    image_path = dataset_dir / "images" / img_name
                    if image_path.exists():
                        content.append({"type": "image", "image": f"file://{image_path}"})
                    value = value.replace("<image>", "").strip()

                if value:
                    content.append({"type": "text", "text": value})

                task_messages.append({"role": "user", "content": content})

        # Generate task output
        text = processor.apply_chat_template(task_messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(task_messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        print("Running task inference...")
        with torch.no_grad():
            outputs = task_model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        task_output = processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

    result["task_output"] = task_output

    print(f"\n{'='*60}")
    print("RESULT")
    print(f"{'='*60}")
    print(f"Routing: {expected_adapter} -> {routed_adapter} ({'CORRECT' if routing_correct else 'WRONG'})")
    print(f"Task output:\n{task_output}")
    print(f"{'='*60}\n")

    return result


@app.function(
    image=image,
    gpu="H100",
    timeout=1800,
    volumes={
        "/moe-data": moe_volume,
        "/checkpoints": checkpoints_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def test_stacked_inference(
    dataset_name: str = "datasets/routing_20251125_065129",
    routing_checkpoint: str = DEFAULT_ROUTING_CHECKPOINT,
    max_samples: int = 9,  # 3 per adapter
) -> dict[str, Any]:
    """Test full stacked inference on eval samples."""
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info

    moe_volume.reload()
    checkpoints_volume.reload()

    device = torch.device("cuda")
    model_name = "Qwen/Qwen3-VL-8B-Instruct"

    print(f"\n{'='*60}")
    print("Stacked Inference Test")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Routing checkpoint: {routing_checkpoint}")

    # Load eval data
    base_name = Path(dataset_name).name
    eval_path = Path(f"/moe-data/{dataset_name}/eval.jsonl")
    if not eval_path.exists():
        eval_path = Path(f"/moe-data/datasets/{base_name}/eval.jsonl")

    eval_data = []
    with open(eval_path) as f:
        for line in f:
            if line.strip():
                eval_data.append(json.loads(line))

    # Get balanced samples (equal per adapter)
    samples_per_adapter = max_samples // 3
    selected = []
    for adapter in VALID_ADAPTERS:
        adapter_samples = [s for s in eval_data if s["metadata"]["adapter"] == adapter][:samples_per_adapter]
        selected.extend(adapter_samples)

    print(f"Testing {len(selected)} samples ({samples_per_adapter} per adapter)")

    # Load models
    print("\nLoading base model...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load routing LoRA
    checkpoint_path = Path(f"/moe-data/{routing_checkpoint}")
    print(f"Loading routing LoRA from {checkpoint_path}...")
    router_model = PeftModel.from_pretrained(base_model, checkpoint_path)
    router_model.eval()

    # Check task LoRAs exist
    print("Checking task LoRAs...")
    for adapter_name in TASK_LORA_BASE_PATHS:
        lora_path = get_task_lora_path(adapter_name)
        if lora_path and Path(lora_path).exists():
            print(f"  Found {adapter_name} at {lora_path}")
        else:
            print(f"  WARNING: {adapter_name} not found")

    results = []
    dataset_dir = Path(f"/moe-data/{dataset_name}")

    for i, sample in enumerate(selected):
        expected_adapter = sample["metadata"]["adapter"]
        print(f"\n--- Sample {i+1}/{len(selected)}: {sample.get('id', 'unknown')} (expected: {expected_adapter}) ---")

        # Build messages
        messages = []
        image_path = None

        for conv in sample["conversations"]:
            if conv["from"] == "system":
                messages.append({"role": "system", "content": conv["value"]})
            elif conv["from"] == "human":
                content = []
                value = conv["value"]

                if "<image>" in value:
                    img_name = Path(sample["image"]).name
                    image_path = dataset_dir / "images" / img_name
                    if image_path.exists():
                        content.append({"type": "image", "image": f"file://{image_path}"})
                    value = value.replace("<image>", "").strip()

                if value:
                    content.append({"type": "text", "text": value})

                messages.append({"role": "user", "content": content})

        # Route
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = router_model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        routed_adapter = processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip().lower()
        routing_correct = routed_adapter == expected_adapter

        print(f"  Routed: {routed_adapter} ({'OK' if routing_correct else 'WRONG'})")

        result = {
            "sample_id": sample.get("id", "unknown"),
            "expected_adapter": expected_adapter,
            "routed_adapter": routed_adapter,
            "routing_correct": routing_correct,
            "task_output": None,
        }

        # Run task LoRA if routing is valid and path exists
        if routed_adapter in VALID_ADAPTERS:
            task_lora_path = get_task_lora_path(routed_adapter)
            if task_lora_path and Path(task_lora_path).exists():
                # Load fresh base model for task to avoid adapter conflicts
                task_base = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                )
                task_model = PeftModel.from_pretrained(task_base, task_lora_path)
                task_model.eval()

                # Generate task output
                with torch.no_grad():
                    outputs = task_model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=False,
                        pad_token_id=processor.tokenizer.pad_token_id,
                    )

                task_output = processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
                result["task_output"] = task_output
                print(f"  Task output: {task_output[:100]}...")

                del task_model
                del task_base
                torch.cuda.empty_cache()

        results.append(result)

    # Summary
    routing_correct = sum(1 for r in results if r["routing_correct"])
    has_output = sum(1 for r in results if r["task_output"])

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Routing accuracy: {routing_correct}/{len(results)} ({100*routing_correct/len(results):.1f}%)")
    print(f"Samples with task output: {has_output}/{len(results)}")

    # Per-adapter breakdown
    for adapter in VALID_ADAPTERS:
        adapter_results = [r for r in results if r["expected_adapter"] == adapter]
        correct = sum(1 for r in adapter_results if r["routing_correct"])
        print(f"  {adapter}: {correct}/{len(adapter_results)}")

    return {
        "routing_accuracy": routing_correct / len(results) * 100,
        "samples_with_output": has_output,
        "total_samples": len(results),
        "results": results,
    }


@app.local_entrypoint()
def main(
    test: bool = False,
    sample_idx: int = -1,
    dataset_name: str = "datasets/routing_20251125_065129",
    max_samples: int = 9,
):
    """Run stacked inference test."""
    if test or sample_idx < 0:
        result = test_stacked_inference.remote(
            dataset_name=dataset_name,
            max_samples=max_samples,
        )
        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Routing accuracy: {result['routing_accuracy']:.1f}%")
        print(f"Samples with task output: {result['samples_with_output']}/{result['total_samples']}")

        # Print each result
        for r in result["results"]:
            status = "OK" if r["routing_correct"] else "WRONG"
            print(f"\n{r['sample_id']}:")
            print(f"  Route: {r['expected_adapter']} -> {r['routed_adapter']} [{status}]")
            print(f"  Output: {r['task_output'] or 'N/A'}")
    else:
        # Load specific sample
        import subprocess
        eval_path = Path(f"datasets/{Path(dataset_name).name}/eval.jsonl")
        with open(eval_path) as f:
            samples = [json.loads(line) for line in f if line.strip()]

        if sample_idx >= len(samples):
            print(f"Sample index {sample_idx} out of range (max {len(samples)-1})")
            return

        result = stacked_inference.remote(sample=samples[sample_idx])
        print(json.dumps(result, indent=2))
