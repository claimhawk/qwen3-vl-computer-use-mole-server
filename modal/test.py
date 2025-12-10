#!/usr/bin/env python3
# Copyright (c) 2025 Tylt LLC. All rights reserved.
# Licensed for research use only. Commercial use requires a license from Tylt LLC.
# Contact: hello@claimhawk.app | See LICENSE for terms.

"""Evaluate routing LoRA on held-out eval data.

Runs inference on eval.jsonl and reports accuracy per adapter class.

Usage:
    modal run modal/eval.py --run-name routing-20251125-052946 --dataset-name datasets/routing_20251125_052946
"""

import json
from pathlib import Path

import modal

# =============================================================================
# CENTRALIZED CONFIGURATION
# =============================================================================
# Volume names and adapter info are loaded from config/adapters.yaml via the SDK.
# Users can customize these by editing the YAML file.

# Numeric label mappings (must match config/adapters.yaml)
LABEL_TO_ADAPTER = {
    "0": "calendar",
    "1": "claim-window",
    "2": "ocr",
    "3": "desktop",
    "4": "appointment",
    "5": "login-window",
    "6": "chart-screen",
}
ADAPTER_TO_LABEL = {v: k for k, v in LABEL_TO_ADAPTER.items()}

try:
    from sdk.modal_compat import (
        get_volume_name,
        get_valid_experts,
        get_base_vlm,
    )
    MOE_VOLUME_NAME = get_volume_name("moe_data")
    VALID_ADAPTERS = get_valid_experts()
    BASE_MODEL = get_base_vlm()
except ImportError:
    # Fallback for Modal remote execution
    MOE_VOLUME_NAME = "moe-lora-data"
    VALID_ADAPTERS = {"calendar", "claim-window", "ocr",
                      "appointment", "login-window", "desktop", "chart-screen"}
    BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

# System prompt uses numeric labels (must match training)
_label_list = "\n".join(f"- {label}: {name}" for label, name in sorted(LABEL_TO_ADAPTER.items()))
EVAL_SYSTEM_PROMPT = f"""You are a Mixture of Experts router. You have been trained to look at an image and a text instruction and determine what adapter to route to.

Valid adapter labels:
{_label_list}

Reply with only the numeric label (0-6) for the adapter that should handle this image and instruction."""

app = modal.App("routing-lora-eval")

# Volumes (using centralized config)
moe_volume = modal.Volume.from_name(MOE_VOLUME_NAME, create_if_missing=False)
model_cache = modal.Volume.from_name("claimhawk-model-cache", create_if_missing=True)
inference_volume = modal.Volume.from_name("moe-inference", create_if_missing=False)

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
        "tqdm>=4.65.0",
    )
)


@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
    volumes={
        "/moe-data": moe_volume,
        "/models": model_cache,  # Pre-cached HF models
        "/inference": inference_volume,  # Deployed adapters
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def evaluate_routing_lora(
    run_name: str,
    dataset_name: str,
    max_samples: int = 100,
    checkpoint_step: int = None,  # Specific checkpoint step (e.g., 60 for checkpoint-60)
    use_deployed: bool = False,  # Use deployed adapter from /inference/routing/adapter
):
    """Evaluate routing LoRA accuracy on held-out eval data."""
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from peft import PeftModel
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from tqdm import tqdm

    # Reload volume
    moe_volume.reload()

    print(f"\n{'='*70}")
    print("Routing LoRA Evaluation")
    print(f"{'='*70}")
    print(f"Run: {run_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Max samples: {max_samples}")
    print(f"Use deployed: {use_deployed}")

    base_dataset_name = Path(dataset_name).name

    if use_deployed:
        # Use the deployed adapter from inference volume
        checkpoint_path = Path("/inference/routing/adapter")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No deployed router found at {checkpoint_path}")
        print(f"Using deployed adapter: {checkpoint_path}")
    else:
        # Find checkpoint from training run
        # Try router_linked path first (new train_router.py format)
        checkpoint_base = Path(f"/moe-data/checkpoints/router_linked/{run_name}")

        if not checkpoint_base.exists():
            # Try old path format
            checkpoint_base = Path(f"/moe-data/checkpoints/{dataset_name}/{run_name}")

        if not checkpoint_base.exists():
            # Try without datasets/ prefix
            checkpoint_base = Path(f"/moe-data/checkpoints/{base_dataset_name}/{run_name}")

        if not checkpoint_base.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_base}")

        # Find checkpoint
        if checkpoint_step is not None:
            # Use specific checkpoint
            checkpoint_path = checkpoint_base / f"checkpoint-{checkpoint_step}"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        else:
            # Find best checkpoint (or final)
            checkpoints = sorted(checkpoint_base.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[1]))
            if checkpoints:
                checkpoint_path = checkpoints[-1]  # Use latest
            else:
                checkpoint_path = checkpoint_base

    print(f"Checkpoint: {checkpoint_path}")

    # Load eval data (check for both eval.jsonl and test.jsonl)
    eval_path = Path(f"/moe-data/{dataset_name}/eval.jsonl")
    if not eval_path.exists():
        eval_path = Path(f"/moe-data/datasets/{base_dataset_name}/eval.jsonl")
    if not eval_path.exists():
        eval_path = Path(f"/moe-data/{dataset_name}/test.jsonl")
    if not eval_path.exists():
        eval_path = Path(f"/moe-data/datasets/{base_dataset_name}/test.jsonl")

    if not eval_path.exists():
        raise FileNotFoundError(f"Eval data not found (checked eval.jsonl and test.jsonl): {eval_path}")

    eval_data = []
    with open(eval_path) as f:
        for line in f:
            if line.strip():
                eval_data.append(json.loads(line))

    # Group by adapter and sample equally from each
    from collections import defaultdict
    import random
    random.seed(42)

    by_adapter = defaultdict(list)
    for sample in eval_data:
        adapter = sample["metadata"]["adapter"]
        by_adapter[adapter].append(sample)

    num_adapters = len(by_adapter)
    samples_per_adapter = max_samples // num_adapters

    balanced_data = []
    for adapter, samples in by_adapter.items():
        random.shuffle(samples)
        balanced_data.extend(samples[:samples_per_adapter])

    random.shuffle(balanced_data)
    eval_data = balanced_data

    print(f"Test samples: {len(eval_data)} ({samples_per_adapter} per adapter, {num_adapters} adapters)")

    # Load model - prefer cached version from volume
    print("\nLoading base model...")
    model_name = BASE_MODEL
    cached_model_path = f"/models/{model_name.replace('/', '--')}"

    # Check if cached model exists on volume
    if Path(cached_model_path).exists():
        print(f"  Using cached model from {cached_model_path}")
        load_path = cached_model_path
    else:
        print(f"  Cached model not found, downloading from HuggingFace: {model_name}")
        load_path = model_name

    processor = AutoProcessor.from_pretrained(load_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        load_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA adapter
    print(f"Loading LoRA from {checkpoint_path}...")
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()
    print("Model loaded")

    # Run evaluation
    print(f"\n{'='*70}")
    print("Running Evaluation")
    print(f"{'='*70}\n")

    results = {
        "correct": 0,
        "total": 0,
        "per_class": {adapter: {"correct": 0, "total": 0} for adapter in VALID_ADAPTERS},
        "predictions": [],
    }

    images_dir = eval_path.parent / "images"

    for sample in tqdm(eval_data, desc="Evaluating"):
        # Get ground truth
        expected_adapter = sample["metadata"]["adapter"]

        # Build prompt - always inject system prompt to match preprocessing
        messages = [{"role": "system", "content": EVAL_SYSTEM_PROMPT}]
        image_path = None

        for conv in sample["conversations"]:
            # Skip any system prompts in the data - we use our own
            if conv["from"] == "system":
                continue
            elif conv["from"] == "human":
                content = []
                value = conv["value"]

                # Check for image
                if "<image>" in value:
                    img_name = Path(sample["image"]).name
                    image_path = images_dir / img_name
                    if image_path.exists():
                        content.append({"type": "image", "image": f"file://{image_path}"})
                    value = value.replace("<image>", "").strip()

                if value:
                    content.append({"type": "text", "text": value})

                messages.append({"role": "user", "content": content})

        # Generate
        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )

            # Decode only new tokens
            input_len = inputs["input_ids"].shape[1]
            generated = processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

            # Model outputs numeric label (0-6), convert to adapter name for comparison
            predicted_label = generated.strip()
            predicted_adapter = LABEL_TO_ADAPTER.get(predicted_label, f"unknown-{predicted_label}")

            # Compare predicted adapter name to expected adapter name
            is_correct = predicted_adapter == expected_adapter

            results["total"] += 1
            results["per_class"][expected_adapter]["total"] += 1

            if is_correct:
                results["correct"] += 1
                results["per_class"][expected_adapter]["correct"] += 1

            results["predictions"].append({
                "expected": expected_adapter,
                "predicted": predicted_adapter,
                "correct": is_correct,
            })

        except Exception as e:
            print(f"Error processing sample: {e}")
            results["total"] += 1
            results["per_class"][expected_adapter]["total"] += 1
            results["predictions"].append({
                "expected": expected_adapter,
                "predicted": f"ERROR: {e}",
                "correct": False,
            })

    # Print results
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}\n")

    overall_acc = results["correct"] / results["total"] * 100 if results["total"] > 0 else 0
    print(f"Overall Accuracy: {results['correct']}/{results['total']} ({overall_acc:.1f}%)")
    print()

    print("Per-class accuracy:")
    per_class_acc = {}
    for adapter in VALID_ADAPTERS:
        stats = results["per_class"][adapter]
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"] * 100
            per_class_acc[adapter] = acc
            print(f"  {adapter:20s}: {stats['correct']:3d}/{stats['total']:3d} ({acc:.1f}%)")
        else:
            per_class_acc[adapter] = None
            print(f"  {adapter:20s}: N/A (0 samples)")

    # Show some errors
    errors = [p for p in results["predictions"] if not p["correct"]]
    if errors:
        print(f"\nSample errors ({len(errors)} total):")
        for err in errors[:5]:
            print(f"  Expected: {err['expected']:20s} | Predicted: {err['predicted']}")

    # Build eval report
    from datetime import datetime
    eval_report = {
        "run_name": run_name,
        "dataset_name": dataset_name,
        "checkpoint": str(checkpoint_path),
        "eval_samples": len(eval_data),
        "timestamp": datetime.now().isoformat(),
        "accuracy": {
            "overall": overall_acc,
            "per_class": per_class_acc,
        },
        "results": {
            "correct": results["correct"],
            "total": results["total"],
            "per_class": results["per_class"],
        },
        "errors": errors[:10],  # First 10 errors for debugging
    }

    # Save eval-report.json to the dataset folder
    dataset_dir = eval_path.parent
    report_path = dataset_dir / "eval-report.json"
    with open(report_path, "w") as f:
        json.dump(eval_report, f, indent=2)
    print(f"\n📄 Eval report saved to: {report_path}")

    # Commit volume so the report is persisted
    moe_volume.commit()
    print("💾 Volume committed")

    return eval_report


@app.local_entrypoint()
def main(
    run_name: str = None,
    dataset_name: str = None,
    max_samples: int = 100,
    checkpoint_step: int = None,
    deployed: bool = False,
):
    """Run evaluation and download report locally.

    Args:
        run_name: Name of the training run (required unless --deployed)
        dataset_name: Name of the dataset (required)
        max_samples: Max samples to evaluate
        checkpoint_step: Specific checkpoint step to evaluate
        deployed: If True, test the currently deployed router at /inference/routing/adapter
    """
    import subprocess
    from pathlib import Path

    if not dataset_name:
        print("ERROR: --dataset-name is required")
        exit(1)

    if deployed:
        # Test the deployed router
        result = evaluate_routing_lora.remote(
            run_name="deployed",
            dataset_name=dataset_name,
            max_samples=max_samples,
            checkpoint_step=None,
            use_deployed=True,
        )
    else:
        if not run_name:
            print("ERROR: --run-name is required unless using --deployed")
            exit(1)
        result = evaluate_routing_lora.remote(
            run_name=run_name,
            dataset_name=dataset_name,
            max_samples=max_samples,
            checkpoint_step=checkpoint_step,
        )

    print(f"\n{'='*70}")
    print("EVAL RESULTS")
    print(f"{'='*70}")
    print(f"Overall accuracy: {result['accuracy']['overall']:.1f}%")
    print(f"Per-class accuracy:")
    for adapter, acc in result['accuracy']['per_class'].items():
        if acc is not None:
            print(f"  {adapter}: {acc:.1f}%")
        else:
            print(f"  {adapter}: N/A")

    # Download eval-report.json to local dataset folder
    base_dataset_name = Path(dataset_name).name
    local_dataset_dir = Path(f"datasets/{base_dataset_name}")
    local_dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading eval-report.json to {local_dataset_dir}...")
    subprocess.run(
        ["uvx", "modal", "volume", "get", "moe-lora-data",
         f"datasets/{base_dataset_name}/eval-report.json",
         str(local_dataset_dir)],
        check=True,
    )
    print(f"📄 Eval report downloaded to: {local_dataset_dir / 'eval-report.json'}")
