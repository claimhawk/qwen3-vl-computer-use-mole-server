#!/usr/bin/env python3
# Copyright (c) 2025 Tylt LLC. All rights reserved.
# Licensed for research use only. Commercial use requires a license from Tylt LLC.
# Contact: hello@claimhawk.app | See LICENSE for terms.

"""Evaluate Router LoRA on validation data from linked experts.

Uses the same manifest format as train_router.py to load validation samples
from expert datasets and evaluate router accuracy.

Usage:
    modal run modal/eval_router.py --run-name router-tb-fixed --manifest router_linked_20251209_233704 --checkpoint-step 40
"""

import json
import random
from pathlib import Path

import modal

# Numeric label mappings (must match config/adapters.yaml)
LABEL_TO_ADAPTER = {
    0: "calendar",
    1: "claim-window",
    2: "ocr",
    3: "desktop",
    4: "appointment",
    5: "login-window",
    6: "chart-screen",
}
ADAPTER_TO_LABEL = {v: k for k, v in LABEL_TO_ADAPTER.items()}

# Volume names
MOE_VOLUME_NAME = "moe-lora-data"
EXPERT_VOLUME_NAME = "claimhawk-lora-training"
MODEL_CACHE_NAME = "claimhawk-model-cache"
BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

app = modal.App("router-eval")

moe_volume = modal.Volume.from_name(MOE_VOLUME_NAME, create_if_missing=False)
expert_volume = modal.Volume.from_name(EXPERT_VOLUME_NAME, create_if_missing=False)
model_cache = modal.Volume.from_name(MODEL_CACHE_NAME, create_if_missing=True)

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
        "/expert-data": expert_volume,
        "/models": model_cache,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def evaluate_router(
    run_name: str,
    manifest_name: str,
    checkpoint_step: int = None,
    max_samples_per_expert: int = 50,
    seed: int = 42,
    use_test_set: bool = False,
):
    """Evaluate router checkpoint on held-out test data or validation data.

    Two modes:
    - TEST SET (use_test_set=True): Uses held-out samples from TRAINING data
      that were never seen during training. These are specified by test_indices
      in the manifest, which are indices into the train/ directory.
    - VALIDATION (use_test_set=False): Uses samples from the val/ directory,
      which may have been seen during training for early stopping.

    Args:
        run_name: Name of the training run
        manifest_name: Name of the manifest file
        checkpoint_step: Specific checkpoint step to evaluate
        max_samples_per_expert: Max samples per expert for validation mode
        seed: Random seed for reproducibility
        use_test_set: If True, use held-out test indices from train data
    """
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from peft import PeftModel
    from tqdm import tqdm

    random.seed(seed)
    torch.manual_seed(seed)

    # Reload volumes
    moe_volume.reload()
    expert_volume.reload()

    eval_mode = "TEST SET (held-out)" if use_test_set else "VALIDATION"
    print(f"\n{'='*70}")
    print(f"Router Evaluation - {eval_mode}")
    print(f"{'='*70}")
    print(f"Run: {run_name}")
    print(f"Manifest: {manifest_name}")
    print(f"Checkpoint step: {checkpoint_step or 'latest'}")
    print(f"Mode: {eval_mode}")
    if not use_test_set:
        print(f"Max samples per expert: {max_samples_per_expert}")

    # Load manifest
    manifest_path = Path(f"/moe-data/router_manifests/{manifest_name}.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"\nExperts: {list(manifest['experts'].keys())}")

    # Find checkpoint
    checkpoint_base = Path(f"/moe-data/checkpoints/router_linked/{run_name}")
    if not checkpoint_base.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_base}")

    if checkpoint_step is not None:
        checkpoint_path = checkpoint_base / f"checkpoint-{checkpoint_step}"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    else:
        # Find best checkpoint (or final)
        checkpoints = sorted(checkpoint_base.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[1]))
        if checkpoints:
            checkpoint_path = checkpoints[-1]
        else:
            checkpoint_path = checkpoint_base / "final"

    print(f"Checkpoint: {checkpoint_path}")

    # Collect evaluation samples
    eval_files = []  # List of (pt_path, label, expert_name)

    for expert_name, info in manifest["experts"].items():
        label = info["label"]

        if use_test_set:
            # Use held-out test indices from train directory
            train_dir = Path(f"/expert-data/{info['train_path']}")
            test_indices = info.get("test_indices", [])

            if not train_dir.exists():
                print(f"  {expert_name}: NO TRAIN DIR")
                continue

            if not test_indices:
                print(f"  {expert_name}: NO TEST INDICES (older manifest?)")
                continue

            # Get specific files by index
            all_train_files = sorted(train_dir.glob("sample_*.pt"))
            expert_test_files = [all_train_files[i] for i in test_indices if i < len(all_train_files)]

            for f in expert_test_files:
                eval_files.append((f, label, expert_name))

            print(f"  {expert_name}: {len(expert_test_files)} test samples")
        else:
            # Use validation directory (existing behavior)
            val_dir = Path(f"/expert-data/{info['val_path']}")

            if not val_dir.exists():
                print(f"  {expert_name}: NO VAL DIR")
                continue

            expert_val_files = sorted(val_dir.glob("sample_*.pt"))
            if len(expert_val_files) > max_samples_per_expert:
                random.shuffle(expert_val_files)
                expert_val_files = expert_val_files[:max_samples_per_expert]

            for f in expert_val_files:
                eval_files.append((f, label, expert_name))

            print(f"  {expert_name}: {len(expert_val_files)} val samples")

    random.shuffle(eval_files)
    sample_type = "test" if use_test_set else "validation"
    print(f"\nTotal {sample_type} samples: {len(eval_files)}")

    # Load model
    print(f"\n{'='*70}")
    print("Loading Model")
    print(f"{'='*70}")

    cached_model_path = f"/models/{BASE_MODEL.replace('/', '--')}"
    if Path(cached_model_path).exists():
        print(f"Using cached model: {cached_model_path}")
        model_path = cached_model_path
    else:
        print(f"Downloading from HuggingFace: {BASE_MODEL}")
        model_path = BASE_MODEL

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA adapter
    print(f"Loading LoRA from {checkpoint_path}...")
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()

    # Pre-tokenize labels for comparison
    label_tokens = {}
    for label in range(7):
        tokens = processor.tokenizer.encode(str(label), add_special_tokens=False)
        label_tokens[label] = tokens

    # Run evaluation
    print(f"\n{'='*70}")
    print("Running Evaluation")
    print(f"{'='*70}\n")

    results = {
        "correct": 0,
        "total": 0,
        "per_class": {name: {"correct": 0, "total": 0} for name in LABEL_TO_ADAPTER.values()},
        "predictions": [],
    }

    for pt_path, expected_label, expert_name in tqdm(eval_files, desc="Evaluating"):
        try:
            # Load the preprocessed sample
            sample = torch.load(pt_path, weights_only=False)

            input_ids = sample["input_ids"]
            attention_mask = sample["attention_mask"]
            pixel_values = sample["pixel_values"]
            image_grid_thw = sample["image_grid_thw"]
            original_labels = sample["labels"]

            # Find where the label starts (first non -100)
            label_start = None
            for i, val in enumerate(original_labels.tolist()):
                if val != -100:
                    label_start = i
                    break

            if label_start is None:
                label_start = len(input_ids) - 10

            # Use only the input portion (before the label)
            gen_input_ids = input_ids[:label_start].unsqueeze(0).to(model.device)
            gen_attention_mask = attention_mask[:label_start].unsqueeze(0).to(model.device)
            gen_pixel_values = pixel_values.to(model.device)
            gen_image_grid_thw = image_grid_thw.to(model.device)

            # Generate prediction
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=gen_input_ids,
                    attention_mask=gen_attention_mask,
                    pixel_values=gen_pixel_values,
                    image_grid_thw=gen_image_grid_thw,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )

            # Decode prediction
            input_len = gen_input_ids.shape[1]
            generated = processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

            # Parse predicted label
            try:
                predicted_label = int(generated.split()[0])
            except (ValueError, IndexError):
                predicted_label = -1

            is_correct = predicted_label == expected_label
            expected_adapter = LABEL_TO_ADAPTER.get(expected_label, f"unknown-{expected_label}")
            predicted_adapter = LABEL_TO_ADAPTER.get(predicted_label, f"unknown-{generated}")

            results["total"] += 1
            results["per_class"][expert_name]["total"] += 1

            if is_correct:
                results["correct"] += 1
                results["per_class"][expert_name]["correct"] += 1

            results["predictions"].append({
                "expected": expected_adapter,
                "expected_label": expected_label,
                "predicted": predicted_adapter,
                "predicted_label": predicted_label,
                "raw_output": generated,
                "correct": is_correct,
            })

        except Exception as e:
            print(f"Error processing {pt_path}: {e}")
            results["total"] += 1
            results["per_class"][expert_name]["total"] += 1
            results["predictions"].append({
                "expected": expert_name,
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
    for adapter in LABEL_TO_ADAPTER.values():
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
        for err in errors[:10]:
            print(f"  Expected: {err['expected']:15s} | Predicted: {err.get('predicted', 'N/A'):15s} | Raw: {err.get('raw_output', 'N/A')[:20]}")

    # Build eval report
    from datetime import datetime
    eval_report = {
        "run_name": run_name,
        "manifest_name": manifest_name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "eval_mode": "test" if use_test_set else "validation",
        "eval_samples": len(eval_files),
        "max_samples_per_expert": max_samples_per_expert if not use_test_set else None,
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
        "errors": errors[:20],
    }

    # Save eval report
    report_path = checkpoint_path / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(eval_report, f, indent=2)
    print(f"\n📄 Eval report saved to: {report_path}")

    moe_volume.commit()

    return eval_report


@app.local_entrypoint()
def main(
    run_name: str,
    manifest: str,
    checkpoint_step: int = None,
    max_per_expert: int = 50,
    test: bool = False,
):
    """Evaluate router on held-out test data or validation data.

    Args:
        run_name: Name of the training run
        manifest: Name of the manifest file (without .json)
        checkpoint_step: Specific checkpoint step (e.g., 40)
        max_per_expert: Max validation samples per expert (ignored if --test)
        test: If True, use held-out test samples from train data (never seen during training)
    """
    result = evaluate_router.remote(
        run_name=run_name,
        manifest_name=manifest,
        checkpoint_step=checkpoint_step,
        max_samples_per_expert=max_per_expert,
        use_test_set=test,
    )

    eval_mode = "TEST SET" if test else "VALIDATION"
    print(f"\n{'='*70}")
    print(f"EVALUATION COMPLETE ({eval_mode})")
    print(f"{'='*70}")
    print(f"Overall accuracy: {result['accuracy']['overall']:.1f}%")
    print(f"\nPer-class accuracy:")
    for adapter, acc in result['accuracy']['per_class'].items():
        if acc is not None:
            print(f"  {adapter}: {acc:.1f}%")
        else:
            print(f"  {adapter}: N/A")
