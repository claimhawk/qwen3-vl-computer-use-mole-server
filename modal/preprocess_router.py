#!/usr/bin/env python3
# Copyright (c) 2025 Tylt LLC. All rights reserved.
# Licensed for research use only. Commercial use requires a license from Tylt LLC.
# Contact: hello@claimhawk.app | See LICENSE for terms.

"""
Router Preprocessing from Expert Datasets

Aggregates preprocessed expert datasets and converts them to router format.
Instead of preprocessing from scratch, this reuses the expert .pt files and
just swaps the labels to numeric routing labels.

Benefits:
- Uses ALL expert training data (not just 500 samples per adapter)
- No image reprocessing needed (much faster)
- Always current - linked to deployed expert datasets
- More data = better router accuracy

Usage:
    modal run modal/preprocess_router.py
"""

import json
import os
import random
from pathlib import Path

import modal

# =============================================================================
# CONFIGURATION
# =============================================================================

# Expert datasets to aggregate (expert_name -> preprocessed dataset path)
# Uses the LATEST preprocessed dataset for each expert
EXPERT_DATASETS = {
    "calendar": "calendar--mike--20251206_165432",
    "claim-window": "claim-window--mike--20251207_163040",
    "appointment": "appointment--mike--20251208_103029",
    "login-window": "login-window--mike--20251208_230108",
    "chart-screen": "chart-screen--mike--20251209_085858",
    # desktop: not preprocessed yet
    # ocr: external model, needs separate handling
}

# Numeric labels (must match config/adapters.yaml)
EXPERT_LABELS = {
    "calendar": 0,
    "claim-window": 1,
    "ocr": 2,
    "desktop": 3,
    "appointment": 4,
    "login-window": 5,
    "chart-screen": 6,
}

# Volume names
MOE_VOLUME_NAME = "moe-lora-data"
EXPERT_VOLUME_NAME = "claimhawk-lora-training"

# Modal setup
app = modal.App("router-preprocessing")
moe_volume = modal.Volume.from_name(MOE_VOLUME_NAME, create_if_missing=True)
expert_volume = modal.Volume.from_name(EXPERT_VOLUME_NAME, create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        extra_options="--index-url https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers>=4.57.0",
        "tqdm>=4.65.0",
    )
)


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=7200,
    volumes={
        "/moe-data": moe_volume,
        "/expert-data": expert_volume,
    },
)
def preprocess_router_from_experts(
    output_name: str = None,
    val_split: float = 0.1,
    test_split: float = 0.05,
    seed: int = 42,
    max_per_expert: int = None,
):
    """
    Aggregate expert datasets into router training format.

    Reads preprocessed .pt files from each expert dataset and converts
    them to router format by replacing the label with the numeric class.
    """
    import torch
    from datetime import datetime
    from tqdm import tqdm
    from transformers import AutoTokenizer

    random.seed(seed)

    # Reload volumes
    expert_volume.reload()
    moe_volume.reload()

    # Generate output name if not provided
    if output_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"router_from_experts_{timestamp}"

    print(f"\n{'='*70}")
    print("Router Preprocessing from Expert Datasets")
    print(f"{'='*70}")
    print(f"Output: {output_name}")
    print(f"Val split: {val_split}")
    print(f"Test split: {test_split}")
    print(f"Max per expert: {max_per_expert or 'unlimited'}")

    # Load tokenizer for creating new labels
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    # Collect all samples from expert datasets
    all_samples = []  # List of (expert_name, label, pt_path)

    print(f"\n{'='*70}")
    print("Collecting Expert Samples")
    print(f"{'='*70}")

    for expert_name, dataset_name in EXPERT_DATASETS.items():
        label = EXPERT_LABELS[expert_name]
        train_dir = Path(f"/expert-data/preprocessed/{dataset_name}/train")

        if not train_dir.exists():
            print(f"  {expert_name}: MISSING (skipping)")
            continue

        pt_files = sorted(train_dir.glob("sample_*.pt"))

        if max_per_expert and len(pt_files) > max_per_expert:
            random.shuffle(pt_files)
            pt_files = pt_files[:max_per_expert]

        for pt_path in pt_files:
            all_samples.append((expert_name, label, pt_path))

        print(f"  {expert_name}: {len(pt_files)} samples (label={label})")

    print(f"\nTotal samples: {len(all_samples)}")

    # Shuffle and split
    random.shuffle(all_samples)

    n_total = len(all_samples)
    n_test = int(n_total * test_split)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val - n_test

    test_samples = all_samples[:n_test]
    val_samples = all_samples[n_test:n_test + n_val]
    train_samples = all_samples[n_test + n_val:]

    print(f"\nSplit: train={n_train}, val={n_val}, test={n_test}")

    # Create output directories
    output_base = Path(f"/moe-data/preprocessed/{output_name}")
    train_dir = output_base / "train"
    val_dir = output_base / "val"
    test_dir = output_base / "test"

    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    def create_router_label(label_num: int) -> torch.Tensor:
        """Create a label tensor for router training (just the number + eos)."""
        # The label should be: -100 (ignore) for all input tokens, then the label token
        # But since we're replacing the expert's label, we just need the new label tokens
        label_str = str(label_num)
        label_tokens = tokenizer.encode(label_str, add_special_tokens=False)
        # Add the end token (151645 = <|im_end|>)
        label_tokens.append(151645)
        return torch.tensor(label_tokens, dtype=torch.int64)

    def process_sample(expert_name: str, label: int, pt_path: Path, output_path: Path):
        """Load expert sample and convert to router format."""
        # Load the expert sample
        sample = torch.load(pt_path, weights_only=False)

        # Get the original tensors
        input_ids = sample["input_ids"]
        attention_mask = sample["attention_mask"]
        pixel_values = sample["pixel_values"]
        image_grid_thw = sample["image_grid_thw"]
        original_labels = sample["labels"]

        # Find where the actual label starts (first non -100 value)
        label_start = None
        for i, val in enumerate(original_labels.tolist()):
            if val != -100:
                label_start = i
                break

        if label_start is None:
            # No label found, skip this sample
            return False

        # Create new labels: -100 for input portion, then new label
        new_label_tokens = create_router_label(label)

        # Truncate input_ids and attention_mask to just before the label
        new_input_ids = input_ids[:label_start]
        new_attention_mask = attention_mask[:label_start]

        # Append the new label tokens to input_ids
        new_input_ids = torch.cat([new_input_ids, new_label_tokens])
        new_attention_mask = torch.cat([
            new_attention_mask,
            torch.ones(len(new_label_tokens), dtype=torch.int64)
        ])

        # Create new labels tensor
        new_labels = torch.full((len(new_input_ids),), -100, dtype=torch.int64)
        new_labels[-len(new_label_tokens):] = new_label_tokens

        # Save the converted sample
        router_sample = {
            "input_ids": new_input_ids,
            "attention_mask": new_attention_mask,
            "labels": new_labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "expert": expert_name,
            "routing_label": label,
        }

        torch.save(router_sample, output_path)
        return True

    # Process all splits
    splits = [
        ("train", train_samples, train_dir),
        ("val", val_samples, val_dir),
        ("test", test_samples, test_dir),
    ]

    stats = {"train": 0, "val": 0, "test": 0}
    expert_stats = {name: {"train": 0, "val": 0, "test": 0} for name in EXPERT_DATASETS}

    for split_name, samples, output_dir in splits:
        print(f"\n{'='*70}")
        print(f"Processing {split_name} split ({len(samples)} samples)")
        print(f"{'='*70}")

        idx = 0
        for expert_name, label, pt_path in tqdm(samples, desc=split_name):
            output_path = output_dir / f"sample_{idx:06d}.pt"
            if process_sample(expert_name, label, pt_path, output_path):
                stats[split_name] += 1
                expert_stats[expert_name][split_name] += 1
                idx += 1

    # Save metadata
    metadata = {
        "name": output_name,
        "created": datetime.now().isoformat(),
        "source_datasets": EXPERT_DATASETS,
        "expert_labels": EXPERT_LABELS,
        "splits": {
            "train": stats["train"],
            "val": stats["val"],
            "test": stats["test"],
        },
        "per_expert": expert_stats,
        "config": {
            "val_split": val_split,
            "test_split": test_split,
            "seed": seed,
            "max_per_expert": max_per_expert,
        },
    }

    with open(output_base / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Commit volume
    moe_volume.commit()

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Output: {output_base}")
    print(f"Train: {stats['train']}")
    print(f"Val: {stats['val']}")
    print(f"Test: {stats['test']}")
    print(f"\nPer-expert breakdown:")
    for expert_name in EXPERT_DATASETS:
        es = expert_stats[expert_name]
        print(f"  {expert_name}: train={es['train']}, val={es['val']}, test={es['test']}")

    return metadata


@app.local_entrypoint()
def main(
    output_name: str = None,
    val_split: float = 0.1,
    test_split: float = 0.05,
    seed: int = 42,
    max_per_expert: int = None,
):
    """Run router preprocessing from expert datasets."""
    result = preprocess_router_from_experts.remote(
        output_name=output_name,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
        max_per_expert=max_per_expert,
    )

    print(f"\n{'='*70}")
    print("PREPROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Dataset: {result['name']}")
    print(f"Train samples: {result['splits']['train']}")
    print(f"Val samples: {result['splits']['val']}")
    print(f"Test samples: {result['splits']['test']}")
