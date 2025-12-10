#!/usr/bin/env python3
# Copyright (c) 2025 Tylt LLC. All rights reserved.
# Licensed for research use only. Commercial use requires a license from Tylt LLC.
# Contact: hello@claimhawk.app | See LICENSE for terms.

"""
Link Router Data from Expert Datasets

Dynamically discovers the latest preprocessed dataset for each expert
by scanning the preprocessed/ directory and using folder timestamps.
NO REPROCESSING - just links the existing .pt files.

The training script handles label remapping at runtime.

Usage:
    modal run modal/link_router_data.py
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

import modal

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

# Default max records per expert (balances classes, prevents huge datasets)
DEFAULT_MAX_RECORDS = 1000
DEFAULT_TEST_SAMPLES = 20  # Held-out test samples per expert

# Volume names
MOE_VOLUME_NAME = "moe-lora-data"
EXPERT_VOLUME_NAME = "claimhawk-lora-training"

app = modal.App("router-link-data")
moe_volume = modal.Volume.from_name(MOE_VOLUME_NAME, create_if_missing=True)
expert_volume = modal.Volume.from_name(EXPERT_VOLUME_NAME, create_if_missing=False)

image = modal.Image.debian_slim(python_version="3.11")


def find_latest_dataset(preprocessed_dir: Path, expert_name: str) -> str | None:
    """Find the latest preprocessed dataset for an expert based on timestamp.

    Dataset naming: {expert}--{researcher}--{timestamp} or ocr--aggregated--{timestamp}
    Timestamp format: YYYYMMDD_HHMMSS

    Returns the dataset folder name (not full path) or None if not found.
    """
    pattern = re.compile(rf"^{re.escape(expert_name)}--.*--(\d{{8}}_\d{{6}})$")

    candidates = []
    for entry in preprocessed_dir.iterdir():
        if entry.is_dir():
            match = pattern.match(entry.name)
            if match:
                timestamp_str = match.group(1)
                candidates.append((timestamp_str, entry.name))

    if not candidates:
        return None

    # Sort by timestamp (YYYYMMDD_HHMMSS sorts lexicographically)
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]  # Return the latest


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=3600,
    volumes={
        "/moe-data": moe_volume,
        "/expert-data": expert_volume,
    },
)
def link_expert_data(
    output_name: str = None,
    max_records: int = None,
    test_samples: int = None,
    expert_overrides: dict = None,
):
    """
    Create a manifest file that points to expert preprocessed datasets.

    Dynamically discovers the latest preprocessed dataset for each expert
    by scanning the preprocessed/ directory and selecting based on timestamp.

    Args:
        output_name: Name for the manifest file
        max_records: Maximum records per expert (limits to balance classes)
        test_samples: Number of held-out test samples per expert (default 20)
        expert_overrides: Dict of expert_name -> sample_count for custom per-expert limits
    """
    expert_volume.reload()
    moe_volume.reload()

    if output_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"router_linked_{timestamp}"

    # Use defaults if not specified
    if max_records is None:
        max_records = DEFAULT_MAX_RECORDS
    if test_samples is None:
        test_samples = DEFAULT_TEST_SAMPLES

    print(f"\n{'='*70}")
    print("Linking Expert Datasets for Router Training")
    print(f"{'='*70}")
    print(f"Output: {output_name}")
    print(f"Max records per expert: {max_records}")
    print(f"Test samples per expert: {test_samples}")

    preprocessed_dir = Path("/expert-data/preprocessed")

    # Discover latest dataset for each expert
    print(f"\n{'='*70}")
    print("Discovering Latest Datasets")
    print(f"{'='*70}")

    expert_datasets = {}
    for expert_name in EXPERT_LABELS.keys():
        dataset_name = find_latest_dataset(preprocessed_dir, expert_name)
        if dataset_name:
            expert_datasets[expert_name] = dataset_name
            print(f"  {expert_name}: {dataset_name}")
        else:
            print(f"  {expert_name}: NOT FOUND (skipping)")

    # Collect info about each expert dataset
    manifest = {
        "name": output_name,
        "created": datetime.now().isoformat(),
        "type": "linked",
        "max_records": max_records,
        "test_samples_per_expert": test_samples,
        "experts": {},
        "totals": {"train": 0, "val": 0, "test": 0},
    }

    print(f"\n{'='*70}")
    print("Linking Datasets")
    print(f"{'='*70}")

    import random as rng
    rng.seed(42)  # Deterministic holdout selection

    for expert_name, dataset_name in expert_datasets.items():
        label = EXPERT_LABELS[expert_name]
        train_dir = Path(f"/expert-data/preprocessed/{dataset_name}/train")
        val_dir = Path(f"/expert-data/preprocessed/{dataset_name}/val")

        if not train_dir.exists():
            print(f"  {expert_name}: MISSING train dir at {train_dir}")
            continue

        # Get all training files
        all_train_files = sorted(train_dir.glob("sample_*.pt"))
        train_count = len(all_train_files)
        val_count = len(list(val_dir.glob("sample_*.pt"))) if val_dir.exists() else 0

        # Hold out test_samples from training data for test set
        # We'll record specific file indices to use as test set
        test_indices = []
        if train_count > test_samples:
            # Randomly select test_samples indices to hold out
            all_indices = list(range(train_count))
            rng.shuffle(all_indices)
            test_indices = sorted(all_indices[:test_samples])

        # Calculate remaining train count after holdout
        available_train = train_count - len(test_indices)
        # Use expert-specific override if provided, else use max_records
        expert_max = expert_overrides.get(expert_name, max_records) if expert_overrides else max_records
        limited_train = min(available_train, expert_max)
        limited_val = min(val_count, expert_max // 4)  # Proportional val limit

        manifest["experts"][expert_name] = {
            "dataset": dataset_name,
            "label": label,
            "train_path": f"preprocessed/{dataset_name}/train",
            "val_path": f"preprocessed/{dataset_name}/val",
            "train_count": limited_train,
            "val_count": limited_val,
            "test_count": len(test_indices),
            "test_indices": test_indices,  # Specific file indices held out for test
            "total_available": train_count,  # Record full count for reference
        }

        manifest["totals"]["train"] += limited_train
        manifest["totals"]["val"] += limited_val
        manifest["totals"]["test"] += len(test_indices)

        limited_note = f" (limited from {available_train})" if available_train > max_records else ""
        print(f"  {expert_name} (label={label}): {limited_train} train, {limited_val} val, {len(test_indices)} test{limited_note}")

    # Save manifest to moe-data volume
    output_dir = Path(f"/moe-data/router_manifests")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / f"{output_name}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    moe_volume.commit()

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Manifest saved: {manifest_path}")
    print(f"Total train: {manifest['totals']['train']}")
    print(f"Total val: {manifest['totals']['val']}")
    print(f"Total test (held-out): {manifest['totals']['test']}")
    print(f"\nExperts linked: {list(manifest['experts'].keys())}")

    return manifest


@app.local_entrypoint()
def main(
    output_name: str = None,
    max_records: int = None,
    test_samples: int = None,
    desktop: int = None,
    appointment: int = None,
    calendar: int = None,
    claim_window: int = None,
    ocr: int = None,
    login_window: int = None,
    chart_screen: int = None,
):
    """Create linked router dataset manifest.

    Args:
        output_name: Name for the manifest
        max_records: Max records per expert (default 1000)
        test_samples: Held-out test samples per expert (default 20)
        desktop: Custom sample count for desktop expert
        appointment: Custom sample count for appointment expert
        calendar: Custom sample count for calendar expert
        claim_window: Custom sample count for claim-window expert
        ocr: Custom sample count for ocr expert
        login_window: Custom sample count for login-window expert
        chart_screen: Custom sample count for chart-screen expert
    """
    # Build expert_overrides from CLI args
    expert_overrides = {}
    if desktop is not None:
        expert_overrides["desktop"] = desktop
    if appointment is not None:
        expert_overrides["appointment"] = appointment
    if calendar is not None:
        expert_overrides["calendar"] = calendar
    if claim_window is not None:
        expert_overrides["claim-window"] = claim_window
    if ocr is not None:
        expert_overrides["ocr"] = ocr
    if login_window is not None:
        expert_overrides["login-window"] = login_window
    if chart_screen is not None:
        expert_overrides["chart-screen"] = chart_screen

    result = link_expert_data.remote(
        output_name=output_name,
        max_records=max_records,
        test_samples=test_samples,
        expert_overrides=expert_overrides if expert_overrides else None,
    )

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")
    print(f"Manifest: {result['name']}")
    print(f"Max records per expert: {result['max_records']}")
    if expert_overrides:
        print(f"Expert overrides: {expert_overrides}")
    print(f"Train samples: {result['totals']['train']}")
    print(f"Val samples: {result['totals']['val']}")
    print(f"Test samples (held-out): {result['totals']['test']}")
    print(f"\nPer-expert breakdown:")
    for name, info in result["experts"].items():
        print(f"  {name}: {info['train_count']} train, {info['test_count']} test")
    print(f"\nUse with: modal run modal/train_router.py --manifest {result['name']}")
