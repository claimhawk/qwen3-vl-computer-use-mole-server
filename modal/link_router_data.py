#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["modal", "pyyaml"]
# ///
# Copyright (c) 2025 Tylt LLC. All rights reserved.
# Licensed for research use only. Commercial use requires a license from Tylt LLC.
# Contact: hello@claimhawk.app | See LICENSE for terms.

"""
Link Router Data from Expert Datasets

Dynamically discovers the latest preprocessed dataset for each expert
by scanning the preprocessed/ directory and using folder timestamps.
NO REPROCESSING - just links the existing .pt files.

The training script handles label remapping at runtime.

Reads configuration from config/dataset.yaml for:
- Per-adapter sample counts
- Test samples per adapter (held out from training data)

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

# Load config from dataset.yaml
CONFIG_PATH = Path(__file__).parent.parent / "config" / "dataset.yaml"


def load_dataset_config_local() -> dict:
    """Load dataset configuration from local config/dataset.yaml.

    Called from local entrypoint before remote execution.
    """
    import yaml

    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    # Fallback defaults if config not found
    return {
        "test_samples_per_adapter": 25,
        "adapters": {name: {"count": 500} for name in EXPERT_LABELS.keys()},
    }


# Default test samples
DEFAULT_TEST_SAMPLES = 25

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
    test_samples: int = None,
    expert_overrides: dict = None,
    adapter_configs: dict = None,
):
    """
    Create a manifest file that points to expert preprocessed datasets.

    Dynamically discovers the latest preprocessed dataset for each expert
    by scanning the preprocessed/ directory and selecting based on timestamp.

    Config values are loaded locally and passed as parameters.

    CLI overrides (expert_overrides) take precedence over config values.

    Args:
        output_name: Name for the manifest file
        test_samples: Number of held-out test samples per expert (default: 25)
        expert_overrides: Dict of expert_name -> sample_count for custom per-expert limits
        adapter_configs: Per-adapter config from config/dataset.yaml (loaded locally)
    """
    expert_volume.reload()
    moe_volume.reload()

    # Use passed config or empty dict
    if adapter_configs is None:
        adapter_configs = {}

    if output_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"router_linked_{timestamp}"

    # Use default if not specified via CLI
    if test_samples is None:
        test_samples = DEFAULT_TEST_SAMPLES

    print(f"\n{'='*70}")
    print("Linking Expert Datasets for Router Training")
    print(f"{'='*70}")
    print(f"Output: {output_name}")
    print(f"Config: {CONFIG_PATH}")
    print(f"Test samples per expert: {test_samples}")
    print(f"Per-adapter counts from config:")

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

    # Build effective counts dict: CLI overrides > config > default (500)
    effective_counts = {}
    for expert_name in EXPERT_LABELS.keys():
        # Priority: CLI override > config > 500
        if expert_overrides and expert_name in expert_overrides:
            effective_counts[expert_name] = expert_overrides[expert_name]
        elif expert_name in adapter_configs:
            effective_counts[expert_name] = adapter_configs[expert_name].get("count", 500)
        else:
            effective_counts[expert_name] = 500  # Fallback default
        print(f"  {expert_name}: {effective_counts[expert_name]}")

    # Collect info about each expert dataset
    manifest = {
        "name": output_name,
        "created": datetime.now().isoformat(),
        "type": "linked",
        "config_path": str(CONFIG_PATH),
        "test_samples_per_expert": test_samples,
        "expert_counts": effective_counts,
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
        # Use effective count for this expert (from config or CLI override)
        expert_max = effective_counts[expert_name]
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

        limited_note = f" (limited from {available_train})" if available_train > expert_max else ""
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
    test_samples: int = None,
    desktop: int = None,
    appointment: int = None,
    calendar: int = None,
    claim_window: int = None,
    ocr: int = None,
    login_window: int = None,
    chart_screen: int = None,
):
    """Create linked router dataset manifest from config/dataset.yaml.

    Per-adapter counts come from config/dataset.yaml by default.
    CLI overrides (--desktop, --appointment, etc.) take precedence over config.

    Args:
        output_name: Name for the manifest
        test_samples: Held-out test samples per expert (from config if not specified)
        desktop: Override sample count for desktop expert
        appointment: Override sample count for appointment expert
        calendar: Override sample count for calendar expert
        claim_window: Override sample count for claim-window expert
        ocr: Override sample count for ocr expert
        login_window: Override sample count for login-window expert
        chart_screen: Override sample count for chart-screen expert
    """
    # Load config locally (before remote call)
    config = load_dataset_config_local()
    adapter_configs = config.get("adapters", {})

    # Use config value for test_samples if not specified via CLI
    if test_samples is None:
        test_samples = config.get("test_samples_per_adapter", DEFAULT_TEST_SAMPLES)

    print(f"Config loaded from: {CONFIG_PATH}")
    print(f"Test samples per expert: {test_samples}")
    print(f"Per-adapter counts from config:")
    for name, cfg in adapter_configs.items():
        print(f"  {name}: {cfg.get('count', 500)}")

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
        test_samples=test_samples,
        expert_overrides=expert_overrides if expert_overrides else None,
        adapter_configs=adapter_configs,
    )

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")
    print(f"Manifest: {result['name']}")
    print(f"Config: {result.get('config_path', 'N/A')}")
    if expert_overrides:
        print(f"CLI overrides: {expert_overrides}")
    print(f"Train samples: {result['totals']['train']}")
    print(f"Val samples: {result['totals']['val']}")
    print(f"Test samples (held-out): {result['totals']['test']}")
    print(f"\nPer-expert breakdown:")
    for name, info in result["experts"].items():
        count = result.get("expert_counts", {}).get(name, "?")
        print(f"  {name}: {info['train_count']} train, {info['test_count']} test (config: {count})")
    print(f"\nUse with: modal run modal/train_router.py --manifest {result['name']}")
