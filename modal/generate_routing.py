#!/usr/bin/env python3
# Copyright (c) 2025 Tylt LLC. All rights reserved.
# Licensed for research use only. Commercial use requires a license from Tylt LLC.
# Contact: hello@claimhawk.app | See LICENSE for terms.

"""Generate balanced routing dataset on Modal.

Reads from claimhawk-training-data volume, generates balanced routing dataset,
saves to moe-lora-data volume. Configuration is loaded from config/dataset.yaml.

Usage:
    modal run modal/generate_routing.py --seed 42
"""

import json
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import modal

# =============================================================================
# CENTRALIZED CONFIGURATION
# =============================================================================
# Volume names and adapter info are loaded from config/adapters.yaml via the SDK.
# Users can customize these by editing the YAML file.

try:
    from sdk.modal_compat import get_volume_name, get_adapter_labels
    LORA_VOLUME_NAME = get_volume_name("lora_training")
    MOE_VOLUME_NAME = get_volume_name("moe_data")
    ADAPTER_LABELS = get_adapter_labels()
except ImportError:
    # Fallback for Modal remote execution
    LORA_VOLUME_NAME = "claimhawk-lora-training"
    MOE_VOLUME_NAME = "moe-lora-data"
    ADAPTER_LABELS = {
        "calendar": 0,
        "claim-window": 1,
        "provider-select": 2,
        "chandra": 3,
        "desktop": 4,
        "appointment": 5,
        "login-window": 6,
        "chart-screen": 7,
    }


def check_script_invocation() -> None:
    """Check if script was invoked from shell script, print warning if not.

    Scripts should be run via ./scripts/generate_dataset.sh to ensure the full
    pipeline (generate + preprocess + train) is executed. Running modal/xxx.py
    directly will skip the subsequent stages.

    The shell script should set FROM_SCRIPT=1 before calling the modal script.
    """
    if os.environ.get("FROM_SCRIPT") != "1":
        print("")
        print("*" * 60)
        print("*" + " " * 58 + "*")
        print("*" + "  WARNING: Running modal script directly!".center(56) + "  *")
        print("*" + " " * 58 + "*")
        print("*" + "  Use ./scripts/generate_dataset.sh for the full pipeline:".center(56) + "  *")
        print("*" + "  - Dataset generation".center(56) + "  *")
        print("*" + "  - Preprocessing".center(56) + "  *")
        print("*" + "  - Training".center(56) + "  *")
        print("*" + " " * 58 + "*")
        print("*" + "  Run: ./scripts/generate_dataset.sh".center(56) + "  *")
        print("*" + "  Or:  ./scripts/generate_dataset.sh --dry  (no training)".center(56) + "  *")
        print("*" + " " * 58 + "*")
        print("*" * 60)
        print("")
        sys.stderr.flush()
        sys.stdout.flush()

app = modal.App("routing-dataset-generator")

# Volumes (using centralized config)
training_data_volume = modal.Volume.from_name(LORA_VOLUME_NAME, create_if_missing=False)
moe_volume = modal.Volume.from_name(MOE_VOLUME_NAME, create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.12").pip_install("pyyaml")


# Dataset config - loaded from config/dataset.yaml locally, with fallback for Modal
# Config structure:
#   splits: {train, val, test} - ratios applied to each adapter's count
#   adapters.<name>: {count, source (optional)} - set count to 0 to disable
# Dataset names are resolved dynamically by finding the latest matching dataset

DATASET_CONFIG = {
    "splits": {"train": 0.8, "val": 0.2},
}
TEST_SAMPLES_PER_ADAPTER = 100

ADAPTER_CONFIGS = {
    "calendar": {"count": 1000},
    "claim-window": {"count": 1000},
    "provider-select": {"count": 1000},
    "appointment": {"count": 1000},
    "login-window": {"count": 1000},
    "desktop": {"count": 1000},
    "chart-screen": {"count": 1000},
    "chandra": {"count": 1000, "source": "ocr-generated"},
}

# Try to load from config/dataset.yaml (works locally, fails silently on Modal)
try:
    import os as _os
    import yaml as _yaml
    _config_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "config", "dataset.yaml")
    with open(_config_path) as _f:
        _config = _yaml.safe_load(_f)
    # Load splits
    DATASET_CONFIG = {
        "splits": _config.get("splits", DATASET_CONFIG["splits"]),
    }
    # Load adapter configs
    if "adapters" in _config:
        ADAPTER_CONFIGS = {}
        for name, info in _config["adapters"].items():
            ADAPTER_CONFIGS[name] = {
                "count": info.get("count", 1000),
                "source": info.get("source"),
            }
    del _os, _yaml, _config_path, _f, _config
except (FileNotFoundError, ImportError):
    pass  # Use hardcoded fallback above


def find_latest_dataset(adapter_name: str, datasets_dir: Path) -> str | None:
    """Find the latest dataset for an adapter by scanning the datasets directory.

    Dataset names follow pattern: {adapter_name}-{user}-{timestamp}
    where timestamp is YYYYMMDD_HHMMSS. Returns the one with latest timestamp.
    """
    import re

    if not datasets_dir.exists():
        return None

    # Match datasets starting with adapter name (allowing for various separators)
    # e.g., calendar-mike-20251203, calendar--mike--20251203, appointment-user-20251203
    pattern = re.compile(rf"^{re.escape(adapter_name)}[-_].*(\d{{8}}_\d{{6}})$")

    matches = []
    for d in datasets_dir.iterdir():
        if d.is_dir():
            match = pattern.match(d.name)
            if match:
                timestamp = match.group(1)
                matches.append((timestamp, d.name))

    if not matches:
        return None

    # Sort by timestamp descending, return latest
    matches.sort(key=lambda x: x[0], reverse=True)
    return matches[0][1]


# OCR prompt templates for Chandra routing
OCR_PROMPT_TEMPLATES = [
    "Read the text in this image and return it using an ocr tool_call",
    "Read the text from this cropped screenshot",
    "Read all visible text in this image",
    "Read and extract the text content from this image",
    "Read what's written in this image and return it via ocr tool_call",
    "Please read the text in this image and return it",
    "I need you to read the text from this screenshot",
    "Can you read the text shown in this image?",
    "Look at this image and read the text content",
    "Here is a screenshot - read the text and return it",
    "This is a cropped region - read the text from it",
    "Here is a screenshot that has been cropped to just the region we want. Read the text from the image and return it using an ocr tool_call",
    "This image contains text that needs to be read",
    "Extract the text from this image - read it carefully",
    "I've cropped this screenshot to the area you need to read",
    "Extract the text from this image",
    "Extract all text content from this screenshot",
    "Please extract the text shown in this cropped image",
    "I need the text extracted from this image",
    "Perform OCR on this image and return the text",
    "Run OCR on this cropped screenshot",
    "Use OCR to get the text from this image",
    "OCR this image and return the result",
    "Apply OCR to extract the text content",
    "Transcribe the text in this image",
    "Transcribe what you see in this screenshot",
    "Please transcribe the text content from this image",
    "Get the text from this image",
    "Get all text content visible in this screenshot",
    "Return the text shown in this image",
    "Return the text content from this cropped region",
    "Read the procedure codes from this image",
    "Extract the patient information shown in this screenshot",
    "Read the text from this claim form section",
    "Extract the provider details from this cropped image",
    "Read the appointment details shown here",
    "Get the insurance information from this image",
    "Extract the diagnosis codes visible in this screenshot",
]


def find_ocr_folders(datasets_dir: Path) -> list[Path]:
    """Find all OCR folders in dataset directories.

    Scans /training-data/datasets/*/ocr/images/ for OCR images generated
    by the screen generators.
    """
    ocr_folders = []
    if not datasets_dir.exists():
        return ocr_folders

    for dataset_dir in datasets_dir.iterdir():
        if dataset_dir.is_dir():
            ocr_images_dir = dataset_dir / "ocr" / "images"
            if ocr_images_dir.exists():
                ocr_folders.append(ocr_images_dir)

    return ocr_folders


def generate_ocr_samples(
    rng: random.Random,
    count: int,
    images_dir: Path,
    datasets_dir: Path,
) -> list[dict]:
    """Generate OCR routing samples for Chandra using OCR images from generators.

    Scans all dataset OCR folders in /training-data/datasets/*/ocr/images/
    and collects images for use in chandra routing training.
    """
    templates = list(OCR_PROMPT_TEMPLATES)
    rng.shuffle(templates)

    # Find all OCR folders from generators
    ocr_folders = find_ocr_folders(datasets_dir)
    if not ocr_folders:
        print(f"  WARNING: No OCR folders found in {datasets_dir}/*/ocr/images/")
        return []

    # Collect all OCR images from all folders
    ocr_images = []
    for folder in ocr_folders:
        # Check both jpg and png
        ocr_images.extend(sorted(folder.glob("*.jpg")))
        ocr_images.extend(sorted(folder.glob("*.png")))

    if not ocr_images:
        print(f"  WARNING: No OCR images found in {len(ocr_folders)} OCR folders")
        return []

    print(f"  Found {len(ocr_images)} OCR images in {len(ocr_folders)} folders")

    # Shuffle images for variety
    rng.shuffle(ocr_images)

    # Copy OCR images to output images dir (with unique names to avoid collisions)
    copied_images = []
    for i, img in enumerate(ocr_images):
        # Use index prefix to ensure unique names across folders
        dst_name = f"ocr_{i:05d}_{img.name}"
        dst = images_dir / dst_name
        if not dst.exists():
            shutil.copy2(img, dst)
        copied_images.append(dst_name)

    samples = []
    for i in range(count):
        prompt = templates[i % len(templates)]
        # Cycle through available OCR images
        img_name = copied_images[i % len(copied_images)]
        sample = {
            "id": f"ocr_{i:04d}",
            "image": f"images/{img_name}",
            "conversations": [
                # No system prompt - preprocessor will add it
                {"from": "human", "value": f"<image>\n{prompt}"},
                {"from": "gpt", "value": "chandra"},
            ],
            "metadata": {
                "adapter": "chandra",
                "label": ADAPTER_LABELS["chandra"],
            },
        }
        samples.append(sample)

    return samples


@app.function(
    image=image,
    volumes={
        "/training-data": training_data_volume,
        "/moe-data": moe_volume,
    },
    timeout=3600,
)
def generate_routing_dataset(
    seed: int = 42,
):
    """Generate routing dataset from adapter sources + OCR.

    Uses config/dataset.yaml to determine:
    - splits: {train, val} ratios applied to each adapter's count
    - adapters: {name: {count}} - set count to 0 to disable
    - Test is always 100 samples per adapter

    Creates three separate splits:
    - train.jsonl: Training data
    - val.jsonl: Validation data for early stopping
    - test.jsonl: Held-out test data (100 per adapter)
    """
    # Load config
    splits = DATASET_CONFIG["splits"]
    adapter_configs = ADAPTER_CONFIGS
    datasets_dir = Path("/training-data/datasets")

    # Find latest dataset for each enabled adapter (count > 0, not ocr-generated)
    adapter_datasets = {}
    for name, cfg in adapter_configs.items():
        count = cfg.get("count", 0)
        if count <= 0:
            continue
        if cfg.get("source") == "ocr-generated":
            continue  # Skip chandra - generated separately
        dataset_name = find_latest_dataset(name, datasets_dir)
        if dataset_name:
            adapter_datasets[name] = dataset_name
        else:
            print(f"WARNING: No dataset found for adapter '{name}'")

    rng = random.Random(seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"/moe-data/datasets/routing_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Generating Routing Dataset from config/dataset.yaml")
    print(f"{'='*70}")
    print(f"Output: {output_dir}")
    print(f"Splits: train={splits['train']:.0%}, val={splits['val']:.0%}, test={TEST_SAMPLES_PER_ADAPTER} fixed")
    print(f"\nAdapter counts:")
    for name, cfg in adapter_configs.items():
        count = cfg.get("count", 0)
        if count > 0:
            train_n = int(count * splits["train"])
            val_n = int(count * splits["val"])
            src = cfg.get("source") or adapter_datasets.get(name, "not found")
            print(f"  {name}: {count} total -> train={train_n}, val={val_n}, test={TEST_SAMPLES_PER_ADAPTER} (from {src})")
    print(f"{'='*70}\n")

    all_train_samples = []
    all_val_samples = []
    all_test_samples = []

    for adapter_name, dataset_name in adapter_datasets.items():
        # Get per-adapter config
        adapter_cfg = adapter_configs.get(adapter_name, {})
        count = adapter_cfg.get("count", 1000)
        train_target = int(count * splits["train"])
        val_target = int(count * splits["val"])
        test_target = TEST_SAMPLES_PER_ADAPTER

        label = ADAPTER_LABELS[adapter_name]
        dataset_path = Path(f"/training-data/datasets/{dataset_name}/train.jsonl")
        source_images_dir = Path(f"/training-data/datasets/{dataset_name}/images")

        print(f"Loading {adapter_name} from {dataset_name}...")

        if not dataset_path.exists():
            print(f"  ERROR: {dataset_path} not found!")
            continue

        # Group tasks by image
        tasks_by_image = {}
        with open(dataset_path) as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    img = sample.get("image", "")
                    if img:
                        if img not in tasks_by_image:
                            tasks_by_image[img] = []
                        tasks_by_image[img].append(sample)

        # Shuffle image order
        image_list = list(tasks_by_image.keys())
        rng.shuffle(image_list)

        total_tasks = sum(len(tasks) for tasks in tasks_by_image.values())
        print(f"  Total images: {len(image_list)}, Total tasks: {total_tasks}")

        def convert_to_routing_sample(task, adapter_name, label):
            """Convert task to routing format: keep system+human, replace gpt with adapter name."""
            # Only keep essential fields
            task_copy = {
                "id": task.get("id", ""),
                "image": task.get("image", ""),
                "metadata": {
                    "adapter": adapter_name,
                    "label": label,
                },
            }

            # Filter conversations: keep only human, replace gpt with adapter name
            # Do NOT include system prompt - preprocessor will add it
            if "conversations" in task:
                new_convs = []
                for conv in task["conversations"]:
                    if conv["from"] == "system":
                        # Skip system prompt - preprocessor adds it
                        continue
                    elif conv["from"] == "human":
                        value = conv["value"]
                        # Fix desktop prompts: desktop icons require double-click, not left click
                        if adapter_name == "desktop":
                            value = value.replace("Left click", "Double click")
                            value = value.replace("left click", "double click")
                            value = value.replace("Click on", "Double click on")
                            value = value.replace("click on", "double click on")
                        new_convs.append({"from": "human", "value": value})
                    elif conv["from"] == "gpt":
                        # Replace with adapter name as the response
                        new_convs.append({"from": "gpt", "value": adapter_name})
                        break  # Only keep first response
                task_copy["conversations"] = new_convs

            return task_copy

        # Collect samples - iterate through images, grab ALL tasks from each image
        # Split into train, val, test using separate image pools (no leakage)
        train_samples = []
        val_samples = []
        test_samples = []
        images_used = {"train": [], "val": [], "test": []}

        # First pass: collect train samples
        for img in image_list:
            if len(train_samples) >= train_target:
                break
            for task in tasks_by_image[img]:
                train_samples.append(convert_to_routing_sample(task, adapter_name, label))
            images_used["train"].append(img)

        # Second pass: collect val samples from remaining images
        for img in image_list:
            if img in images_used["train"]:
                continue
            if len(val_samples) >= val_target:
                break
            for task in tasks_by_image[img]:
                val_samples.append(convert_to_routing_sample(task, adapter_name, label))
            images_used["val"].append(img)

        # Third pass: collect test samples from remaining images
        for img in image_list:
            if img in images_used["train"] or img in images_used["val"]:
                continue
            if len(test_samples) >= test_target:
                break
            for task in tasks_by_image[img]:
                test_samples.append(convert_to_routing_sample(task, adapter_name, label))
            images_used["test"].append(img)

        all_train_samples.extend(train_samples)
        all_val_samples.extend(val_samples)
        all_test_samples.extend(test_samples)

        print(f"  {adapter_name:20s} -> train={len(train_samples):4d}, val={len(val_samples):4d}, test={len(test_samples):4d}")

        # Copy images
        images_copied = set()
        for sample in train_samples + val_samples + test_samples:
            img_path = sample.get("image", "")
            if img_path:
                img_name = Path(img_path).name
                if img_name in images_copied:
                    sample["image"] = f"images/{img_name}"
                    continue

                src = source_images_dir / img_name
                if not src.exists():
                    src = Path(f"/training-data/datasets/{dataset_name}") / img_path
                if src.exists():
                    dst = images_dir / img_name
                    if not dst.exists():
                        shutil.copy2(src, dst)
                    images_copied.add(img_name)
                sample["image"] = f"images/{img_name}"

    # Generate OCR samples for Chandra (if count > 0)
    chandra_cfg = adapter_configs.get("chandra", {})
    chandra_count = chandra_cfg.get("count", 0)
    if chandra_count > 0:
        chandra_train = int(chandra_count * splits["train"])
        chandra_val = int(chandra_count * splits["val"])
        chandra_test = TEST_SAMPLES_PER_ADAPTER

        print(f"\nGenerating OCR samples for chandra...")
        # Use OCR images from generator datasets (*/ocr/images/)
        ocr_train = generate_ocr_samples(rng, chandra_train, images_dir, datasets_dir)
        ocr_val = generate_ocr_samples(rng, chandra_val, images_dir, datasets_dir)
        ocr_test = generate_ocr_samples(rng, chandra_test, images_dir, datasets_dir)

        all_train_samples.extend(ocr_train)
        all_val_samples.extend(ocr_val)
        all_test_samples.extend(ocr_test)

        print(f"  {'chandra':20s} -> train={len(ocr_train):4d}, val={len(ocr_val):4d}, test={len(ocr_test):4d}")

    # Shuffle all splits
    rng.shuffle(all_train_samples)
    rng.shuffle(all_val_samples)
    rng.shuffle(all_test_samples)

    print(f"\nTotal samples:")
    print(f"  Train: {len(all_train_samples)} (for training)")
    print(f"  Val: {len(all_val_samples)} (for early stopping during training)")
    print(f"  Test: {len(all_test_samples)} (held-out for final accuracy)")

    # Check balance
    print(f"\nLabel distribution (train):")
    dist = {}
    for s in all_train_samples:
        a = s["metadata"]["adapter"]
        dist[a] = dist.get(a, 0) + 1
    for a, c in sorted(dist.items()):
        print(f"  {a}: {c}")

    # Save
    def save_jsonl(path, data):
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"  Saved {len(data)} to {path.name}")

    all_data = all_train_samples + all_val_samples + all_test_samples
    save_jsonl(output_dir / "data.jsonl", all_data)
    save_jsonl(output_dir / "train.jsonl", all_train_samples)
    save_jsonl(output_dir / "val.jsonl", all_val_samples)
    save_jsonl(output_dir / "test.jsonl", all_test_samples)

    # Metadata
    adapters_meta = {name: {"label": ADAPTER_LABELS[name], "source": ds} for name, ds in adapter_datasets.items()}
    adapters_meta["chandra"] = {"label": ADAPTER_LABELS["chandra"], "source": "generated"}

    metadata = {
        "dataset_name": "routing",
        "created_at": datetime.now().isoformat(),
        "seed": seed,
        "splits": splits,
        "adapters": adapters_meta,
        "samples": {"train": len(all_train_samples), "val": len(all_val_samples), "test": len(all_test_samples)},
        "label_distribution_train": dist,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Commit volume
    moe_volume.commit()

    print(f"\n{'='*70}")
    print(f"Dataset saved to: datasets/routing_{timestamp}")
    print(f"{'='*70}\n")

    return f"routing_{timestamp}"


@app.local_entrypoint()
def main(
    seed: int = 42,
):
    import subprocess
    from pathlib import Path

    # Check if invoked from shell script - warn if not
    check_script_invocation()

    dataset_name = generate_routing_dataset.remote(seed=seed)
    print(f"\nGenerated: {dataset_name}")

    # Download to local ./datasets/ for inspection
    local_dir = Path(f"datasets/{dataset_name}")
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading to {local_dir}...")
    subprocess.run(
        ["uvx", "modal", "volume", "get", "moe-lora-data", f"datasets/{dataset_name}", str(local_dir.parent)],
        check=True,
    )
    print(f"Downloaded to: {local_dir}")

    print(f"\nNext: ./scripts/preprocess.sh --dataset-name datasets/{dataset_name}")
