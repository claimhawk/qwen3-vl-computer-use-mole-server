#!/usr/bin/env python3
# Copyright (c) 2025 Tylt LLC. All rights reserved.
# Licensed for research use only. Commercial use requires a license from Tylt LLC.
# Contact: hello@claimhawk.app | See LICENSE for terms.

# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml>=6.0"]
# ///

"""Generate MoE routing dataset locally.

Reads from generator project datasets and creates a balanced routing dataset.
The dataset is saved locally, then uploaded to Modal via scripts/upload.sh.

Usage:
    python generator.py
    python generator.py --config config/dataset.yaml
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import yaml

# =============================================================================
# CONFIGURATION
# =============================================================================

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

# Map adapter names to their generator project directories
ADAPTER_GENERATORS = {
    "calendar": "calendar-generator",
    "claim-window": "claim-window-generator",
    "provider-select": "claim-window-generator",  # provider-select is in claim-window-generator
    "appointment": "appointment-generator",
    "login-window": "login-window-generator",
    "desktop": "desktop-generator",
    "chart-screen": "chart-screen-generator",
}

# Default config
DEFAULT_CONFIG = {
    "splits": {"train": 0.8, "val": 0.2},
    "test_samples_per_adapter": 100,
    "adapters": {
        "calendar": {"count": 1000},
        "claim-window": {"count": 1000},
        "provider-select": {"count": 1000},
        "appointment": {"count": 1000},
        "login-window": {"count": 1000},
        "desktop": {"count": 1000},
        "chart-screen": {"count": 1000},
        "chandra": {"count": 1000, "source": "ocr-generated"},
    },
}

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


def get_researcher_name() -> str | None:
    """Get researcher name from .researcher file if it exists."""
    researcher_file = Path(".researcher")
    if researcher_file.exists():
        content = researcher_file.read_text().strip()
        for line in content.split("\n"):
            if line.startswith("Name:"):
                return line.split(":", 1)[1].strip().lower()
    return None


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config
    return DEFAULT_CONFIG


def find_generators_root() -> Path:
    """Find the generators root directory."""
    # Look for ../generators relative to this script
    script_dir = Path(__file__).parent
    generators_root = script_dir.parent / "generators"
    if generators_root.exists():
        return generators_root

    # Fallback: look in common locations
    for path in [
        Path.home() / "development/claimhawk/projects/generators",
        Path("/Users/michaeloneal/development/claimhawk/projects/generators"),
    ]:
        if path.exists():
            return path

    raise FileNotFoundError("Cannot find generators root directory")


def find_latest_dataset(adapter_name: str, generators_root: Path) -> tuple[str, Path] | None:
    """Find the latest dataset for an adapter in local generator projects.

    Returns (dataset_name, dataset_path) or None if not found.
    """
    import re

    generator_name = ADAPTER_GENERATORS.get(adapter_name)
    if not generator_name:
        return None

    datasets_dir = generators_root / generator_name / "datasets"
    if not datasets_dir.exists():
        return None

    # Match datasets starting with adapter name
    pattern = re.compile(rf"^{re.escape(adapter_name)}[-_].*(\d{{8}}_\d{{6}})$")

    matches = []
    for d in datasets_dir.iterdir():
        if d.is_dir():
            match = pattern.match(d.name)
            if match:
                timestamp = match.group(1)
                matches.append((timestamp, d.name, d))

    if not matches:
        return None

    # Sort by timestamp descending, return latest
    matches.sort(key=lambda x: x[0], reverse=True)
    return matches[0][1], matches[0][2]


def convert_to_routing_sample(
    task: dict,
    adapter_name: str,
    label: int,
    source_dataset: str,
) -> dict:
    """Convert a task to routing format.

    Image paths are stored as {dataset-name}/images/{filename} so they can be
    resolved on Modal as /training-data/datasets/{dataset-name}/images/{filename}.
    """
    # Get original image path and convert to dataset-relative path
    orig_image = task.get("image", "")
    if orig_image:
        # Extract just the filename
        img_name = Path(orig_image).name
        # Store as {dataset}/images/{filename} for Modal resolution
        image_path = f"{source_dataset}/images/{img_name}"
    else:
        image_path = ""

    sample = {
        "id": task.get("id", ""),
        "image": image_path,
        "metadata": {
            "adapter": adapter_name,
            "label": label,
            "source_dataset": source_dataset,
        },
    }

    # Filter conversations: keep only human, replace gpt with adapter name
    if "conversations" in task:
        new_convs = []
        for conv in task["conversations"]:
            if conv["from"] == "system":
                continue  # Skip system prompt
            elif conv["from"] == "human":
                value = conv["value"]
                # Fix desktop prompts: desktop icons require double-click
                if adapter_name == "desktop":
                    value = value.replace("Left click", "Double click")
                    value = value.replace("left click", "double click")
                    value = value.replace("Click on", "Double click on")
                    value = value.replace("click on", "double click on")
                new_convs.append({"from": "human", "value": value})
            elif conv["from"] == "gpt":
                new_convs.append({"from": "gpt", "value": adapter_name})
                break  # Only keep first response
        sample["conversations"] = new_convs

    return sample


def generate_ocr_samples(
    rng: random.Random,
    count: int,
    generators_root: Path,
) -> list[dict]:
    """Generate OCR routing samples for Chandra.

    Image paths are stored as {dataset-name}/ocr/images/{filename} so they can be
    resolved on Modal as /training-data/datasets/{dataset-name}/ocr/images/{filename}.
    """
    templates = list(OCR_PROMPT_TEMPLATES)
    rng.shuffle(templates)

    # Find all OCR folders and collect images with their dataset info
    ocr_images: list[tuple[str, str]] = []  # (dataset_name, relative_path)

    for generator_dir in generators_root.iterdir():
        if not generator_dir.is_dir() or not generator_dir.name.endswith("-generator"):
            continue

        datasets_dir = generator_dir / "datasets"
        if not datasets_dir.exists():
            continue

        for dataset_dir in datasets_dir.iterdir():
            if dataset_dir.is_dir():
                ocr_images_dir = dataset_dir / "ocr" / "images"
                if ocr_images_dir.exists():
                    dataset_name = dataset_dir.name
                    for img_file in ocr_images_dir.glob("*.png"):
                        ocr_images.append((dataset_name, f"ocr/images/{img_file.name}"))
                    for img_file in ocr_images_dir.glob("*.jpg"):
                        ocr_images.append((dataset_name, f"ocr/images/{img_file.name}"))

    if not ocr_images:
        print("  WARNING: No OCR images found in generator datasets")
        return []

    print(f"  Found {len(ocr_images)} OCR images")

    # Shuffle and select only what we need
    rng.shuffle(ocr_images)
    selected_images = ocr_images[:count] if len(ocr_images) >= count else ocr_images

    # Generate samples - reference images by their source dataset path
    samples = []
    for i in range(count):
        prompt = templates[i % len(templates)]
        dataset_name, img_path = selected_images[i % len(selected_images)]
        # Store as {dataset}/ocr/images/{filename} for Modal resolution
        image_path = f"{dataset_name}/{img_path}"
        sample = {
            "id": f"ocr_{i:04d}",
            "image": image_path,
            "conversations": [
                {"from": "human", "value": f"<image>\n{prompt}"},
                {"from": "gpt", "value": "chandra"},
            ],
            "metadata": {
                "adapter": "chandra",
                "label": ADAPTER_LABELS["chandra"],
                "source_dataset": dataset_name,
            },
        }
        samples.append(sample)

    return samples


def generate_routing_dataset(
    config: dict,
    output_dir: Path,
    generators_root: Path,
    seed: int = 42,
) -> None:
    """Generate the routing dataset."""
    rng = random.Random(seed)

    splits = config.get("splits", DEFAULT_CONFIG["splits"])
    test_per_adapter = config.get("test_samples_per_adapter", 100)
    adapter_configs = config.get("adapters", DEFAULT_CONFIG["adapters"])

    # Create output directory (no images dir - we reference source datasets)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Generating MoE Routing Dataset")
    print(f"{'='*70}")
    print(f"Output: {output_dir}")
    print(f"Splits: train={splits['train']:.0%}, val={splits['val']:.0%}, test={test_per_adapter} fixed")
    print(f"\nAdapter counts:")

    # Find datasets for each adapter
    adapter_datasets = {}
    for name, cfg in adapter_configs.items():
        count = cfg.get("count", 0)
        if count <= 0:
            continue
        if cfg.get("source") == "ocr-generated":
            print(f"  {name}: {count} total (from ocr-generated)")
            continue

        result = find_latest_dataset(name, generators_root)
        if result:
            dataset_name, dataset_path = result
            adapter_datasets[name] = (dataset_name, dataset_path)
            train_n = int(count * splits["train"])
            val_n = int(count * splits["val"])
            print(f"  {name}: {count} total -> train={train_n}, val={val_n}, test={test_per_adapter} (from {dataset_name})")
        else:
            print(f"  WARNING: No dataset found for adapter '{name}'")

    print(f"{'='*70}\n")

    all_train_samples = []
    all_val_samples = []
    all_test_samples = []

    # Process each adapter dataset
    for adapter_name, (dataset_name, dataset_path) in adapter_datasets.items():
        adapter_cfg = adapter_configs.get(adapter_name, {})
        count = adapter_cfg.get("count", 1000)
        train_target = int(count * splits["train"])
        val_target = int(count * splits["val"])
        test_target = test_per_adapter

        label = ADAPTER_LABELS[adapter_name]
        train_jsonl = dataset_path / "train.jsonl"
        source_images_dir = dataset_path / "images"

        print(f"Loading {adapter_name} from {dataset_name}...")

        if not train_jsonl.exists():
            print(f"  ERROR: {train_jsonl} not found!")
            continue

        # Group tasks by image
        tasks_by_image = {}
        with open(train_jsonl) as f:
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

        # Split by image (no leakage)
        train_samples = []
        val_samples = []
        test_samples = []
        images_used = {"train": [], "val": [], "test": []}

        # First pass: train
        for img in image_list:
            if len(train_samples) >= train_target:
                break
            for task in tasks_by_image[img]:
                train_samples.append(convert_to_routing_sample(task, adapter_name, label, dataset_name))
            images_used["train"].append(img)

        # Second pass: val
        for img in image_list:
            if img in images_used["train"]:
                continue
            if len(val_samples) >= val_target:
                break
            for task in tasks_by_image[img]:
                val_samples.append(convert_to_routing_sample(task, adapter_name, label, dataset_name))
            images_used["val"].append(img)

        # Third pass: test
        for img in image_list:
            if img in images_used["train"] or img in images_used["val"]:
                continue
            if len(test_samples) >= test_target:
                break
            for task in tasks_by_image[img]:
                test_samples.append(convert_to_routing_sample(task, adapter_name, label, dataset_name))
            images_used["test"].append(img)

        all_train_samples.extend(train_samples)
        all_val_samples.extend(val_samples)
        all_test_samples.extend(test_samples)

        print(f"  {adapter_name:20s} -> train={len(train_samples):4d}, val={len(val_samples):4d}, test={len(test_samples):4d}")

    # Generate OCR samples for Chandra
    chandra_cfg = adapter_configs.get("chandra", {})
    chandra_count = chandra_cfg.get("count", 0)
    if chandra_count > 0:
        chandra_train = int(chandra_count * splits["train"])
        chandra_val = int(chandra_count * splits["val"])
        chandra_test = test_per_adapter

        print(f"\nGenerating OCR samples for chandra...")
        ocr_train = generate_ocr_samples(rng, chandra_train, generators_root)
        ocr_val = generate_ocr_samples(rng, chandra_val, generators_root)
        ocr_test = generate_ocr_samples(rng, chandra_test, generators_root)

        all_train_samples.extend(ocr_train)
        all_val_samples.extend(ocr_val)
        all_test_samples.extend(ocr_test)

        print(f"  {'chandra':20s} -> train={len(ocr_train):4d}, val={len(ocr_val):4d}, test={len(ocr_test):4d}")

    # Shuffle all splits
    rng.shuffle(all_train_samples)
    rng.shuffle(all_val_samples)
    rng.shuffle(all_test_samples)

    print(f"\nTotal samples:")
    print(f"  Train: {len(all_train_samples)}")
    print(f"  Val: {len(all_val_samples)}")
    print(f"  Test: {len(all_test_samples)}")

    # Check balance
    print(f"\nLabel distribution (train):")
    dist = {}
    for s in all_train_samples:
        a = s["metadata"]["adapter"]
        dist[a] = dist.get(a, 0) + 1
    for a, c in sorted(dist.items()):
        print(f"  {a}: {c}")

    # Save JSONL files
    def save_jsonl(path: Path, data: list) -> None:
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"  Saved {len(data)} to {path.name}")

    all_data = all_train_samples + all_val_samples + all_test_samples
    save_jsonl(output_dir / "data.jsonl", all_data)
    save_jsonl(output_dir / "train.jsonl", all_train_samples)
    save_jsonl(output_dir / "val.jsonl", all_val_samples)
    save_jsonl(output_dir / "test.jsonl", all_test_samples)

    # Save metadata
    adapters_meta = {
        name: {"label": ADAPTER_LABELS[name], "source": ds_name}
        for name, (ds_name, _) in adapter_datasets.items()
    }
    adapters_meta["chandra"] = {"label": ADAPTER_LABELS["chandra"], "source": "ocr-generated"}

    metadata = {
        "dataset_name": "routing",
        "created_at": datetime.now().isoformat(),
        "seed": seed,
        "splits": splits,
        "adapters": adapters_meta,
        "samples": {
            "train": len(all_train_samples),
            "val": len(all_val_samples),
            "test": len(all_test_samples),
        },
        "label_distribution_train": dist,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Dataset saved to: {output_dir}")
    print(f"{'='*70}\n")


def main() -> None:
    """Run dataset generation."""
    parser = argparse.ArgumentParser(description="Generate MoE routing dataset")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/dataset.yaml"),
        help="Path to dataset config YAML",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    # Check if invoked from script
    if os.environ.get("FROM_SCRIPT") != "1":
        print("")
        print("*" * 60)
        print("*  WARNING: Running generator.py directly!".ljust(58) + "*")
        print("*  Use ./scripts/generate.sh for the full pipeline.".ljust(58) + "*")
        print("*" * 60)
        print("")

    # Load config
    config = load_config(args.config)

    # Find generators root
    generators_root = find_generators_root()
    print(f"Generators root: {generators_root}")

    # Build dataset name
    researcher = get_researcher_name()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if researcher:
        dataset_name = f"routing--{researcher}--{timestamp}"
    else:
        dataset_name = f"routing_{timestamp}"

    output_dir = Path("datasets") / dataset_name

    # Generate
    generate_routing_dataset(
        config=config,
        output_dir=output_dir,
        generators_root=generators_root,
        seed=args.seed,
    )

    print(f"Next: ./scripts/upload.sh datasets/{dataset_name}")


if __name__ == "__main__":
    main()
