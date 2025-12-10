#!/usr/bin/env python3
# Copyright (c) 2025 Tylt LLC. All rights reserved.
# Licensed for research use only. Commercial use requires a license from Tylt LLC.
# Contact: hello@claimhawk.app | See LICENSE for terms.

# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml>=6.0"]
# ///

"""Generate MoE routing dataset from deployed experts on Modal.

Queries Modal's training-history volume for the latest deployed dataset per expert,
downloads the datasets from claimhawk-lora-training, and creates a balanced
routing dataset. The dataset is saved locally, then uploaded to Modal.

Usage:
    python generator.py
    python generator.py --config config/dataset.yaml
"""

import argparse
import json
import os
import random
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import yaml

# =============================================================================
# CONFIGURATION - Load from adapters.yaml (single source of truth)
# =============================================================================


def _load_adapters_config() -> dict:
    """Load adapter configuration from adapters.yaml."""
    # Look for adapters.yaml in config/ relative to parent directory
    script_dir = Path(__file__).parent
    config_path = script_dir.parent / "config" / "adapters.yaml"
    if not config_path.exists():
        # Fallback to common locations
        for path in [
            Path.home() / "development/claimhawk/config/adapters.yaml",
            Path("/Users/michaeloneal/development/claimhawk/config/adapters.yaml"),
        ]:
            if path.exists():
                config_path = path
                break
    if not config_path.exists():
        raise FileNotFoundError(f"adapters.yaml not found at {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def _get_adapter_labels() -> dict[str, int]:
    """Get adapter labels from adapters.yaml."""
    config = _load_adapters_config()
    labels = {}
    for name, expert in config.get("experts", {}).items():
        if "label" in expert:
            labels[name] = expert["label"]
    return labels


def _get_adapter_generators() -> dict[str, str]:
    """Get adapter -> generator mapping from adapters.yaml."""
    config = _load_adapters_config()
    generators = {}
    for name, expert in config.get("experts", {}).items():
        gen = expert.get("generator")
        if gen:
            generators[name] = gen
    return generators


# Load from adapters.yaml
ADAPTER_LABELS = _get_adapter_labels()
ADAPTER_GENERATORS = _get_adapter_generators()

# Modal volume names (from adapters.yaml)
_config = _load_adapters_config()
TRAINING_HISTORY_VOLUME = _config.get("volumes", {}).get("training_history", {}).get("name", "claimhawk-training-history")
LORA_TRAINING_VOLUME = _config.get("volumes", {}).get("lora_training", {}).get("name", "claimhawk-lora-training")

# Default config - adapters from YAML with default counts
# 500 samples per adapter is sufficient for router training
# Test: 25 samples per adapter
DEFAULT_CONFIG = {
    "splits": {"train": 0.8, "val": 0.2},
    "test_samples_per_adapter": 25,
    "adapters": {
        name: {"count": 500} if name != "ocr" else {"count": 500, "source": "ocr-generated"}
        for name in ADAPTER_LABELS.keys()
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


def get_deployed_dataset_from_modal(adapter_name: str) -> str | None:
    """Get the latest deployed dataset name for an adapter from Modal training history.

    Returns dataset_name or None if not found.
    """
    try:
        # Download runs.jsonl from training history
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            temp_path = f.name

        result = subprocess.run(
            [
                "uvx", "modal", "volume", "get",
                TRAINING_HISTORY_VOLUME,
                f"/{adapter_name}/runs.jsonl",
                temp_path,
                "--force",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"  Warning: No training history for {adapter_name}")
            return None

        # Read the last line (most recent run)
        with open(temp_path) as f:
            lines = f.readlines()
            if not lines:
                return None
            last_run = json.loads(lines[-1])
            dataset_name = last_run.get("dataset_name")
            return dataset_name

    except Exception as e:
        print(f"  Warning: Failed to get training history for {adapter_name}: {e}")
        return None
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def download_dataset_from_modal(dataset_name: str, local_dir: Path) -> Path | None:
    """Download a dataset from Modal's lora-training volume.

    Returns the local path to the downloaded dataset or None if failed.
    """
    dataset_path = local_dir / dataset_name
    dataset_path.mkdir(parents=True, exist_ok=True)

    # Download train.jsonl
    train_path = dataset_path / "train.jsonl"
    result = subprocess.run(
        [
            "uvx", "modal", "volume", "get",
            LORA_TRAINING_VOLUME,
            f"/datasets/{dataset_name}/train.jsonl",
            str(train_path),
            "--force",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"  Error downloading {dataset_name}/train.jsonl: {result.stderr}")
        return None

    return dataset_path


def find_latest_dataset(
    adapter_name: str,
    generators_root: Path,
    cache_dir: Path | None = None,
) -> tuple[str, Path] | None:
    """Find the latest deployed dataset for an adapter from Modal.

    Queries training-history volume for the deployed dataset, then downloads
    from lora-training volume.

    Returns (dataset_name, dataset_path) or None if not found.
    """
    # Get dataset name from Modal training history
    dataset_name = get_deployed_dataset_from_modal(adapter_name)
    if not dataset_name:
        print(f"  No deployed dataset found for {adapter_name}")
        return None

    # Use cache dir if provided, else use temp directory
    if cache_dir is None:
        cache_dir = Path(tempfile.gettempdir()) / "moe-routing-datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check if already cached
    cached_path = cache_dir / dataset_name
    if cached_path.exists() and (cached_path / "train.jsonl").exists():
        return dataset_name, cached_path

    # Download from Modal
    dataset_path = download_dataset_from_modal(dataset_name, cache_dir)
    if dataset_path is None:
        return None

    return dataset_name, dataset_path


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
                # Use numeric label instead of adapter name for single-token output
                new_convs.append({"from": "gpt", "value": str(label)})
                break  # Only keep first response
        sample["conversations"] = new_convs

    return sample


def get_ocr_images_from_modal(deployed_datasets: dict[str, str]) -> list[tuple[str, str]]:
    """Get OCR image paths from deployed datasets on Modal.

    Returns list of (dataset_name, relative_path) tuples.
    """
    ocr_images: list[tuple[str, str]] = []

    for adapter_name, dataset_name in deployed_datasets.items():
        # List OCR images from this dataset on Modal
        result = subprocess.run(
            [
                "uvx", "modal", "volume", "ls",
                LORA_TRAINING_VOLUME,
                f"/datasets/{dataset_name}/ocr/images/",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            # No OCR folder for this dataset, that's OK
            continue

        # Parse the listing - each line is a full path like datasets/{name}/ocr/images/file.jpg
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            # Extract just the filename
            if "/" in line:
                filename = line.split("/")[-1]
                if filename.endswith((".jpg", ".png")):
                    ocr_images.append((dataset_name, f"ocr/images/{filename}"))

    return ocr_images


def generate_ocr_samples(
    rng: random.Random,
    count: int,
    deployed_datasets: dict[str, str],
) -> list[dict]:
    """Generate OCR routing samples from deployed datasets on Modal.

    Image paths are stored as {dataset-name}/ocr/images/{filename} so they can be
    resolved on Modal as /training-data/datasets/{dataset-name}/ocr/images/{filename}.
    """
    templates = list(OCR_PROMPT_TEMPLATES)
    rng.shuffle(templates)

    # Get OCR images from deployed datasets on Modal
    ocr_images = get_ocr_images_from_modal(deployed_datasets)

    if not ocr_images:
        print("  WARNING: No OCR images found in deployed datasets on Modal")
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
        # Use numeric label instead of adapter name for single-token output
        ocr_label = ADAPTER_LABELS["ocr"]
        sample = {
            "id": f"ocr_{i:04d}",
            "image": image_path,
            "conversations": [
                {"from": "human", "value": f"<image>\n{prompt}"},
                {"from": "gpt", "value": str(ocr_label)},
            ],
            "metadata": {
                "adapter": "ocr",
                "label": ocr_label,
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
    """Generate the routing dataset from deployed experts on Modal."""
    rng = random.Random(seed)

    splits = config.get("splits", DEFAULT_CONFIG["splits"])
    test_per_adapter = config.get("test_samples_per_adapter", 100)
    adapter_configs = config.get("adapters", DEFAULT_CONFIG["adapters"])

    # Create cache directory for downloaded datasets
    cache_dir = Path(tempfile.gettempdir()) / "moe-routing-datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create output directory (no images dir - we reference source datasets)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("Generating MoE Routing Dataset from Modal")
    print(f"{'='*70}")
    print(f"Output: {output_dir}")
    print(f"Cache: {cache_dir}")
    print(f"Splits: train={splits['train']:.0%}, val={splits['val']:.0%}, test={test_per_adapter} fixed")
    print("\nQuerying Modal for deployed datasets...")

    # Find datasets for each adapter from Modal training history
    adapter_datasets = {}
    deployed_datasets: dict[str, str] = {}  # adapter_name -> dataset_name
    for name, cfg in adapter_configs.items():
        count = cfg.get("count", 0)
        if count <= 0:
            continue
        if cfg.get("source") == "ocr-generated":
            print(f"  {name}: {count} total (from ocr-generated)")
            continue

        result = find_latest_dataset(name, generators_root, cache_dir)
        if result:
            dataset_name, dataset_path = result
            adapter_datasets[name] = (dataset_name, dataset_path)
            deployed_datasets[name] = dataset_name
            train_n = int(count * splits["train"])
            val_n = int(count * splits["val"])
            print(f"  {name}: {count} total -> train={train_n}, val={val_n}, test={test_per_adapter} (from {dataset_name})")
        else:
            print(f"  WARNING: No deployed dataset found for adapter '{name}'")

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

        print(f"Loading {adapter_name} from {dataset_name}...")

        if not train_jsonl.exists():
            print(f"  ERROR: {train_jsonl} not found!")
            continue

        # Group tasks by task_type and track which image each task belongs to
        tasks_by_type: dict[str, list[dict]] = {}
        images_by_type: dict[str, set[str]] = {}
        with open(train_jsonl) as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    task_type = sample.get("metadata", {}).get("task_type", "unknown")
                    img = sample.get("image", "")
                    if task_type not in tasks_by_type:
                        tasks_by_type[task_type] = []
                        images_by_type[task_type] = set()
                    tasks_by_type[task_type].append(sample)
                    if img:
                        images_by_type[task_type].add(img)

        # Shuffle tasks within each type
        for task_type in tasks_by_type:
            rng.shuffle(tasks_by_type[task_type])

        num_task_types = len(tasks_by_type)
        total_tasks = sum(len(tasks) for tasks in tasks_by_type.values())
        print(f"  Total task types: {num_task_types}, Total tasks: {total_tasks}")

        # Flatten all tasks and shuffle for random sampling
        all_tasks = []
        for task_type, tasks in tasks_by_type.items():
            for task in tasks:
                task["_task_type"] = task_type
                all_tasks.append(task)
        rng.shuffle(all_tasks)

        # Track used images across all splits to prevent leakage
        used_images: set[str] = set()
        train_samples = []
        val_samples = []
        test_samples = []
        task_type_dist: dict[str, int] = {}

        # Sample TEST FIRST to ensure we get held-out test samples
        # Then fill train and val from remaining images
        for task in all_tasks:
            img = task.get("image", "")
            if img and img in used_images:
                continue
            if len(test_samples) < test_target:
                test_samples.append(convert_to_routing_sample(task, adapter_name, label, dataset_name))
                if img:
                    used_images.add(img)
            else:
                break

        # Now fill train and val from remaining tasks
        for task in all_tasks:
            img = task.get("image", "")
            task_type = task.get("_task_type", "unknown")

            # Skip if image already used (prevent leakage)
            if img and img in used_images:
                continue

            # Fill train first, then val
            if len(train_samples) < train_target:
                train_samples.append(convert_to_routing_sample(task, adapter_name, label, dataset_name))
                if img:
                    used_images.add(img)
                task_type_dist[task_type] = task_type_dist.get(task_type, 0) + 1
            elif len(val_samples) < val_target:
                val_samples.append(convert_to_routing_sample(task, adapter_name, label, dataset_name))
                if img:
                    used_images.add(img)
            else:
                break  # Got enough for all splits

        all_train_samples.extend(train_samples)
        all_val_samples.extend(val_samples)
        all_test_samples.extend(test_samples)

        print(f"  {adapter_name:20s} -> train={len(train_samples):4d}, val={len(val_samples):4d}, test={len(test_samples):4d}")
        print(f"    Task type distribution: {task_type_dist}")

    # Generate OCR samples from deployed datasets on Modal
    ocr_cfg = adapter_configs.get("ocr", {})
    ocr_count = ocr_cfg.get("count", 0)
    if ocr_count > 0:
        ocr_train_count = int(ocr_count * splits["train"])
        ocr_val_count = int(ocr_count * splits["val"])
        ocr_test_count = test_per_adapter

        print("\nGenerating OCR samples from Modal...")
        ocr_train = generate_ocr_samples(rng, ocr_train_count, deployed_datasets)
        ocr_val = generate_ocr_samples(rng, ocr_val_count, deployed_datasets)
        ocr_test = generate_ocr_samples(rng, ocr_test_count, deployed_datasets)

        all_train_samples.extend(ocr_train)
        all_val_samples.extend(ocr_val)
        all_test_samples.extend(ocr_test)

        print(f"  {'ocr':20s} -> train={len(ocr_train):4d}, val={len(ocr_val):4d}, test={len(ocr_test):4d}")

    # Shuffle all splits
    rng.shuffle(all_train_samples)
    rng.shuffle(all_val_samples)
    rng.shuffle(all_test_samples)

    print("\nTotal samples:")
    print(f"  Train: {len(all_train_samples)}")
    print(f"  Val: {len(all_val_samples)}")
    print(f"  Test: {len(all_test_samples)}")

    # Check balance
    print("\nLabel distribution (train):")
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
    adapters_meta["ocr"] = {"label": ADAPTER_LABELS["ocr"], "source": "ocr-generated"}

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
