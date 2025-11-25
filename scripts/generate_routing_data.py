#!/usr/bin/env python3
"""Generate routing training data from source adapter datasets.

Samples by TASK COUNT (not image count) to ensure balanced classes.
Each source dataset has images with many tasks - we loop through images
adding all tasks until we reach the target task count per class.

Usage:
    python scripts/generate_routing_data.py \\
        --calendar-dataset /path/to/calendar/train.jsonl \\
        --claim-window-dataset /path/to/claim-window/train.jsonl \\
        --provider-select-dataset /path/to/provider-select/train.jsonl \\
        --train-tasks 1000 \\
        --eval-tasks 100

Directory structure:
    ./datasets/routing_20251124_123456/
        ├── data.jsonl           (all samples combined)
        ├── train.jsonl          (80% split of training samples)
        ├── val.jsonl            (20% split of training samples)
        ├── eval.jsonl           (held out eval set)
        └── metadata.json        (dataset generation info)
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AdapterConfig:
    """Configuration for a single adapter source."""
    name: str
    label: int
    dataset_path: Path


# Adapter label mapping
ADAPTER_LABELS = {
    "calendar": 0,
    "claim-window": 1,
    "provider-select": 2,
}


def load_samples_by_task_count(
    adapter_config: AdapterConfig,
    target_tasks: int,
    rng: random.Random,
    exclude_images: set[str] | None = None,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Load samples until reaching target task count.

    Algorithm:
    1. Group all samples by image
    2. Shuffle images
    3. Loop through images, adding ALL tasks for each image
    4. Stop when task count >= target

    Args:
        adapter_config: Adapter configuration
        target_tasks: Target number of tasks to collect
        rng: Random number generator
        exclude_images: Images to skip (e.g., already used for train)

    Returns:
        Tuple of (samples list, set of images used)
    """
    if not adapter_config.dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {adapter_config.dataset_path}")

    exclude_images = exclude_images or set()

    # Load all samples and group by image
    samples_by_image: dict[str, list[dict[str, Any]]] = {}
    with adapter_config.dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                if "conversations" not in sample:
                    continue
                img_path = sample.get("image", "")
                if img_path and img_path not in exclude_images:
                    if img_path not in samples_by_image:
                        samples_by_image[img_path] = []
                    samples_by_image[img_path].append(sample)

    # Shuffle images
    images = list(samples_by_image.keys())
    rng.shuffle(images)

    # Collect tasks until target reached
    collected_samples = []
    images_used = set()

    for img_path in images:
        if len(collected_samples) >= target_tasks:
            break

        tasks = samples_by_image[img_path]
        for task in tasks:
            # Add routing fields
            task_copy = task.copy()
            task_copy["adapter"] = adapter_config.name
            task_copy["label"] = adapter_config.label
            collected_samples.append(task_copy)

        images_used.add(img_path)

    return collected_samples, images_used


def generate_routing_dataset(
    adapters: list[AdapterConfig],
    train_tasks_per_adapter: int,
    eval_tasks_per_adapter: int,
    train_val_split: float,
    output_dir: Path,
    prefix: str,
    seed: int,
) -> None:
    """Generate routing dataset with balanced classes."""
    rng = random.Random(seed)

    print(f"\n{'='*70}")
    print(f"Generating BALANCED Routing Dataset: {prefix}")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"Train tasks per adapter: {train_tasks_per_adapter}")
    print(f"Eval tasks per adapter: {eval_tasks_per_adapter}")
    print(f"Train/val split: {train_val_split:.0%}/{1-train_val_split:.0%}")
    print(f"Random seed: {seed}")
    print(f"{'='*70}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect samples for each adapter
    all_train_samples = []
    all_eval_samples = []
    adapter_sources: dict[str, Path] = {}

    for adapter in adapters:
        print(f"Loading {adapter.name} (label={adapter.label})...")
        adapter_sources[adapter.name] = adapter.dataset_path.parent

        # Load train samples (by task count)
        train_samples, train_images = load_samples_by_task_count(
            adapter, train_tasks_per_adapter, rng
        )

        # Load eval samples (exclude train images)
        eval_samples, _ = load_samples_by_task_count(
            adapter, eval_tasks_per_adapter, rng, exclude_images=train_images
        )

        all_train_samples.extend(train_samples)
        all_eval_samples.extend(eval_samples)

        print(f"  {adapter.name:20s} -> train={len(train_samples):4d}, eval={len(eval_samples):4d}")

    # Shuffle
    rng.shuffle(all_train_samples)
    rng.shuffle(all_eval_samples)

    # Split train into train/val
    split_idx = int(len(all_train_samples) * train_val_split)
    train_data = all_train_samples[:split_idx]
    val_data = all_train_samples[split_idx:]

    print(f"\nTotal samples:")
    print(f"  Train: {len(train_data)}")
    print(f"  Val: {len(val_data)}")
    print(f"  Eval: {len(all_eval_samples)}")

    # Copy images
    print(f"\nCopying images...")
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    all_data = train_data + val_data + all_eval_samples
    images_copied = 0
    images_failed = 0

    for sample in all_data:
        if "image" in sample:
            img_path = Path(sample["image"])
            adapter_name = sample["adapter"]
            source_dir = adapter_sources.get(adapter_name)

            if source_dir:
                # Try different path resolutions
                possible_paths = [source_dir / img_path]
                if len(img_path.parts) > 1:
                    possible_paths.append(source_dir / Path(*img_path.parts[1:]))

                source_img = None
                for p in possible_paths:
                    if p.exists():
                        source_img = p
                        break

                if source_img:
                    dest = images_dir / img_path.name
                    if not dest.exists():
                        shutil.copy2(source_img, dest)
                    sample["image"] = f"images/{img_path.name}"
                    images_copied += 1
                else:
                    images_failed += 1

    print(f"  Copied {images_copied} images ({images_failed} failed)")

    # Save datasets
    _save_jsonl(output_dir / "data.jsonl", all_data)
    _save_jsonl(output_dir / "train.jsonl", train_data)
    _save_jsonl(output_dir / "val.jsonl", val_data)
    _save_jsonl(output_dir / "eval.jsonl", all_eval_samples)

    # Save metadata
    _save_metadata(output_dir / "metadata.json", adapters, train_data, val_data, all_eval_samples, seed, train_val_split)

    print(f"\n{'='*70}")
    print("Dataset generation complete!")
    print(f"{'='*70}\n")


def _save_jsonl(path: Path, data: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"  Saved {len(data)} samples to {path.name}")


def _save_metadata(
    path: Path,
    adapters: list[AdapterConfig],
    train_data: list[dict[str, Any]],
    val_data: list[dict[str, Any]],
    eval_data: list[dict[str, Any]],
    seed: int,
    train_val_split: float,
) -> None:
    def dist(data):
        d = {}
        for item in data:
            a = item["adapter"]
            d[a] = d.get(a, 0) + 1
        return d

    metadata = {
        "dataset_name": "routing",
        "created_at": datetime.now().isoformat(),
        "seed": seed,
        "train_val_split": train_val_split,
        "adapters": {a.name: {"label": a.label, "source": str(a.dataset_path)} for a in adapters},
        "samples": {
            "train": len(train_data),
            "val": len(val_data),
            "eval": len(eval_data),
            "total": len(train_data) + len(val_data) + len(eval_data),
        },
        "label_distribution_train": dist(train_data),
        "label_distribution_val": dist(val_data),
        "label_distribution_eval": dist(eval_data),
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata to {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate balanced routing dataset")
    parser.add_argument("--calendar-dataset", type=Path, required=True)
    parser.add_argument("--claim-window-dataset", type=Path, required=True)
    parser.add_argument("--provider-select-dataset", type=Path, required=True)
    parser.add_argument("--train-tasks", type=int, default=1000, help="Tasks per adapter for training")
    parser.add_argument("--eval-tasks", type=int, default=100, help="Tasks per adapter for eval")
    parser.add_argument("--train-val-split", type=float, default=0.8)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prefix", type=str, default="routing")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    adapters = [
        AdapterConfig("calendar", ADAPTER_LABELS["calendar"], args.calendar_dataset),
        AdapterConfig("claim-window", ADAPTER_LABELS["claim-window"], args.claim_window_dataset),
        AdapterConfig("provider-select", ADAPTER_LABELS["provider-select"], args.provider_select_dataset),
    ]

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path("datasets") / f"{args.prefix}_{timestamp}"

    generate_routing_dataset(
        adapters=adapters,
        train_tasks_per_adapter=args.train_tasks,
        eval_tasks_per_adapter=args.eval_tasks,
        train_val_split=args.train_val_split,
        output_dir=args.output_dir,
        prefix=args.prefix,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
