#!/usr/bin/env python3
"""Generate balanced routing dataset on Modal.

Reads from claimhawk-training-data volume, generates balanced routing dataset,
saves to moe-lora-data volume.

Usage:
    modal run modal/generate_routing.py --train-tasks 1000 --eval-tasks 100
"""

import json
import random
import shutil
from datetime import datetime
from pathlib import Path

import modal

app = modal.App("routing-dataset-generator")

# Volumes
training_data_volume = modal.Volume.from_name("claimhawk-training-data", create_if_missing=False)
moe_volume = modal.Volume.from_name("moe-lora-data", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.12")

# Dataset mappings
ADAPTER_DATASETS = {
    "calendar": "mike-im-day-clicks-system-prompt-8B_20251120_180854",
    "claim-window": "claim-window_20251123_221931",
    "provider-select": "select-provider-dropdown_20251123_191451",
}

ADAPTER_LABELS = {
    "calendar": 0,
    "claim-window": 1,
    "provider-select": 2,
}


@app.function(
    image=image,
    volumes={
        "/training-data": training_data_volume,
        "/moe-data": moe_volume,
    },
    timeout=3600,
)
def generate_routing_dataset(
    train_tasks: int = 1000,
    eval_tasks: int = 100,
    train_val_split: float = 0.8,
    seed: int = 42,
):
    """Generate balanced routing dataset from 3 adapter sources."""

    rng = random.Random(seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"/moe-data/datasets/routing_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Generating BALANCED Routing Dataset")
    print(f"{'='*70}")
    print(f"Output: {output_dir}")
    print(f"Train tasks per adapter: {train_tasks}")
    print(f"Eval tasks per adapter: {eval_tasks}")
    print(f"{'='*70}\n")

    all_train_samples = []
    all_eval_samples = []

    for adapter_name, dataset_name in ADAPTER_DATASETS.items():
        label = ADAPTER_LABELS[adapter_name]
        dataset_path = Path(f"/training-data/{dataset_name}/train.jsonl")
        source_images_dir = Path(f"/training-data/{dataset_name}/images")

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

            # Filter conversations: keep system and human, replace gpt with adapter name
            if "conversations" in task:
                new_convs = []
                for conv in task["conversations"]:
                    if conv["from"] == "system":
                        new_convs.append(conv)
                    elif conv["from"] == "human":
                        new_convs.append(conv)
                    elif conv["from"] == "gpt":
                        # Replace with adapter name as the response
                        new_convs.append({"from": "gpt", "value": adapter_name})
                        break  # Only keep first response
                task_copy["conversations"] = new_convs

            return task_copy

        # Collect train tasks - iterate through images, grab ALL tasks from each image
        train_samples = []
        train_images_used = []
        for img in image_list:
            if len(train_samples) >= train_tasks:
                break
            # Add ALL tasks from this image
            for task in tasks_by_image[img]:
                train_samples.append(convert_to_routing_sample(task, adapter_name, label))
            train_images_used.append(img)

        # Collect eval tasks from remaining images
        eval_samples = []
        for img in image_list:
            if img in train_images_used:
                continue
            if len(eval_samples) >= eval_tasks:
                break
            for task in tasks_by_image[img]:
                eval_samples.append(convert_to_routing_sample(task, adapter_name, label))

        all_train_samples.extend(train_samples)
        all_eval_samples.extend(eval_samples)

        print(f"  {adapter_name:20s} -> train={len(train_samples):4d} (from {len(train_images_used)} images), eval={len(eval_samples):4d}")

        # Copy images
        images_copied = set()
        for sample in train_samples + eval_samples:
            img_path = sample.get("image", "")
            if img_path:
                img_name = Path(img_path).name
                if img_name in images_copied:
                    sample["image"] = f"images/{img_name}"
                    continue

                src = source_images_dir / img_name
                if not src.exists():
                    src = Path(f"/training-data/{dataset_name}") / img_path
                if src.exists():
                    dst = images_dir / img_name
                    if not dst.exists():
                        shutil.copy2(src, dst)
                    images_copied.add(img_name)
                sample["image"] = f"images/{img_name}"

    # Shuffle combined dataset
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

    # Check balance
    print(f"\nLabel distribution (train):")
    dist = {}
    for s in train_data:
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

    all_data = train_data + val_data + all_eval_samples
    save_jsonl(output_dir / "data.jsonl", all_data)
    save_jsonl(output_dir / "train.jsonl", train_data)
    save_jsonl(output_dir / "val.jsonl", val_data)
    save_jsonl(output_dir / "eval.jsonl", all_eval_samples)

    # Metadata
    metadata = {
        "dataset_name": "routing",
        "created_at": datetime.now().isoformat(),
        "seed": seed,
        "train_val_split": train_val_split,
        "adapters": {name: {"label": ADAPTER_LABELS[name], "source": ds} for name, ds in ADAPTER_DATASETS.items()},
        "samples": {"train": len(train_data), "val": len(val_data), "eval": len(all_eval_samples)},
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
    train_tasks: int = 1000,
    eval_tasks: int = 100,
    train_val_split: float = 0.8,
    seed: int = 42,
):
    import subprocess
    from pathlib import Path

    dataset_name = generate_routing_dataset.remote(
        train_tasks=train_tasks,
        eval_tasks=eval_tasks,
        train_val_split=train_val_split,
        seed=seed,
    )
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
