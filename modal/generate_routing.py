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
    val_tasks: int = 100,
    eval_tasks: int = 100,
    seed: int = 42,
):
    """Generate balanced routing dataset from 3 adapter sources.

    Creates three separate splits:
    - train.jsonl: Training data (train_tasks total, balanced across adapters)
    - val.jsonl: Validation data used during training for early stopping (val_tasks total)
    - eval.jsonl: Held-out evaluation data for final accuracy testing (eval_tasks total)
    """

    rng = random.Random(seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"/moe-data/datasets/routing_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    # Per-adapter targets (divide by 3 adapters)
    train_per_adapter = train_tasks // 3
    val_per_adapter = val_tasks // 3
    eval_per_adapter = eval_tasks // 3

    print(f"\n{'='*70}")
    print(f"Generating BALANCED Routing Dataset")
    print(f"{'='*70}")
    print(f"Output: {output_dir}")
    print(f"Train: {train_tasks} total ({train_per_adapter} per adapter)")
    print(f"Val: {val_tasks} total ({val_per_adapter} per adapter) - for training early stopping")
    print(f"Eval: {eval_tasks} total ({eval_per_adapter} per adapter) - held-out for final accuracy")
    print(f"{'='*70}\n")

    all_train_samples = []
    all_val_samples = []
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

        # Collect samples - iterate through images, grab ALL tasks from each image
        # Split into train, val, eval using separate image pools (no leakage)
        train_samples = []
        val_samples = []
        eval_samples = []
        images_used = {"train": [], "val": [], "eval": []}

        # First pass: collect train samples
        for img in image_list:
            if len(train_samples) >= train_per_adapter:
                break
            for task in tasks_by_image[img]:
                train_samples.append(convert_to_routing_sample(task, adapter_name, label))
            images_used["train"].append(img)

        # Second pass: collect val samples from remaining images
        for img in image_list:
            if img in images_used["train"]:
                continue
            if len(val_samples) >= val_per_adapter:
                break
            for task in tasks_by_image[img]:
                val_samples.append(convert_to_routing_sample(task, adapter_name, label))
            images_used["val"].append(img)

        # Third pass: collect eval samples from remaining images
        for img in image_list:
            if img in images_used["train"] or img in images_used["val"]:
                continue
            if len(eval_samples) >= eval_per_adapter:
                break
            for task in tasks_by_image[img]:
                eval_samples.append(convert_to_routing_sample(task, adapter_name, label))
            images_used["eval"].append(img)

        all_train_samples.extend(train_samples)
        all_val_samples.extend(val_samples)
        all_eval_samples.extend(eval_samples)

        print(f"  {adapter_name:20s} -> train={len(train_samples):4d}, val={len(val_samples):4d}, eval={len(eval_samples):4d}")

        # Copy images
        images_copied = set()
        for sample in train_samples + val_samples + eval_samples:
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

    # Shuffle all splits
    rng.shuffle(all_train_samples)
    rng.shuffle(all_val_samples)
    rng.shuffle(all_eval_samples)

    print(f"\nTotal samples:")
    print(f"  Train: {len(all_train_samples)} (for training)")
    print(f"  Val: {len(all_val_samples)} (for early stopping during training)")
    print(f"  Eval: {len(all_eval_samples)} (held-out for final accuracy)")

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

    all_data = all_train_samples + all_val_samples + all_eval_samples
    save_jsonl(output_dir / "data.jsonl", all_data)
    save_jsonl(output_dir / "train.jsonl", all_train_samples)
    save_jsonl(output_dir / "val.jsonl", all_val_samples)
    save_jsonl(output_dir / "eval.jsonl", all_eval_samples)

    # Metadata
    metadata = {
        "dataset_name": "routing",
        "created_at": datetime.now().isoformat(),
        "seed": seed,
        "adapters": {name: {"label": ADAPTER_LABELS[name], "source": ds} for name, ds in ADAPTER_DATASETS.items()},
        "samples": {"train": len(all_train_samples), "val": len(all_val_samples), "eval": len(all_eval_samples)},
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
    val_tasks: int = 100,
    eval_tasks: int = 100,
    seed: int = 42,
):
    import subprocess
    from pathlib import Path

    dataset_name = generate_routing_dataset.remote(
        train_tasks=train_tasks,
        val_tasks=val_tasks,
        eval_tasks=eval_tasks,
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
