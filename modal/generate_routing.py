#!/usr/bin/env python3
"""Generate balanced routing dataset on Modal.

Reads from claimhawk-training-data volume, generates balanced routing dataset,
saves to moe-lora-data volume.

Usage:
    modal run modal/generate_routing.py --train-tasks 1000 --eval-tasks 100
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

image = modal.Image.debian_slim(python_version="3.12")


# Dataset mappings - loaded from config/loras.json locally, with fallback for Modal
ADAPTER_DATASETS = {
    "calendar": "calendar-mike-20251202_113010",
    "claim-window": "procedure-scroll-mike-20251202_142525",
    "provider-select": "provider-select-mike-20251202_144036",
    "appointment": "appointment_20251202_111820",
    "login-window": "login-window-michaeloneal-20251202_113305",
    "desktop": "desktop-mike-20251201_214626",
    "chart-screen": "chart-screen-mike-20251202_115044",
}
# Try to load from config (works locally, fails silently on Modal)
try:
    import os as _os
    _config_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "config", "loras.json")
    with open(_config_path) as _f:
        _config = json.load(_f)
    ADAPTER_DATASETS = {name: info["dataset"] for name, info in _config["loras"].items() if "dataset" in info}
    del _os, _config_path, _f, _config
except FileNotFoundError:
    pass  # Use hardcoded fallback above


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


def generate_ocr_samples(rng, count: int, images_dir: Path, ocr_source_dir: Path) -> list[dict]:
    """Generate OCR routing samples for Chandra using actual OCR images."""
    templates = list(OCR_PROMPT_TEMPLATES)
    rng.shuffle(templates)

    # Get list of OCR images
    ocr_images = sorted(ocr_source_dir.glob("*.png")) if ocr_source_dir.exists() else []
    if not ocr_images:
        print(f"  WARNING: No OCR images found in {ocr_source_dir}")
        return []

    # Copy OCR images to output images dir
    for img in ocr_images:
        dst = images_dir / f"ocr_{img.name}"
        if not dst.exists():
            shutil.copy2(img, dst)

    samples = []
    for i in range(count):
        prompt = templates[i % len(templates)]
        # Cycle through available OCR images
        img = ocr_images[i % len(ocr_images)]
        sample = {
            "id": f"ocr_{i:04d}",
            "image": f"images/ocr_{img.name}",
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
    train_tasks: int = 1000,
    val_tasks: int = 100,
    eval_tasks: int = 100,
    seed: int = 42,
):
    """Generate balanced routing dataset from adapter sources + OCR.

    Creates three separate splits:
    - train.jsonl: Training data (train_tasks total, balanced across adapters)
    - val.jsonl: Validation data used during training for early stopping (val_tasks total)
    - eval.jsonl: Held-out evaluation data for final accuracy testing (eval_tasks total)
    """

    # Use pre-loaded adapter config
    adapter_datasets = ADAPTER_DATASETS
    num_adapters = len(adapter_datasets) + 1  # +1 for chandra

    rng = random.Random(seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"/moe-data/datasets/routing_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    # Per-adapter targets (divide by num_adapters)
    train_per_adapter = train_tasks // num_adapters
    val_per_adapter = val_tasks // num_adapters
    eval_per_adapter = eval_tasks // num_adapters

    print(f"\n{'='*70}")
    print(f"Generating BALANCED Routing Dataset")
    print(f"{'='*70}")
    print(f"Output: {output_dir}")
    print(f"Adapters: {list(adapter_datasets.keys())} + chandra")
    print(f"Train: {train_tasks} total ({train_per_adapter} per adapter)")
    print(f"Val: {val_tasks} total ({val_per_adapter} per adapter) - for training early stopping")
    print(f"Eval: {eval_tasks} total ({eval_per_adapter} per adapter) - held-out for final accuracy")
    print(f"{'='*70}\n")

    all_train_samples = []
    all_val_samples = []
    all_eval_samples = []

    for adapter_name, dataset_name in adapter_datasets.items():
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
                    src = Path(f"/training-data/datasets/{dataset_name}") / img_path
                if src.exists():
                    dst = images_dir / img_name
                    if not dst.exists():
                        shutil.copy2(src, dst)
                    images_copied.add(img_name)
                sample["image"] = f"images/{img_name}"

    # Generate OCR samples for Chandra
    print(f"\nGenerating OCR samples for chandra...")
    ocr_source_dir = Path("/moe-data/ocr-images/ocr")
    ocr_train = generate_ocr_samples(rng, train_per_adapter, images_dir, ocr_source_dir)
    ocr_val = generate_ocr_samples(rng, val_per_adapter, images_dir, ocr_source_dir)
    ocr_eval = generate_ocr_samples(rng, eval_per_adapter, images_dir, ocr_source_dir)

    all_train_samples.extend(ocr_train)
    all_val_samples.extend(ocr_val)
    all_eval_samples.extend(ocr_eval)

    print(f"  {'chandra':20s} -> train={len(ocr_train):4d}, val={len(ocr_val):4d}, eval={len(ocr_eval):4d}")

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
    adapters_meta = {name: {"label": ADAPTER_LABELS[name], "source": ds} for name, ds in adapter_datasets.items()}
    adapters_meta["chandra"] = {"label": ADAPTER_LABELS["chandra"], "source": "generated"}

    metadata = {
        "dataset_name": "routing",
        "created_at": datetime.now().isoformat(),
        "seed": seed,
        "adapters": adapters_meta,
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

    # Check if invoked from shell script - warn if not
    check_script_invocation()

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
