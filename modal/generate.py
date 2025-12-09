#!/usr/bin/env python3
# Copyright (c) 2025 Tylt LLC. All rights reserved.
# Licensed for research use only. Commercial use requires a license from Tylt LLC.
# Contact: hello@claimhawk.app | See LICENSE for terms.

"""Generate routing dataset on Modal - all 4 steps in one command.

STEP 1: Verify all expert datasets exist on Modal
STEP 2: Assemble routing dataset from expert datasets
STEP 3: Preprocess (tokenize with Qwen processor)
STEP 4: Train router LoRA

Reads adapter config from adapters.yaml (single source of truth).
All data stays on Modal - no local downloads/uploads.

Usage:
    modal run modal/generate.py
    modal run modal/generate.py --samples-per-adapter 500 --test-per-adapter 25
"""

import json
import random
from datetime import datetime
from pathlib import Path

import modal

# =============================================================================
# MODAL APP SETUP
# =============================================================================

app = modal.App("routing-dataset-generator")

# Volumes
training_history = modal.Volume.from_name("claimhawk-training-history", create_if_missing=False)
lora_training = modal.Volume.from_name("claimhawk-lora-training", create_if_missing=False)
moe_data = modal.Volume.from_name("moe-lora-data", create_if_missing=True)

# Image (no extra deps needed for verify/assemble)
image = modal.Image.debian_slim(python_version="3.11")


# =============================================================================
# CONFIGURATION - experts and their routing labels
# Matches config/adapters.yaml (single source of truth)
# =============================================================================

EXPERTS = {
    "calendar": 0,
    "claim-window": 1,
    "ocr": 2,
    "desktop": 3,
    "appointment": 4,
    "login-window": 5,
    "chart-screen": 6,
}

# OCR prompt templates
OCR_PROMPTS = [
    "Read the text in this image and return it using an ocr tool_call",
    "Read the text from this cropped screenshot",
    "Extract the text from this image",
    "Perform OCR on this image and return the text",
    "Transcribe the text in this image",
    "Get the text from this image",
]


def get_experts() -> dict[str, int]:
    """Get expert names and labels from config."""
    return EXPERTS


# =============================================================================
# STEP 1: VERIFY DATA EXISTS
# =============================================================================

@app.function(
    image=image,
    volumes={
        "/history": training_history,
        "/training": lora_training,
    },
    timeout=300,
)
def verify_datasets() -> dict[str, str | None]:
    """Verify all expert datasets exist. Returns {expert: dataset_name} or {expert: None} if missing."""
    experts = get_experts()
    results = {}

    print("\n" + "=" * 70)
    print("STEP 1: VERIFY DATASETS EXIST")
    print("=" * 70 + "\n")

    for expert in experts:
        if expert == "ocr":
            # OCR is generated, not from a dataset
            results[expert] = "ocr-generated"
            print(f"  {expert:20s}: ocr-generated (synthetic)")
            continue

        # Read training history to get latest dataset
        history_path = Path(f"/history/{expert}/runs.jsonl")
        if not history_path.exists():
            results[expert] = None
            print(f"  {expert:20s}: MISSING (no training history)")
            continue

        # Get last line (most recent run)
        with open(history_path) as f:
            lines = f.readlines()
            if not lines:
                results[expert] = None
                print(f"  {expert:20s}: MISSING (empty history)")
                continue
            last_run = json.loads(lines[-1])
            dataset_name = last_run.get("dataset_name")

        if not dataset_name:
            results[expert] = None
            print(f"  {expert:20s}: MISSING (no dataset in history)")
            continue

        # Verify dataset exists on training volume
        dataset_path = Path(f"/training/datasets/{dataset_name}")
        train_path = dataset_path / "train.jsonl"

        if not train_path.exists():
            results[expert] = None
            print(f"  {expert:20s}: MISSING ({dataset_name} not found)")
            continue

        results[expert] = dataset_name
        print(f"  {expert:20s}: {dataset_name}")

    # Check for missing
    missing = [e for e, d in results.items() if d is None]
    if missing:
        print(f"\n{'=' * 70}")
        print(f"ERROR: Missing datasets for: {', '.join(missing)}")
        print(f"{'=' * 70}\n")
        raise ValueError(f"Missing datasets: {missing}")

    print(f"\n  All {len(experts)} experts have datasets available.\n")
    return results


# =============================================================================
# STEP 2: ASSEMBLE DATASET
# =============================================================================

@app.function(
    image=image,
    volumes={
        "/history": training_history,
        "/training": lora_training,
        "/moe": moe_data,
    },
    timeout=600,
)
def assemble_dataset(
    dataset_map: dict[str, str],
    samples_per_adapter: int = 500,
    test_per_adapter: int = 25,
    seed: int = 42,
) -> str:
    """Assemble routing dataset from expert datasets. Returns dataset_name."""
    experts = get_experts()
    rng = random.Random(seed)

    # Generate dataset name
    # Save directly to /moe/{dataset_name} so preprocess.py can find it at /data/{dataset_name}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = f"routing_{timestamp}"
    output_dir = Path(f"/moe/{dataset_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("STEP 2: ASSEMBLE ROUTING DATASET")
    print("=" * 70)
    print(f"\nOutput: {output_dir}")
    print(f"Samples per adapter: {samples_per_adapter}")
    print(f"Test per adapter: {test_per_adapter}\n")

    train_target = int(samples_per_adapter * 0.8)
    val_target = int(samples_per_adapter * 0.2)

    all_train = []
    all_val = []
    all_test = []

    for expert, label in experts.items():
        source_dataset = dataset_map[expert]

        if expert == "ocr":
            # Generate OCR samples from other datasets' images
            ocr_samples = generate_ocr_samples(
                rng, dataset_map, label, train_target, val_target, test_per_adapter
            )
            all_train.extend(ocr_samples["train"])
            all_val.extend(ocr_samples["val"])
            all_test.extend(ocr_samples["test"])
            print(f"  {expert:20s}: train={len(ocr_samples['train'])}, val={len(ocr_samples['val'])}, test={len(ocr_samples['test'])} (synthetic)")
            continue

        # Load expert dataset
        train_path = Path(f"/training/datasets/{source_dataset}/train.jsonl")
        tasks = []
        with open(train_path) as f:
            for line in f:
                if line.strip():
                    tasks.append(json.loads(line))

        rng.shuffle(tasks)

        # Sample for each split (test first to ensure held-out)
        used_images = set()
        train_samples = []
        val_samples = []
        test_samples = []

        # Test first
        for task in tasks:
            img = task.get("image", "")
            if img and img in used_images:
                continue
            if len(test_samples) < test_per_adapter:
                test_samples.append(convert_sample(task, expert, label, source_dataset))
                if img:
                    used_images.add(img)
            else:
                break

        # Then train and val
        for task in tasks:
            img = task.get("image", "")
            if img and img in used_images:
                continue
            if len(train_samples) < train_target:
                train_samples.append(convert_sample(task, expert, label, source_dataset))
                if img:
                    used_images.add(img)
            elif len(val_samples) < val_target:
                val_samples.append(convert_sample(task, expert, label, source_dataset))
                if img:
                    used_images.add(img)
            else:
                break

        all_train.extend(train_samples)
        all_val.extend(val_samples)
        all_test.extend(test_samples)

        print(f"  {expert:20s}: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)} (from {source_dataset})")

    # Shuffle
    rng.shuffle(all_train)
    rng.shuffle(all_val)
    rng.shuffle(all_test)

    # Save
    with open(output_dir / "train.jsonl", "w") as f:
        for sample in all_train:
            f.write(json.dumps(sample) + "\n")

    with open(output_dir / "val.jsonl", "w") as f:
        for sample in all_val:
            f.write(json.dumps(sample) + "\n")

    with open(output_dir / "test.jsonl", "w") as f:
        for sample in all_test:
            f.write(json.dumps(sample) + "\n")

    # Metadata
    metadata = {
        "dataset_name": dataset_name,
        "created": datetime.now().isoformat(),
        "samples_per_adapter": samples_per_adapter,
        "test_per_adapter": test_per_adapter,
        "seed": seed,
        "source_datasets": dataset_map,
        "counts": {
            "train": len(all_train),
            "val": len(all_val),
            "test": len(all_test),
        },
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Commit volume
    moe_data.commit()

    print(f"\n  Total: train={len(all_train)}, val={len(all_val)}, test={len(all_test)}")
    print(f"  Saved to: {output_dir}\n")

    return dataset_name


def convert_sample(task: dict, adapter: str, label: int, source_dataset: str) -> dict:
    """Convert expert task to routing sample."""
    orig_image = task.get("image", "")
    if orig_image:
        img_name = Path(orig_image).name
        image_path = f"{source_dataset}/images/{img_name}"
    else:
        image_path = ""

    # Extract human message
    human_msg = ""
    for conv in task.get("conversations", []):
        if conv.get("from") == "human":
            human_msg = conv.get("value", "").replace("<image>", "").strip()
            break

    return {
        "id": task.get("id", ""),
        "image": image_path,
        "conversations": [
            {"from": "human", "value": f"<image>\n{human_msg}" if human_msg else "<image>"},
            {"from": "gpt", "value": adapter},
        ],
        "metadata": {
            "adapter": adapter,
            "label": label,
            "source_dataset": source_dataset,
        },
    }


def generate_ocr_samples(
    rng: random.Random,
    dataset_map: dict[str, str],
    label: int,
    train_count: int,
    val_count: int,
    test_count: int,
) -> dict[str, list]:
    """Generate OCR routing samples from other datasets' images."""
    # Collect images from all datasets
    all_images = []
    for expert, dataset_name in dataset_map.items():
        if expert == "ocr" or dataset_name == "ocr-generated":
            continue
        images_dir = Path(f"/training/datasets/{dataset_name}/images")
        if images_dir.exists():
            for img in images_dir.glob("*"):
                if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    all_images.append((dataset_name, img.name))

    rng.shuffle(all_images)
    print(f"  Found {len(all_images)} OCR candidate images")

    samples = {"train": [], "val": [], "test": []}
    counts = {"train": train_count, "val": val_count, "test": test_count}

    idx = 0
    for split, count in counts.items():
        for _ in range(count):
            if idx >= len(all_images):
                break
            dataset_name, img_name = all_images[idx]
            prompt = rng.choice(OCR_PROMPTS)
            samples[split].append({
                "id": f"ocr-{split}-{idx}",
                "image": f"{dataset_name}/images/{img_name}",
                "conversations": [
                    {"from": "human", "value": f"<image>\n{prompt}"},
                    {"from": "gpt", "value": "ocr"},
                ],
                "metadata": {
                    "adapter": "ocr",
                    "label": label,
                    "source_dataset": "ocr-generated",
                },
            })
            idx += 1

    return samples


# =============================================================================
# MAIN ENTRYPOINT
# =============================================================================

@app.local_entrypoint()
def main(
    samples_per_adapter: int = 500,
    test_per_adapter: int = 25,
    seed: int = 42,
    skip_train: bool = False,
):
    """Run the full pipeline: verify -> assemble -> preprocess -> train."""
    print("\n" + "=" * 70)
    print("ROUTING DATASET GENERATOR (Modal-native)")
    print("=" * 70)
    print(f"\nSamples per adapter: {samples_per_adapter}")
    print(f"Test per adapter: {test_per_adapter}")
    print(f"Seed: {seed}")
    print(f"Skip train: {skip_train}\n")

    # STEP 1: Verify
    dataset_map = verify_datasets.remote()

    # STEP 2: Assemble
    dataset_name = assemble_dataset.remote(
        dataset_map=dataset_map,
        samples_per_adapter=samples_per_adapter,
        test_per_adapter=test_per_adapter,
        seed=seed,
    )

    print(f"\n{'=' * 70}")
    print(f"Dataset created: {dataset_name}")
    print(f"{'=' * 70}\n")

    if skip_train:
        print("Skipping preprocess and train (--skip-train)")
        return dataset_name

    # STEP 3 & 4: Preprocess and Train
    # Call existing preprocess.py and training.py
    import subprocess

    print("\n" + "=" * 70)
    print("STEP 3: PREPROCESS")
    print("=" * 70 + "\n")

    # Run preprocessing in detached mode and wait for completion
    # This avoids the 30-second grace period timeout on local entrypoint
    preprocess_result = subprocess.run([
        "uvx", "modal", "run", "-d", "modal/preprocess.py",
        "--dataset-name", dataset_name,
    ])

    # Wait for preprocessing to complete by polling for metadata.json
    import time
    max_wait = 3600  # 1 hour max
    poll_interval = 30  # Check every 30 seconds
    waited = 0

    print("Waiting for preprocessing to complete...")
    while waited < max_wait:
        time.sleep(poll_interval)
        waited += poll_interval

        check_result = subprocess.run([
            "uvx", "modal", "volume", "ls", "moe-lora-data",
            f"preprocessed/{dataset_name}/",
        ], capture_output=True, text=True)

        if "metadata.json" in check_result.stdout:
            print(f"✅ Preprocessing completed after {waited}s")
            break

        print(f"  Still waiting... ({waited}s elapsed)")
    else:
        raise RuntimeError(f"Preprocessing timed out after {max_wait}s")

    print("\n" + "=" * 70)
    print("STEP 4: TRAIN")
    print("=" * 70 + "\n")

    run_name = f"router-{datetime.now().strftime('%Y%m%d')}"
    subprocess.run([
        "uvx", "modal", "run", "modal/training.py",
        "--run-name", run_name,
        "--dataset-name", dataset_name,
    ], check=True)

    print("\n" + "=" * 70)
    print("STEP 5: TEST NEW MODEL")
    print("=" * 70 + "\n")

    # Run test and capture output to get accuracy
    result = subprocess.run([
        "uvx", "modal", "run", "modal/test.py",
        "--run-name", run_name,
        "--dataset-name", dataset_name,
    ], check=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    # Parse accuracy from output
    new_accuracy = None
    for line in result.stdout.split("\n"):
        if "Overall Accuracy:" in line:
            # Extract percentage like "Overall Accuracy: 85.00%"
            try:
                new_accuracy = float(line.split(":")[1].strip().replace("%", ""))
            except (IndexError, ValueError):
                pass

    print("\n" + "=" * 70)
    print("STEP 6: DEPLOY IF IMPROVED")
    print("=" * 70 + "\n")

    # Test deployed version to compare
    print("Testing currently deployed router...")
    deployed_result = subprocess.run([
        "uvx", "modal", "run", "modal/test.py",
        "--dataset-name", dataset_name,
        "--deployed",
    ], capture_output=True, text=True)

    deployed_accuracy = None
    if deployed_result.returncode == 0:
        for line in deployed_result.stdout.split("\n"):
            if "Overall Accuracy:" in line:
                try:
                    deployed_accuracy = float(line.split(":")[1].strip().replace("%", ""))
                except (IndexError, ValueError):
                    pass
        print(deployed_result.stdout)
    else:
        print("No deployed router found or test failed - will deploy new model")

    # Compare and deploy if improved
    should_deploy = False
    if new_accuracy is not None:
        if deployed_accuracy is None:
            print(f"\nNo deployed model to compare. New accuracy: {new_accuracy:.2f}%")
            should_deploy = True
        elif new_accuracy > deployed_accuracy:
            print(f"\n✅ NEW MODEL IS BETTER!")
            print(f"   New: {new_accuracy:.2f}% vs Deployed: {deployed_accuracy:.2f}%")
            should_deploy = True
        else:
            print(f"\n❌ New model not better than deployed")
            print(f"   New: {new_accuracy:.2f}% vs Deployed: {deployed_accuracy:.2f}%")
            print("   Skipping deployment")
    else:
        print("Could not parse new model accuracy - skipping deployment")

    if should_deploy:
        print("\nDeploying new model...")
        subprocess.run([
            "uvx", "modal", "run", "modal/deploy.py",
            "--latest",
        ], check=True)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Dataset: {dataset_name}")
    print(f"Run: {run_name}")
    if new_accuracy:
        print(f"New Model Accuracy: {new_accuracy:.2f}%")
    if deployed_accuracy:
        print(f"Deployed Accuracy: {deployed_accuracy:.2f}%")
    if should_deploy:
        print("Status: DEPLOYED")
    else:
        print("Status: NOT DEPLOYED (not improved)")
    print("=" * 70 + "\n")

    return dataset_name
