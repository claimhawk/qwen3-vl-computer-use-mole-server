#!/usr/bin/env python3
# Copyright (c) 2025 Tylt LLC. All rights reserved.
# Licensed for research use only. Commercial use requires a license from Tylt LLC.
# Contact: hello@claimhawk.app | See LICENSE for terms.

"""
Preprocess OCR data from expert dataset ocr/ subfolders.

Scans expert datasets for ocr/ folders and preprocesses them for router training.
Output is saved to claimhawk-lora-training preprocessed/ so it can be linked
by link_router_data.py just like other experts.

Usage:
    modal run modal/preprocess_ocr.py
"""

import json
import random
from datetime import datetime
from pathlib import Path

import modal

# Expert datasets that have ocr/ subfolders
EXPERT_DATASETS_WITH_OCR = [
    "claim-window--mike--20251207_163040",
    "appointment--mike--20251208_103029",
    "chart-screen--mike--20251209_085858",
]

# OCR label for router (must match config/adapters.yaml)
OCR_LABEL = 2

# Volume names
EXPERT_VOLUME_NAME = "claimhawk-lora-training"
MODEL_CACHE_NAME = "claimhawk-model-cache"
BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

# Router system prompt (must match training)
ROUTER_SYSTEM_PROMPT = """You are a Mixture of Experts router. You have been trained to look at an image and a text instruction and determine what adapter to route to.

Valid adapter labels:
- 0: calendar
- 1: claim-window
- 2: ocr
- 3: desktop
- 4: appointment
- 5: login-window
- 6: chart-screen

Reply with only the numeric label (0-6) for the adapter that should handle this image and instruction."""

app = modal.App("preprocess-ocr")
expert_volume = modal.Volume.from_name(EXPERT_VOLUME_NAME, create_if_missing=False)
model_cache = modal.Volume.from_name(MODEL_CACHE_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        extra_options="--index-url https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers>=4.57.0",
        "qwen-vl-utils",
        "Pillow>=10.0.0",
        "tqdm>=4.65.0",
    )
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,
    volumes={
        "/expert-data": expert_volume,
        "/models": model_cache,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def preprocess_ocr(
    output_name: str = None,
    val_split: float = 0.1,
    max_samples: int = None,
    seed: int = 42,
):
    """Preprocess OCR images from expert dataset ocr/ folders."""
    import torch
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    from PIL import Image
    from tqdm import tqdm

    random.seed(seed)
    torch.manual_seed(seed)

    expert_volume.reload()

    if output_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"ocr--aggregated--{timestamp}"

    print(f"\n{'='*70}")
    print("OCR Preprocessing from Expert Datasets")
    print(f"{'='*70}")
    print(f"Output: {output_name}")
    print(f"Val split: {val_split}")
    print(f"Max samples: {max_samples or 'unlimited'}")

    # Load processor
    cached_model_path = f"/models/{BASE_MODEL.replace('/', '--')}"
    if Path(cached_model_path).exists():
        print(f"Using cached processor: {cached_model_path}")
        processor = AutoProcessor.from_pretrained(cached_model_path, trust_remote_code=True)
    else:
        print(f"Loading processor from HuggingFace: {BASE_MODEL}")
        processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # Collect all OCR images from expert datasets
    all_images = []  # List of (image_path, source_dataset)

    print(f"\n{'='*70}")
    print("Collecting OCR Images")
    print(f"{'='*70}")

    for dataset_name in EXPERT_DATASETS_WITH_OCR:
        ocr_dir = Path(f"/expert-data/datasets/{dataset_name}/ocr")
        if not ocr_dir.exists():
            print(f"  {dataset_name}: NO OCR FOLDER")
            continue

        images_dir = ocr_dir / "images"
        if not images_dir.exists():
            print(f"  {dataset_name}: NO IMAGES FOLDER")
            continue

        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        for img_path in image_files:
            all_images.append((img_path, dataset_name))

        print(f"  {dataset_name}: {len(image_files)} images")

    print(f"\nTotal OCR images: {len(all_images)}")

    if max_samples and len(all_images) > max_samples:
        random.shuffle(all_images)
        all_images = all_images[:max_samples]
        print(f"Limited to {max_samples} samples")

    # Split into train/val
    random.shuffle(all_images)
    n_val = int(len(all_images) * val_split)
    val_images = all_images[:n_val]
    train_images = all_images[n_val:]

    print(f"Train: {len(train_images)}, Val: {len(val_images)}")

    # Create output directories
    output_base = Path(f"/expert-data/preprocessed/{output_name}")
    train_dir = output_base / "train"
    val_dir = output_base / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    def preprocess_image(image_path: Path) -> dict:
        """Preprocess a single OCR image for router training."""
        # Load image
        pil_image = Image.open(image_path).convert("RGB")

        # Build messages with router prompt
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": "What adapter should handle this image?"},
                ],
            },
            {
                "role": "assistant",
                "content": str(OCR_LABEL),  # OCR label
            },
        ]

        # Process with Qwen processor
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, _ = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            return_tensors="pt",
            padding=False,
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]

        # Create labels: -100 for input, actual tokens for response
        # Find where assistant response starts
        labels = input_ids.clone()

        # The label is just the OCR number + <|im_end|>
        # Mask everything except the last few tokens (the label)
        label_tokens = processor.tokenizer.encode(str(OCR_LABEL), add_special_tokens=False)
        label_len = len(label_tokens) + 1  # +1 for <|im_end|>

        labels[:-label_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }

    # Process train split
    print(f"\n{'='*70}")
    print(f"Processing train split ({len(train_images)} images)")
    print(f"{'='*70}")

    train_count = 0
    for idx, (img_path, source) in enumerate(tqdm(train_images, desc="train")):
        try:
            sample = preprocess_image(img_path)
            output_path = train_dir / f"sample_{train_count:06d}.pt"
            torch.save(sample, output_path)
            train_count += 1
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Process val split
    print(f"\n{'='*70}")
    print(f"Processing val split ({len(val_images)} images)")
    print(f"{'='*70}")

    val_count = 0
    for idx, (img_path, source) in enumerate(tqdm(val_images, desc="val")):
        try:
            sample = preprocess_image(img_path)
            output_path = val_dir / f"sample_{val_count:06d}.pt"
            torch.save(sample, output_path)
            val_count += 1
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Save metadata
    metadata = {
        "name": output_name,
        "created": datetime.now().isoformat(),
        "type": "ocr-aggregated",
        "label": OCR_LABEL,
        "source_datasets": EXPERT_DATASETS_WITH_OCR,
        "splits": {
            "train": train_count,
            "val": val_count,
        },
        "config": {
            "val_split": val_split,
            "max_samples": max_samples,
            "seed": seed,
        },
    }

    with open(output_base / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    expert_volume.commit()

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Output: {output_base}")
    print(f"Train: {train_count}")
    print(f"Val: {val_count}")

    return metadata


@app.local_entrypoint()
def main(
    output_name: str = None,
    val_split: float = 0.1,
    max_samples: int = None,
):
    """Preprocess OCR data from expert datasets."""
    result = preprocess_ocr.remote(
        output_name=output_name,
        val_split=val_split,
        max_samples=max_samples,
    )

    print(f"\n{'='*70}")
    print("OCR PREPROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Dataset: {result['name']}")
    print(f"Train: {result['splits']['train']}")
    print(f"Val: {result['splits']['val']}")
    print(f"\nAdd to link_router_data.py EXPERT_DATASETS:")
    print(f'    "ocr": "{result["name"]}",')
