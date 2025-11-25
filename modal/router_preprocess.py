#!/usr/bin/env python3
"""
Router Data Preprocessing on Modal

Preprocess routing dataset (JSONL + images) on Modal's CPU instances.
Saves preprocessed tensors to a Modal volume for reuse across training runs.

This eliminates the need to preprocess on expensive GPU instances and
speeds up training startup time.

Usage:
    modal run modal/router_preprocess.py --dataset-dir routing_20251124_115228
"""

import json
from pathlib import Path
from typing import Any

import modal

# Modal App Setup
app = modal.App("router-preprocessing")

# Volume - defined at module level so we can reload it
VOLUME = modal.Volume.from_name("moe-lora-data", create_if_missing=True)

# Docker Image with Dependencies (CPU-only, no GPU needed)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
    )
    .pip_install(
        "transformers>=4.57.0",
        "qwen-vl-utils",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
    )
)


@app.function(
    image=image,
    cpu=16,  # Use high CPU for faster preprocessing
    memory=32768,  # 32GB RAM
    timeout=7200,  # 2 hours max
    volumes={
        "/data": VOLUME,
    },
)
def preprocess_routing_dataset(dataset_name: str):
    """
    Preprocess routing dataset on Modal CPU instance.

    Loads raw JSONL + images from /data/datasets/{dataset_name} and saves
    preprocessed tensors with routing labels to /data/preprocessed/routing_{dataset_name}.

    Args:
        dataset_name: Name of the dataset directory (e.g., "routing_20251124_115228")
    """
    import torch
    from transformers import AutoProcessor
    from PIL import Image
    from tqdm import tqdm
    from qwen_vl_utils import process_vision_info

    # Reload the mounted volume to see latest committed data
    VOLUME.reload()

    # Get paths
    data_root = Path("/data")
    dataset_path = data_root / "datasets" / dataset_name
    preprocessed_path = data_root / "preprocessed" / dataset_name

    print(f"\n{'='*80}")
    print("🚀 Starting Router Preprocessing on Modal")
    print(f"{'='*80}\n")
    print(f"Dataset: {dataset_name}")
    print(f"Input path: {dataset_path}")
    print(f"Output path: {preprocessed_path}")

    # ============================================================================
    # STEP 1: Load Dataset
    # ============================================================================

    print(f"\n{'='*80}")
    print("📦 STEP 1: Loading Dataset")
    print(f"{'='*80}\n")

    # Find train and val files
    train_file = dataset_path / "train.jsonl"
    val_file = dataset_path / "val.jsonl"

    if not train_file.exists():
        raise FileNotFoundError(f"Train file not found: {train_file}")
    if not val_file.exists():
        raise FileNotFoundError(f"Val file not found: {val_file}")

    # Load JSONL data
    def load_jsonl(path):
        data = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    train_data = load_jsonl(train_file)
    val_data = load_jsonl(val_file)

    print(f"📊 Dataset size:")
    print(f"   Train samples: {len(train_data)}")
    print(f"   Val samples: {len(val_data)}")

    # ============================================================================
    # STEP 2: Load Processor
    # ============================================================================

    print(f"\n{'='*80}")
    print("📦 STEP 2: Loading Processor")
    print(f"{'='*80}\n")

    model_name = "Qwen/Qwen3-VL-8B-Instruct"
    print(f"Loading processor: {model_name}")

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    print("✅ Processor loaded")

    # ============================================================================
    # STEP 2.5: Cache Image Embeddings (Performance Optimization)
    # ============================================================================

    print(f"\n{'='*80}")
    print("🖼️  STEP 2.5: Caching Image Embeddings")
    print(f"{'='*80}\n")

    # Find all unique images referenced in the dataset
    unique_images = set()
    for sample in train_data + val_data:
        if "image" in sample:
            unique_images.add(sample["image"])

    print(f"Found {len(unique_images)} unique images (from {len(train_data) + len(val_data)} total samples)")
    if len(unique_images) > 0:
        print(f"Reuse ratio: {(len(train_data) + len(val_data)) / len(unique_images):.1f}x")

    # Cache for processed image tensors
    image_cache = {}

    print("\nProcessing unique images...")
    for img_path in tqdm(sorted(unique_images), desc="Caching images"):
        # Image paths should be relative to the dataset directory
        # The source datasets use paths like "calendar_x/images/screen.png"
        # We need to resolve these relative to the source dataset location

        # For routing datasets, the "image" field should contain the absolute path
        # or a path relative to the generator's source dataset location
        # Since we're working with routing data, we need to handle the image path carefully

        # First try as-is (might be absolute)
        full_path = Path(img_path)

        # If not absolute or doesn't exist, try relative to dataset path
        if not full_path.is_absolute() or not full_path.exists():
            # Try relative to current dataset
            full_path = dataset_path / img_path
            if not full_path.exists():
                # Try stripping first path component if it matches dataset name
                parts = Path(img_path).parts
                if len(parts) > 1:
                    full_path = dataset_path / Path(*parts[1:])

        if not full_path.exists():
            print(f"⚠️  Warning: Image not found: {img_path}")
            continue

        # Process image using the processor
        image = Image.open(full_path)
        # Get image tensor directly without text
        image_inputs, _ = process_vision_info(
            [{"role": "user", "content": [{"type": "image", "image": f"file://{full_path}"}]}],
            image_patch_size=16
        )

        # Cache the processed image tensor using original img_path as key
        image_cache[img_path] = {
            "pixel_values": image_inputs[0] if image_inputs else None,
            "image": image
        }

    print(f"✅ Cached {len(image_cache)} images")

    # ============================================================================
    # STEP 3: Preprocess Data
    # ============================================================================

    print(f"\n{'='*80}")
    print("🔧 STEP 3: Preprocessing Data")
    print(f"{'='*80}\n")

    def prepare_routing_sample(sample, image_cache):
        """
        Prepare a single routing sample for training.
        Converts from JSONL format to model input tensors + routing label.
        Uses cached image embeddings for performance.

        Returns None if the image is missing (instead of crashing).
        """
        # Get image from cache (already processed)
        img_path = sample.get("image")
        if not img_path or img_path not in image_cache:
            # Skip samples with missing images instead of crashing
            return None

        # Get cached image data
        cached_image = image_cache[img_path]

        # Get conversations and routing label
        conversations = sample["conversations"]
        label = sample["label"]  # Integer label for routing

        # Convert to Qwen-VL's expected format
        messages = []

        for msg in conversations:
            # Map dataset roles to model roles
            if msg.get("from") == "system":
                role = "system"
            elif msg.get("from") == "human":
                role = "user"
            else:  # "gpt" or "assistant"
                role = "assistant"

            content_list = []

            # Parse the value - check for <image> token
            value = msg.get("value", "")
            if "<image>" in value:
                # Add image placeholder
                content_list.append({"type": "image"})
                # Remove <image> token and add remaining text
                text = value.replace("<image>", "").strip()
                if text:
                    content_list.append({"type": "text", "text": text})
            else:
                # Just text content
                content_list.append({"type": "text", "text": value})

            messages.append({"role": role, "content": content_list})

        # Get the text template (without actually processing the image)
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # Use cached image inputs instead of reprocessing
        image_inputs = [cached_image["pixel_values"]] if cached_image["pixel_values"] is not None else None

        # Now process text with cached images
        model_inputs = processor(
            text=[text],
            images=image_inputs,
            videos=None,  # No videos in this dataset
            return_tensors="pt",
            padding=False,
            do_resize=False
        )

        # Return processed sample with routing label
        input_ids = model_inputs["input_ids"][0] if isinstance(model_inputs["input_ids"][0], torch.Tensor) else torch.tensor(model_inputs["input_ids"][0])
        attention_mask = model_inputs["attention_mask"][0] if isinstance(model_inputs["attention_mask"][0], torch.Tensor) else torch.tensor(model_inputs["attention_mask"][0])

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.long),  # Routing label
        }

        if "pixel_values" in model_inputs:
            result["pixel_values"] = model_inputs["pixel_values"]
            result["image_grid_thw"] = model_inputs["image_grid_thw"]

        return result

    # Process training data
    print("Processing training data...")
    train_processed = []
    train_output_dir = Path(preprocessed_path) / "train"
    train_output_dir.mkdir(parents=True, exist_ok=True)

    BATCH_SIZE = 8  # Process 8 samples at once
    batch_buffer = []
    batch_indices = []

    for i, sample in enumerate(tqdm(train_data, desc="Train")):
        try:
            processed = prepare_routing_sample(sample, image_cache)

            # Convert tensors to CPU and add to batch buffer
            processed_cpu = {
                "input_ids": processed["input_ids"].cpu(),
                "attention_mask": processed["attention_mask"].cpu(),
                "label": processed["label"].cpu(),
            }
            if "pixel_values" in processed:
                processed_cpu["pixel_values"] = processed["pixel_values"].cpu()
                processed_cpu["image_grid_thw"] = processed["image_grid_thw"].cpu()

            batch_buffer.append(processed_cpu)
            batch_indices.append(i)

            # Save batch when full or at end of dataset
            if len(batch_buffer) >= BATCH_SIZE or i == len(train_data) - 1:
                # Save all samples in batch at once (reduces I/O overhead)
                for batch_idx, batch_sample in zip(batch_indices, batch_buffer):
                    sample_path = train_output_dir / f"sample_{batch_idx:06d}.pt"
                    torch.save(batch_sample, sample_path)
                    train_processed.append(str(sample_path))

                batch_buffer = []
                batch_indices = []

        except Exception as e:
            print(f"\n❌ Error processing train sample {i}: {e}")
            raise

    # Process validation data
    print("\nProcessing validation data...")
    val_processed = []
    val_output_dir = Path(preprocessed_path) / "val"
    val_output_dir.mkdir(parents=True, exist_ok=True)

    batch_buffer = []
    batch_indices = []

    for i, sample in enumerate(tqdm(val_data, desc="Val")):
        try:
            processed = prepare_routing_sample(sample, image_cache)

            # Convert tensors to CPU and add to batch buffer
            processed_cpu = {
                "input_ids": processed["input_ids"].cpu(),
                "attention_mask": processed["attention_mask"].cpu(),
                "label": processed["label"].cpu(),
            }
            if "pixel_values" in processed:
                processed_cpu["pixel_values"] = processed["pixel_values"].cpu()
                processed_cpu["image_grid_thw"] = processed["image_grid_thw"].cpu()

            batch_buffer.append(processed_cpu)
            batch_indices.append(i)

            # Save batch when full or at end of dataset
            if len(batch_buffer) >= BATCH_SIZE or i == len(val_data) - 1:
                # Save all samples in batch at once (reduces I/O overhead)
                for batch_idx, batch_sample in zip(batch_indices, batch_buffer):
                    sample_path = val_output_dir / f"sample_{batch_idx:06d}.pt"
                    torch.save(batch_sample, sample_path)
                    val_processed.append(str(sample_path))

                batch_buffer = []
                batch_indices = []

        except Exception as e:
            print(f"\n❌ Error processing val sample {i}: {e}")
            raise

    # Save metadata
    metadata = {
        "train_samples": len(train_processed),
        "val_samples": len(val_processed),
        "model_name": model_name,
        "dataset_name": dataset_name,
        "task": "routing",
        "num_classes": 3,
        "class_names": ["calendar", "claim-window", "provider-select"],
    }

    metadata_path = Path(preprocessed_path) / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
        f.flush()  # Ensure data is written to disk
        import os
        os.fsync(f.fileno())  # Force write to underlying storage

    print(f"\n✅ Preprocessing complete!")
    print(f"   Train samples: {len(train_processed)}")
    print(f"   Val samples: {len(val_processed)}")

    # Calculate total size
    total_size = sum(f.stat().st_size for f in Path(preprocessed_path).rglob("*.pt")) / (1024**3)
    print(f"   Total preprocessed size: {total_size:.2f} GB")

    # Verify metadata file was written
    if metadata_path.exists():
        print(f"✅ Metadata file exists: {metadata_path} ({metadata_path.stat().st_size} bytes)")
    else:
        print(f"❌ ERROR: Metadata file not found: {metadata_path}")

    # Explicitly commit the volume to ensure metadata.json is persisted
    print(f"\n📝 Committing volume to persist metadata...")
    VOLUME.commit()
    print(f"✅ Volume committed successfully")

    print(f"\n✅ Preprocessed data saved to Modal volume: {preprocessed_path}")

    print(f"\n{'='*80}")
    print("🎉 PREPROCESSING COMPLETE!")
    print(f"{'='*80}\n")
    print(f"Preprocessed data is now available in the 'moe-lora-data' volume")
    print(f"\nNext step:")
    print(f"  modal run modal/router_train.py --dataset-dir {dataset_name}")

    return {
        "train_samples": len(train_processed),
        "val_samples": len(val_processed),
        "total_size_gb": total_size,
    }


@app.local_entrypoint()
def main(dataset_dir: str):
    """
    Local entrypoint for running preprocessing.

    Usage:
        modal run modal/router_preprocess.py --dataset-dir routing_20251124_115228
    """
    print(f"\n{'='*80}")
    print("Submitting preprocessing job to Modal...")
    print(f"Dataset: {dataset_dir}")
    print(f"{'='*80}\n")

    result = preprocess_routing_dataset.remote(dataset_dir)

    print(f"\n{'='*80}")
    print("Preprocessing job completed!")
    print(f"{'='*80}\n")
    print(f"Results: {result}")
