#!/usr/bin/env python3
"""
Qwen3-VL Data Preprocessing on Modal

Preprocess the raw JSONL + images dataset on Modal's CPU instances.
Saves preprocessed tensors to a Modal volume for reuse across training runs.

This eliminates the need to preprocess on expensive GPU instances and
speeds up training startup time.

Usage:
    modal run modal/preprocess.py --dataset-dir dataset
"""

import gzip
import json
import os
import tarfile
from pathlib import Path
from typing import Any

import modal

# Modal App Setup
app = modal.App("moe-lora-preprocessing")

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
def preprocess_dataset_impl(dataset_name: str):
    """
    Preprocess the dataset on Modal CPU instance.

    Loads raw JSONL + images from /dataset_data (mounted volume) and saves
    preprocessed tensors to /preprocessed_data (mounted volume).

    Args:
        dataset_name: Name of the dataset (for logging/output naming)
    """
    import torch
    from transformers import AutoProcessor
    from PIL import Image
    from tqdm import tqdm
    from qwen_vl_utils import process_vision_info

    # Reload the mounted volume to see latest committed data
    # This is critical - volumes mount with a snapshot and need explicit reload
    VOLUME.reload()

    # Get paths from environment or use defaults
    # Both raw and preprocessed data are in the same volume mounted at /data
    data_root = Path("/data")
    dataset_base_path = data_root / dataset_name

    # Extract base dataset name (get last path component: "datasets/routing_xxx" -> "routing_xxx")
    base_dataset_name = Path(dataset_name).name
    preprocessed_base_path = data_root / "preprocessed" / base_dataset_name

    print(f"\n{'='*80}")
    print("🚀 Starting Qwen3-VL Preprocessing on Modal")
    print(f"{'='*80}\n")
    print(f"Dataset: {dataset_name}")
    print(f"Data root: {data_root}")
    print(f"Dataset path: {dataset_base_path}")
    print(f"Output path: {preprocessed_base_path}")

    # ============================================================================
    # STEP 1: Load Dataset from Modal Volume
    # ============================================================================

    print(f"\n{'='*80}")
    print("📦 STEP 1: Loading Dataset")
    print(f"{'='*80}\n")

    # Check for chunked dataset (old format, unlikely with experiment datasets)
    chunks_dir = Path(dataset_base_path) / "chunks"
    if chunks_dir.exists():
        print("📦 Chunked dataset detected - reassembling...")

        # Find manifest
        manifest_path = chunks_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            print(f"   Total chunks: {manifest['total_chunks']}")
            print(f"   Archive size: {manifest['archive_size'] / (1024**2):.2f} MB")
            print(f"   Checksum: {manifest['checksum']}")

        # Find all chunks
        chunk_files = sorted(chunks_dir.glob("chunk_*"))
        print(f"   Found {len(chunk_files)} chunk files")

        # Reassemble
        reassemble_path = Path(dataset_base_path) / "dataset.tar.gz"
        print(f"   Reassembling to: {reassemble_path}")

        with open(reassemble_path, 'wb') as outfile:
            for chunk_path in tqdm(chunk_files, desc="Reassembling chunks"):
                with open(chunk_path, 'rb') as infile:
                    outfile.write(infile.read())

        print(f"   ✅ Reassembled: {os.path.getsize(reassemble_path) / (1024**2):.2f} MB")

        # Verify gzip integrity
        print("   Verifying gzip integrity...")
        try:
            with gzip.open(reassemble_path, 'rb') as gz:
                while gz.read(10 * 1024 * 1024):  # Read in 10MB chunks
                    pass
            print("   ✅ Gzip integrity verified")
        except Exception as e:
            print(f"   ❌ Gzip verification failed: {e}")
            raise

        # Extract
        print("   Extracting archive...")
        with tarfile.open(reassemble_path, "r:gz") as tar:
            tar.extractall(path=dataset_base_path)
        print("   ✅ Extraction complete")

        # Cleanup archive
        reassemble_path.unlink()
        print("   🧹 Cleaned up archive")

    # Find train and val files (or data.jsonl for ramiro datasets)
    dataset_path = Path(dataset_base_path)

    # Debug: List what's actually in the directory
    print(f"\n🔍 Debugging - Listing contents of {dataset_path}:")
    if dataset_path.exists():
        print(f"   Directory exists: YES")
        all_files = list(dataset_path.iterdir())
        print(f"   Found {len(all_files)} items:")
        for item in all_files[:20]:  # Show first 20 items
            print(f"      - {item.name} ({'dir' if item.is_dir() else 'file'})")
    else:
        print(f"   Directory exists: NO")
        print(f"   Parent directory: {dataset_path.parent}")
        if dataset_path.parent.exists():
            print(f"   Contents of parent:")
            for item in dataset_path.parent.iterdir():
                print(f"      - {item.name}")

    train_files = list(dataset_path.glob("train*.jsonl"))
    val_files = list(dataset_path.glob("val*.jsonl"))
    test_files = list(dataset_path.glob("test*.jsonl"))  # Also check for test.jsonl
    data_files = list(dataset_path.glob("data.jsonl"))

    # Load JSONL data
    def load_jsonl(path):
        data = []
        with open(path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    # Priority: Use existing train/val or train/test splits if available
    # Only fall back to data.jsonl with auto-split if no pre-split files exist
    if train_files and (val_files or test_files):
        # Use existing split files (preferred - preserves user's intended split)
        train_path = train_files[0]
        val_path = val_files[0] if val_files else test_files[0]

        print(f"\n✅ Using existing dataset split:")
        print(f"   Train: {train_path.name}")
        print(f"   Val: {val_path.name}")

        train_data = load_jsonl(train_path)
        val_data = load_jsonl(val_path)
    elif data_files:
        # Ramiro dataset format - single data.jsonl file, need to auto-split
        print(f"\n✅ Found ramiro dataset format (no pre-split files):")
        print(f"   Data file: {data_files[0].name}")
        print(f"   Auto-splitting 90/10...")

        all_data = load_jsonl(data_files[0])
        split_idx = int(len(all_data) * 0.9)
        train_data = all_data[:split_idx]
        val_data = all_data[split_idx:]
    else:
        raise FileNotFoundError(f"Could not find train*.jsonl/val*.jsonl, train*.jsonl/test*.jsonl, or data.jsonl in {dataset_path}")

    print(f"\n📊 Dataset size:")
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
        unique_images.add(sample["image"])

    print(f"Found {len(unique_images)} unique images (from {len(train_data) + len(val_data)} total samples)")
    print(f"Reuse ratio: {(len(train_data) + len(val_data)) / len(unique_images):.1f}x")

    # Cache for processed image tensors
    image_cache = {}

    print("\nProcessing unique images...")
    for img_path in tqdm(sorted(unique_images), desc="Caching images"):
        # Image paths in JSONL might be like "calendar_20251114_224610/images/screen_2024_01.png"
        # But dataset_base_path is "/training_data/calendar_20251114_224610/calendar_20251114_224610"
        # So we need to strip the first component if it matches the base dataset name
        img_path_str = str(img_path)

        # Extract the base dataset name (first part if nested like "calendar_x/calendar_x")
        base_dataset_name = dataset_name.split('/')[0] if '/' in dataset_name else dataset_name

        # If image path starts with the base dataset name, strip it to avoid duplication
        if img_path_str.startswith(f"{base_dataset_name}/"):
            # Strip the base name prefix: "calendar_20251114_224610/images/..." → "images/..."
            img_path_str = img_path_str[len(base_dataset_name)+1:]

        full_path = Path(dataset_base_path) / img_path_str
        if not full_path.exists():
            print(f"⚠️  Warning: Image not found: {full_path}")
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

    def prepare_sample(sample, dataset_dir, dataset_name, image_cache):
        """
        Prepare a single sample for training.
        Converts from JSONL format to model input tensors.
        Uses cached image embeddings for performance.
        """
        # Get image from cache (already processed)
        img_path = sample["image"]
        if img_path not in image_cache:
            raise FileNotFoundError(f"Image not in cache: {img_path}")

        # Get cached image data
        cached_image = image_cache[img_path]

        # Get conversations in old format:
        # [{"from": "human", "value": "<image>\nquery"}, {"from": "gpt", "value": "response"}]
        old_conversations = sample["conversations"]

        # Convert to Qwen-VL's expected format (but only for text, image is cached)
        # NOTE: System prompt should already be in the dataset - we just pass it through
        messages = []

        for msg in old_conversations:
            # Map dataset roles to model roles
            if msg["from"] == "system":
                role = "system"
            elif msg["from"] == "human":
                role = "user"
            else:  # "gpt" or "assistant"
                role = "assistant"
            content_list = []

            # Parse the value - check for <image> token
            value = msg["value"]
            if "<image>" in value:
                # For text template, still need to include image placeholder
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

        # Return processed sample with squeezed tensors
        input_ids = model_inputs["input_ids"][0] if isinstance(model_inputs["input_ids"][0], torch.Tensor) else torch.tensor(model_inputs["input_ids"][0])
        attention_mask = model_inputs["attention_mask"][0] if isinstance(model_inputs["attention_mask"][0], torch.Tensor) else torch.tensor(model_inputs["attention_mask"][0])

        # Create labels: Only train on assistant responses
        IGNORE_INDEX = -100
        labels = torch.full_like(input_ids, IGNORE_INDEX)

        # Find assistant response tokens (<|im_start|>assistant ... <|im_end|>)
        input_ids_list = input_ids.tolist()
        L = len(input_ids_list)
        pos = 0

        while pos < L:
            # Look for <|im_start|>assistant (token ID 77091)
            if input_ids_list[pos] == 77091:
                # Skip "<|im_start|>" and "assistant"
                ans_start = pos + 2
                ans_end = ans_start

                # Find <|im_end|> (token ID 151645)
                while ans_end < L and input_ids_list[ans_end] != 151645:
                    ans_end += 1

                if ans_end < L:
                    # Label the assistant response INCLUDING <|im_end|> and the token after it
                    labels[ans_start : ans_end + 2] = input_ids[ans_start : ans_end + 2]
                    pos = ans_end
            pos += 1

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if "pixel_values" in model_inputs:
            result["pixel_values"] = model_inputs["pixel_values"]
            result["image_grid_thw"] = model_inputs["image_grid_thw"]

        return result

    # Process training data with batching for better performance
    print("Processing training data with batching...")
    train_processed = []
    train_output_dir = Path(preprocessed_base_path) / "train"
    train_output_dir.mkdir(parents=True, exist_ok=True)

    BATCH_SIZE = 8  # Process 8 samples at once
    batch_buffer = []
    batch_indices = []

    for i, sample in enumerate(tqdm(train_data, desc="Train")):
        try:
            processed = prepare_sample(sample, dataset_base_path, dataset_name, image_cache)

            # Convert tensors to CPU and add to batch buffer
            processed_cpu = {
                "input_ids": processed["input_ids"].cpu(),
                "attention_mask": processed["attention_mask"].cpu(),
                "labels": processed["labels"].cpu(),
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

    # Process validation data with batching
    print("\nProcessing validation data with batching...")
    val_processed = []
    val_output_dir = Path(preprocessed_base_path) / "val"
    val_output_dir.mkdir(parents=True, exist_ok=True)

    batch_buffer = []
    batch_indices = []

    for i, sample in enumerate(tqdm(val_data, desc="Val")):
        try:
            processed = prepare_sample(sample, dataset_base_path, dataset_name, image_cache)

            # Convert tensors to CPU and add to batch buffer
            processed_cpu = {
                "input_ids": processed["input_ids"].cpu(),
                "attention_mask": processed["attention_mask"].cpu(),
                "labels": processed["labels"].cpu(),
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
    }

    metadata_path = Path(preprocessed_base_path) / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Preprocessing complete!")
    print(f"   Train samples: {len(train_processed)}")
    print(f"   Val samples: {len(val_processed)}")

    # Calculate total size
    total_size = sum(f.stat().st_size for f in Path(preprocessed_base_path).rglob("*.pt")) / (1024**3)
    print(f"   Total preprocessed size: {total_size:.2f} GB")

    # Note: volume commit happens automatically when function exits
    print(f"\n✅ Preprocessed data saved to Modal volume: {preprocessed_base_path}")

    print(f"\n{'='*80}")
    print("🎉 PREPROCESSING COMPLETE!")
    print(f"{'='*80}\n")
    print(f"Preprocessed data is now available in the 'claimhawk-training-data' volume")
    print(f"\nNext step:")
    print(f"  modal run modal/training.py --run-name my_training_run")

    return {
        "train_samples": len(train_processed),
        "val_samples": len(val_processed),
        "total_size_gb": total_size,
    }


# Distributed preprocessing function
@app.function(
    image=image,
    cpu=8,  # Smaller per-container for parallelization
    memory=16384,  # 16GB RAM
    timeout=3600,  # 1 hour max per batch
)
def preprocess_batch(batch_info: dict):
    """
    Process a batch of samples in parallel.

    Args:
        batch_info: Dict with 'samples', 'batch_idx', 'split', 'dataset_name', 'dataset_path', 'output_path'
    """
    import torch
    from transformers import AutoProcessor
    from PIL import Image
    from tqdm import tqdm
    from qwen_vl_utils import process_vision_info

    samples = batch_info['samples']
    batch_idx = batch_info['batch_idx']
    split = batch_info['split']  # 'train' or 'val'
    dataset_name = batch_info['dataset_name']
    dataset_path = batch_info['dataset_path']
    output_path = batch_info['output_path']

    print(f"\n[Batch {batch_idx}] Processing {len(samples)} {split} samples")
    print(f"Dataset: {dataset_name}")

    # Load processor (cached across batches)
    model_name = "Qwen/Qwen3-VL-8B-Instruct"
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    def prepare_sample(sample, dataset_dir):
        """Prepare a single sample for training."""
        image_path = Path(dataset_dir) / sample["image"]
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        old_conversations = sample["conversations"]
        messages = []

        # Add computer use system prompt (critical for 32B model alignment)
        description_prompt = """Use a mouse and keyboard to interact with a computer, and take screenshots.
* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.
* The screen's resolution is 1000x1000.
* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.
* If you tried clicking on a program or link but it failed to load even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked."""

        action_description_prompt = """* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.
* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.
* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen (simulated as double-click since it's the closest action).
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `hscroll`: Performs a horizontal scroll (mapped to regular scroll).
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
* `answer`: Answer a question."""

        tools_def = {
            "type": "function",
            "function": {
                "name_for_human": "computer_use",
                "name": "computer_use",
                "description": description_prompt,
                "parameters": {
                    "properties": {
                        "action": {
                            "description": action_description_prompt,
                            "enum": ["key", "type", "mouse_move", "left_click", "left_click_drag", "right_click", "middle_click", "double_click", "scroll", "wait", "terminate"],
                            "type": "string"
                        },
                        "keys": {"description": "Required only by `action=key`.", "type": "array"},
                        "text": {"description": "Required only by `action=type`.", "type": "string"},
                        "coordinate": {"description": "The x,y coordinates for mouse actions.", "type": "array"},
                        "pixels": {"description": "The amount of scrolling.", "type": "number"},
                        "time": {"description": "The seconds to wait.", "type": "number"},
                        "status": {
                            "description": "The status of the task.",
                            "type": "string",
                            "enum": ["success", "failure"]
                        }
                    },
                    "required": ["action"],
                    "type": "object"
                },
                "args_format": "Format the arguments as a JSON object."
            }
        }

        system_prompt = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
""" + json.dumps(tools_def) + """
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

# Response format

Response format for every step:
1) Action: a short imperative describing what to do in the UI.
2) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.

Rules:
- Output exactly in the order: Action, <tool_call>.
- Be brief: one sentence for Action.
- Do not output anything else outside those parts.
- If finishing, use action=terminate in the tool call."""

        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        })

        for msg in old_conversations:
            role = "user" if msg["from"] == "human" else "assistant"
            content_list = []
            value = msg["value"]

            if "<image>" in value:
                content_list.append({"type": "image", "image": f"file://{image_path}"})
                text = value.replace("<image>", "").strip()
                if text:
                    content_list.append({"type": "text", "text": text})
            else:
                content_list.append({"type": "text", "text": value})

            messages.append({"role": role, "content": content_list})

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)

        model_inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=False,
            do_resize=False
        )

        result = {
            "input_ids": model_inputs["input_ids"][0] if isinstance(model_inputs["input_ids"][0], torch.Tensor) else torch.tensor(model_inputs["input_ids"][0]),
            "attention_mask": model_inputs["attention_mask"][0] if isinstance(model_inputs["attention_mask"][0], torch.Tensor) else torch.tensor(model_inputs["attention_mask"][0]),
        }

        if "pixel_values" in model_inputs:
            result["pixel_values"] = model_inputs["pixel_values"]
            result["image_grid_thw"] = model_inputs["image_grid_thw"]

        return result

    # Mount volumes dynamically
    experiment_datasets_volume = modal.Volume.from_name("experiment-datasets")
    experiment_preprocessed_volume = modal.Volume.from_name("experiment-preprocessed", create_if_missing=True)

    # Reload to access data
    experiment_datasets_volume.reload()
    experiment_preprocessed_volume.reload()

    # Process batch
    processed_files = []
    output_dir = Path(output_path) / split
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(tqdm(samples, desc=f"Batch {batch_idx}")):
        try:
            processed = prepare_sample(sample, dataset_path)

            # Convert tensors to CPU
            processed_cpu = {
                "input_ids": processed["input_ids"].cpu(),
                "attention_mask": processed["attention_mask"].cpu(),
            }
            if "pixel_values" in processed:
                processed_cpu["pixel_values"] = processed["pixel_values"].cpu()
                processed_cpu["image_grid_thw"] = processed["image_grid_thw"].cpu()

            # Save with batch-aware naming
            sample_idx = batch_info['start_idx'] + i
            sample_path = output_dir / f"sample_{sample_idx:06d}.pt"
            torch.save(processed_cpu, sample_path)
            processed_files.append(str(sample_path))

        except Exception as e:
            print(f"\n❌ Error processing sample {i} in batch {batch_idx}: {e}")
            raise

    print(f"[Batch {batch_idx}] ✅ Processed {len(processed_files)} samples")

    # Commit volume changes
    experiment_preprocessed_volume.commit()

    return {
        "batch_idx": batch_idx,
        "split": split,
        "samples_processed": len(processed_files),
        "files": processed_files
    }


@app.local_entrypoint()
def main(dataset_name: str = "claimhawk-training-data", parallel: bool = False, batch_size: int = 100):
    """
    Local entrypoint for running preprocessing.

    Usage:
        # Sequential (single container):
        modal run modal/preprocess.py --dataset-name ramiro-2511050557-prod

        # Parallel (distributed across many containers):
        modal run modal/preprocess.py --dataset-name ramiro-2511050557-prod --parallel

        # Custom batch size:
        modal run modal/preprocess.py --dataset-name ramiro-2511050557-prod --parallel --batch-size 50
    """
    print(f"\n{'='*80}")
    print("Submitting preprocessing job to Modal...")
    print(f"Dataset: {dataset_name}")
    print(f"Mode: {'PARALLEL' if parallel else 'SEQUENTIAL'}")
    if parallel:
        print(f"Batch size: {batch_size} samples per container")
    print(f"{'='*80}\n")

    if not parallel:
        # Sequential mode - call preprocessing function directly
        result = preprocess_dataset_impl.remote(dataset_name)

        print(f"\n{'='*80}")
        print("Preprocessing job completed!")
        print(f"{'='*80}\n")
        print(f"Results: {result}")

    else:
        # Parallel mode - distributed preprocessing
        print("Loading dataset for distributed processing...")

        # Create coordinator app to load data and dispatch batches
        coordinator_app = modal.App(f"moe-lora-preprocessing-coordinator-{dataset_name}")
        training_data_volume = modal.Volume.from_name("claimhawk-training-data")
        preprocessed_data_volume = modal.Volume.from_name("claimhawk-training-data", create_if_missing=True)

        @coordinator_app.function(
            image=image,
            cpu=2,
            memory=4096,
            timeout=1800,
            volumes={
                "/training_data": training_data_volume,
                "/preprocessed_data": preprocessed_data_volume,
            },
            serialized=True,
        )
        def coordinate_preprocessing():
            """Load data and create batch specifications for parallel processing."""
            import json
            from pathlib import Path

            dataset_path = f"/training_data/{dataset_name}"
            # Extract base dataset name (remove nested path)
            base_dataset_name = dataset_name.split('/')[0] if '/' in dataset_name else dataset_name
            output_path = f"/preprocessed_data/{base_dataset_name}"

            print(f"Loading data from: {dataset_path}")

            # Load data.jsonl
            data_file = Path(dataset_path) / "data.jsonl"
            if not data_file.exists():
                raise FileNotFoundError(f"Dataset file not found: {data_file}")

            # Load all samples
            all_data = []
            with open(data_file, 'r') as f:
                for line in f:
                    all_data.append(json.loads(line))

            print(f"Loaded {len(all_data)} samples")

            # Split into train/val (90/10)
            split_idx = int(len(all_data) * 0.9)
            train_data = all_data[:split_idx]
            val_data = all_data[split_idx:]

            print(f"Train: {len(train_data)} samples, Val: {len(val_data)} samples")

            # Create batches
            batches = []

            def create_batches(data, split_name, batch_size):
                batch_list = []
                for i in range(0, len(data), batch_size):
                    batch_samples = data[i:i + batch_size]
                    batch_list.append({
                        'samples': batch_samples,
                        'batch_idx': len(batch_list),
                        'split': split_name,
                        'dataset_name': dataset_name,
                        'dataset_path': dataset_path,
                        'output_path': output_path,
                        'start_idx': i,
                    })
                return batch_list

            train_batches = create_batches(train_data, 'train', batch_size)
            val_batches = create_batches(val_data, 'val', batch_size)

            all_batches = train_batches + val_batches

            print(f"\nCreated {len(train_batches)} train batches + {len(val_batches)} val batches")
            print(f"Total batches: {len(all_batches)}")
            print(f"Batch size: {batch_size} samples per batch")

            return {
                'batches': all_batches,
                'train_samples': len(train_data),
                'val_samples': len(val_data),
                'num_batches': len(all_batches),
            }

        print("Step 1: Coordinating batch creation...")
        with coordinator_app.run():
            coord_result = coordinate_preprocessing.remote()

        batches = coord_result['batches']
        print(f"\n✅ Created {len(batches)} batches")
        print(f"   Train samples: {coord_result['train_samples']}")
        print(f"   Val samples: {coord_result['val_samples']}")

        print(f"\nStep 2: Processing {len(batches)} batches in parallel...")
        print("This will spawn multiple containers simultaneously...")

        # Mount volumes for batch processing
        batch_app = modal.App(f"moe-lora-batch-processing-{dataset_name}")

        @batch_app.function(
            image=image,
            cpu=8,
            memory=16384,
            timeout=3600,
            volumes={
                "/training_data": training_data_volume,
                "/preprocessed_data": preprocessed_data_volume,
            },
            serialized=True,
        )
        def process_batch_wrapper(batch_info):
            """Process a batch - inline implementation for proper Modal execution."""
            import torch
            from transformers import AutoProcessor
            from PIL import Image
            from tqdm import tqdm
            from qwen_vl_utils import process_vision_info
            from pathlib import Path

            samples = batch_info['samples']
            batch_idx = batch_info['batch_idx']
            split = batch_info['split']
            dataset_name_local = batch_info['dataset_name']
            dataset_path = batch_info['dataset_path']
            output_path = batch_info['output_path']

            print(f"\n[Batch {batch_idx}] Processing {len(samples)} {split} samples")

            # Load processor
            model_name = "Qwen/Qwen3-VL-8B-Instruct"
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

            def prepare_sample(sample, dataset_dir):
                """Prepare a single sample for training."""
                image_path = Path(dataset_dir) / sample["image"]
                if not image_path.exists():
                    raise FileNotFoundError(f"Image not found: {image_path}")

                old_conversations = sample["conversations"]
                messages = []

                # Add computer use system prompt (REQUIRED for train/inference alignment)
                description_prompt = """Use a mouse and keyboard to interact with a computer, and take screenshots.
* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.
* The screen's resolution is 1000x1000.
* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.
* If you tried clicking on a program or link but it failed to load even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked."""

                action_description_prompt = """* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.
* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.
* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen (simulated as double-click since it's the closest action).
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `hscroll`: Performs a horizontal scroll (mapped to regular scroll).
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
* `answer`: Answer a question."""

                tools_def = {
                    "type": "function",
                    "function": {
                        "name_for_human": "computer_use",
                        "name": "computer_use",
                        "description": description_prompt,
                        "parameters": {
                            "properties": {
                                "action": {
                                    "description": action_description_prompt,
                                    "enum": ["key", "type", "mouse_move", "left_click", "left_click_drag", "right_click", "middle_click", "double_click", "scroll", "wait", "terminate"],
                                    "type": "string"
                                },
                                "keys": {"description": "Required only by `action=key`.", "type": "array"},
                                "text": {"description": "Required only by `action=type`.", "type": "string"},
                                "coordinate": {"description": "The x,y coordinates for mouse actions.", "type": "array"},
                                "pixels": {"description": "The amount of scrolling.", "type": "number"},
                                "time": {"description": "The seconds to wait.", "type": "number"},
                                "status": {
                                    "description": "The status of the task.",
                                    "type": "string",
                                    "enum": ["success", "failure"]
                                }
                            },
                            "required": ["action"],
                            "type": "object"
                        },
                        "args_format": "Format the arguments as a JSON object."
                    }
                }

                system_prompt = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
""" + json.dumps(tools_def) + """
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

# Response format

Response format for every step:
1) Action: a short imperative describing what to do in the UI.
2) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.

Rules:
- Output exactly in the order: Action, <tool_call>.
- Be brief: one sentence for Action.
- Do not output anything else outside those parts.
- If finishing, use action=terminate in the tool call."""

                messages.append({
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}]
                })

                for msg in old_conversations:
                    role = "user" if msg["from"] == "human" else "assistant"
                    content_list = []
                    value = msg["value"]

                    if "<image>" in value:
                        content_list.append({"type": "image", "image": f"file://{image_path}"})
                        text = value.replace("<image>", "").strip()
                        if text:
                            content_list.append({"type": "text", "text": text})
                    else:
                        content_list.append({"type": "text", "text": value})

                    messages.append({"role": role, "content": content_list})

                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)

                model_inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    return_tensors="pt",
                    padding=False,
                    do_resize=False
                )

                result = {
                    "input_ids": model_inputs["input_ids"][0] if isinstance(model_inputs["input_ids"][0], torch.Tensor) else torch.tensor(model_inputs["input_ids"][0]),
                    "attention_mask": model_inputs["attention_mask"][0] if isinstance(model_inputs["attention_mask"][0], torch.Tensor) else torch.tensor(model_inputs["attention_mask"][0]),
                }

                if "pixel_values" in model_inputs:
                    result["pixel_values"] = model_inputs["pixel_values"]
                    result["image_grid_thw"] = model_inputs["image_grid_thw"]

                return result

            # Process batch
            processed_files = []
            output_dir = Path(output_path) / split
            output_dir.mkdir(parents=True, exist_ok=True)

            for i, sample in enumerate(tqdm(samples, desc=f"Batch {batch_idx}")):
                try:
                    processed = prepare_sample(sample, dataset_path)

                    # Convert tensors to CPU
                    processed_cpu = {
                        "input_ids": processed["input_ids"].cpu(),
                        "attention_mask": processed["attention_mask"].cpu(),
                    }
                    if "pixel_values" in processed:
                        processed_cpu["pixel_values"] = processed["pixel_values"].cpu()
                        processed_cpu["image_grid_thw"] = processed["image_grid_thw"].cpu()

                    # Save with batch-aware naming
                    sample_idx = batch_info['start_idx'] + i
                    sample_path = output_dir / f"sample_{sample_idx:06d}.pt"
                    torch.save(processed_cpu, sample_path)
                    processed_files.append(str(sample_path))

                except Exception as e:
                    print(f"\n❌ Error processing sample {i} in batch {batch_idx}: {e}")
                    raise

            print(f"[Batch {batch_idx}] ✅ Processed {len(processed_files)} samples")

            return {
                "batch_idx": batch_idx,
                "split": split,
                "samples_processed": len(processed_files),
                "files": processed_files
            }

        # Process all batches in parallel using .map()
        with batch_app.run():
            results = list(process_batch_wrapper.map(batches))

        # Aggregate results
        total_processed = sum(r['samples_processed'] for r in results)
        train_processed = sum(r['samples_processed'] for r in results if r['split'] == 'train')
        val_processed = sum(r['samples_processed'] for r in results if r['split'] == 'val')

        print(f"\n{'='*80}")
        print("🎉 PARALLEL PREPROCESSING COMPLETE!")
        print(f"{'='*80}")
        print(f"Total samples processed: {total_processed}")
        print(f"  Train: {train_processed}")
        print(f"  Val: {val_processed}")
        print(f"Batches processed: {len(results)}")
        print(f"Containers used: {len(batches)}")

        # Create metadata file
        metadata = {
            "train_samples": train_processed,
            "val_samples": val_processed,
            "total_samples": total_processed,
            "dataset_name": dataset_name,
            "batch_size": batch_size,
            "num_batches": len(batches),
            "mode": "parallel",
        }

        print(f"\nMetadata: {metadata}")
