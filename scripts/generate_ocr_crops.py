# Copyright (c) 2025 Tylt LLC. All rights reserved.
# CONFIDENTIAL AND PROPRIETARY. Unauthorized use, copying, or distribution
# is strictly prohibited. For licensing inquiries: hello@claimhawk.app

"""Generate OCR training images by cropping grids and dropdowns from generator datasets.

This script extracts text-heavy regions from existing generator images:
- Grids: Procedure grids from claim-window screenshots
- Dropdowns: Provider dropdown overlays from provider-select screenshots

Output: Cropped images for OCR/chandra routing training.

Usage:
    python scripts/generate_ocr_crops.py --output-dir datasets/ocr

The script will:
1. Find generator datasets (claim-window, provider-select)
2. Extract grid regions and dropdown overlays
3. Generate READ prompts for chandra routing
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

from PIL import Image


# Crop region configurations
CROP_CONFIGS = {
    "claim-window-grid": {
        # From canvas.yaml: grid y_start: 217, y_end: 384
        # Full width minus scrollbar
        "x": 0,
        "y": 217,
        "width": 1098,  # 1117 - 19 (scrollbar)
        "height": 167,  # 384 - 217
        "description": "procedure grid",
    },
    "provider-select-dropdown": {
        # Dropdown config: width=150, item_height=16, 6-18 items
        # Position is derived from field config
        "width": 150,
        "item_height": 16,
        "description": "provider dropdown",
    },
}

# Field positions from canvas.yaml - dropdown attaches below field
FIELD_POSITIONS = {
    "billing_provider": {"x": 351, "y": 98, "width": 155, "height": 21},
    "treating_provider": {"x": 351, "y": 119, "width": 155, "height": 21},
}

# OCR prompt templates for chandra routing
OCR_PROMPT_TEMPLATES = [
    "Read the text in this image",
    "Extract all text from this image",
    "What text is shown in this image?",
    "Read and transcribe the text",
    "OCR this image and return the text",
    "What does this image say?",
    "Read the contents of this image",
    "Extract the text content",
    "Transcribe the text in this image",
    "Read all visible text",
]


def crop_claim_window_grid(
    image_path: Path,
    output_dir: Path,
    idx: int,
) -> dict | None:
    """Crop the procedure grid from a claim-window image.

    Args:
        image_path: Path to the claim-window image
        output_dir: Output directory for cropped images
        idx: Index for output filename

    Returns:
        Record dict with image path and prompt, or None if failed
    """
    try:
        img = Image.open(image_path)
        config = CROP_CONFIGS["claim-window-grid"]

        # Crop the grid region
        x, y = config["x"], config["y"]
        w, h = config["width"], config["height"]

        # Ensure we don't go out of bounds
        img_w, img_h = img.size
        if x + w > img_w or y + h > img_h:
            return None

        cropped = img.crop((x, y, x + w, y + h))

        # Save cropped image
        output_path = output_dir / f"grid_{idx:05d}.png"
        cropped.save(output_path)

        # Generate OCR prompt
        prompt = random.choice(OCR_PROMPT_TEMPLATES)

        return {
            "id": f"ocr_grid_{idx:05d}",
            "image": f"images/ocr/grid_{idx:05d}.png",
            "source": str(image_path.name),
            "region_type": "grid",
            "prompt": prompt,
        }
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def crop_provider_dropdown(
    image_path: Path,
    metadata: dict,
    output_dir: Path,
    idx: int,
) -> dict | None:
    """Crop the dropdown overlay from a provider-select image.

    Args:
        image_path: Path to the provider-select image
        metadata: Sample metadata containing field and dropdown_items
        output_dir: Output directory for cropped images
        idx: Index for output filename

    Returns:
        Record dict with image path and prompt, or None if failed
    """
    try:
        # Get dropdown items and field from metadata
        dropdown_items = metadata.get("dropdown_items", [])
        field_name = metadata.get("field")

        if not dropdown_items or not field_name:
            return None

        # Get field position to derive dropdown position
        # dropdown_position could be in metadata, or we derive from field
        dropdown_pos = metadata.get("dropdown_position")
        if not dropdown_pos:
            field_config = FIELD_POSITIONS.get(field_name)
            if not field_config:
                return None
            # Dropdown position: same x as field, y offset -19 (from click_dropdown.py)
            dropdown_pos = {
                "x": field_config["x"],
                "y": field_config["y"] - 19,
            }

        img = Image.open(image_path)

        # Calculate dropdown bounds
        x = dropdown_pos["x"]
        y = dropdown_pos["y"]
        width = CROP_CONFIGS["provider-select-dropdown"]["width"]
        item_height = CROP_CONFIGS["provider-select-dropdown"]["item_height"]
        height = len(dropdown_items) * item_height + 2  # +2 for border

        # Ensure we don't go out of bounds
        img_w, img_h = img.size
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x + width > img_w:
            width = img_w - x
        if y + height > img_h:
            height = img_h - y

        cropped = img.crop((x, y, x + width, y + height))

        # Save cropped image
        output_path = output_dir / f"dropdown_{idx:05d}.png"
        cropped.save(output_path)

        # Generate OCR prompt
        prompt = random.choice(OCR_PROMPT_TEMPLATES)

        return {
            "id": f"ocr_dropdown_{idx:05d}",
            "image": f"images/ocr/dropdown_{idx:05d}.png",
            "source": str(image_path.name),
            "region_type": "dropdown",
            "prompt": prompt,
            "expected_text": dropdown_items,
        }
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def find_generator_datasets(generators_dir: Path) -> dict[str, list[Path]]:
    """Find available generator datasets.

    Returns:
        Dict mapping generator type to list of dataset paths
    """
    datasets = {
        "claim-window": [],
        "provider-select": [],
    }

    # Claim window datasets
    claim_window_gen = generators_dir / "claim-window-generator" / "datasets"
    if claim_window_gen.exists():
        for ds in claim_window_gen.iterdir():
            if ds.is_dir() and ds.name.startswith("claim-window"):
                datasets["claim-window"].append(ds)
            elif ds.is_dir() and ds.name.startswith("provider-select"):
                datasets["provider-select"].append(ds)

    return datasets


def process_claim_window_dataset(
    dataset_path: Path,
    output_dir: Path,
    start_idx: int,
    max_samples: int = 100,
) -> tuple[list[dict], int]:
    """Process a claim-window dataset and extract grid crops.

    Args:
        dataset_path: Path to the dataset
        output_dir: Output directory for cropped images
        start_idx: Starting index for output filenames
        max_samples: Maximum number of samples to process

    Returns:
        Tuple of (list of records, next index)
    """
    records = []
    idx = start_idx

    # Read train.jsonl to get image paths
    train_file = dataset_path / "train.jsonl"
    if not train_file.exists():
        print(f"No train.jsonl found in {dataset_path}")
        return records, idx

    images_dir = dataset_path / "images"
    if not images_dir.exists():
        print(f"No images directory found in {dataset_path}")
        return records, idx

    # Get unique images (same image may appear multiple times with different tasks)
    seen_images: set[str] = set()

    with open(train_file) as f:
        for line in f:
            if len(records) >= max_samples:
                break

            sample = json.loads(line)
            image_rel_path = sample.get("image", "")

            if image_rel_path in seen_images:
                continue
            seen_images.add(image_rel_path)

            image_path = dataset_path / image_rel_path
            if not image_path.exists():
                continue

            record = crop_claim_window_grid(image_path, output_dir, idx)
            if record:
                records.append(record)
                idx += 1

    return records, idx


def process_provider_select_dataset(
    dataset_path: Path,
    output_dir: Path,
    start_idx: int,
    max_samples: int = 100,
) -> tuple[list[dict], int]:
    """Process a provider-select dataset and extract dropdown crops.

    Args:
        dataset_path: Path to the dataset
        output_dir: Output directory for cropped images
        start_idx: Starting index for output filenames
        max_samples: Maximum number of samples to process

    Returns:
        Tuple of (list of records, next index)
    """
    records = []
    idx = start_idx

    # Read train.jsonl to get image paths and metadata
    train_file = dataset_path / "train.jsonl"
    if not train_file.exists():
        print(f"No train.jsonl found in {dataset_path}")
        return records, idx

    # Get unique images with dropdown metadata
    seen_images: set[str] = set()

    with open(train_file) as f:
        for line in f:
            if len(records) >= max_samples:
                break

            sample = json.loads(line)
            metadata = sample.get("metadata", {})

            # Only process select-provider tasks (they have dropdown overlays)
            if metadata.get("task_type") != "select-provider":
                continue

            image_rel_path = sample.get("image", "")
            if image_rel_path in seen_images:
                continue
            seen_images.add(image_rel_path)

            image_path = dataset_path / image_rel_path
            if not image_path.exists():
                continue

            record = crop_provider_dropdown(image_path, metadata, output_dir, idx)
            if record:
                records.append(record)
                idx += 1

    return records, idx


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate OCR training images from generator datasets"
    )
    parser.add_argument(
        "--generators-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent / "generators",
        help="Path to generators directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "datasets" / "ocr",
        help="Output directory for OCR dataset",
    )
    parser.add_argument(
        "--max-grids",
        type=int,
        default=200,
        help="Maximum number of grid crops to generate",
    )
    parser.add_argument(
        "--max-dropdowns",
        type=int,
        default=200,
        help="Maximum number of dropdown crops to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()
    random.seed(args.seed)

    # Create output directories
    output_dir = args.output_dir
    images_dir = output_dir / "images" / "ocr"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Looking for datasets in: {args.generators_dir}")
    datasets = find_generator_datasets(args.generators_dir)

    print(f"Found {len(datasets['claim-window'])} claim-window datasets")
    print(f"Found {len(datasets['provider-select'])} provider-select datasets")

    all_records: list[dict] = []
    idx = 0

    # Process claim-window datasets (extract grids)
    for ds_path in datasets["claim-window"]:
        print(f"Processing claim-window dataset: {ds_path.name}")
        remaining = args.max_grids - sum(1 for r in all_records if r["region_type"] == "grid")
        if remaining <= 0:
            break
        records, idx = process_claim_window_dataset(ds_path, images_dir, idx, remaining)
        all_records.extend(records)
        print(f"  Extracted {len(records)} grid crops")

    # Process provider-select datasets (extract dropdowns)
    for ds_path in datasets["provider-select"]:
        print(f"Processing provider-select dataset: {ds_path.name}")
        remaining = args.max_dropdowns - sum(1 for r in all_records if r["region_type"] == "dropdown")
        if remaining <= 0:
            break
        records, idx = process_provider_select_dataset(ds_path, images_dir, idx, remaining)
        all_records.extend(records)
        print(f"  Extracted {len(records)} dropdown crops")

    # Write train.jsonl in the format expected by routing generator
    train_file = output_dir / "train.jsonl"
    with open(train_file, "w") as f:
        for record in all_records:
            # Format for routing training
            entry = {
                "id": record["id"],
                "image": record["image"],
                "conversations": [
                    {"from": "human", "value": f"<image>\n{record['prompt']}"},
                    {"from": "gpt", "value": "chandra"},  # OCR routes to chandra
                ],
                "metadata": {
                    "source": record.get("source"),
                    "region_type": record["region_type"],
                },
            }
            f.write(json.dumps(entry) + "\n")

    # Write config.json
    config = {
        "name": "ocr-crops",
        "generated_at": datetime.now().isoformat(),
        "grid_count": sum(1 for r in all_records if r["region_type"] == "grid"),
        "dropdown_count": sum(1 for r in all_records if r["region_type"] == "dropdown"),
        "total_count": len(all_records),
        "seed": args.seed,
    }
    config_file = output_dir / "config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nGenerated OCR dataset:")
    print(f"  Grid crops: {config['grid_count']}")
    print(f"  Dropdown crops: {config['dropdown_count']}")
    print(f"  Total: {config['total_count']}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
