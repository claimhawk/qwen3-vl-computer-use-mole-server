#!/usr/bin/env python3
# Copyright (c) 2025 Tylt LLC. All rights reserved.
"""Generate OCR routing dataset for training the router to identify Chandra tasks.

This creates JSONL entries that train the router to output "chandra" when it
sees OCR-related prompts.

Usage:
    python routing-generator/generate_ocr_routing.py --output datasets/ocr_routing.jsonl --count 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ocr_prompts import generate_ocr_prompts

# System prompt (loaded from canonical source)
SYSTEM_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "SYSTEM_PROMPT.txt"


def load_system_prompt() -> str:
    """Load the system prompt from the canonical file."""
    if SYSTEM_PROMPT_PATH.exists():
        return SYSTEM_PROMPT_PATH.read_text().strip()
    # Fallback minimal prompt
    return "You are a computer use agent."


def generate_ocr_routing_entry(prompt: str, system_prompt: str, entry_id: str) -> dict:
    """Generate a single routing dataset entry for OCR.

    The entry trains the router to output "chandra" for OCR tasks.
    """
    return {
        "id": entry_id,
        "conversations": [
            {"from": "system", "value": system_prompt},
            {"from": "human", "value": f"<image>\n{prompt}"},
            {"from": "gpt", "value": "chandra"},
        ],
        "image": "",  # Will be filled with actual OCR images
        "metadata": {
            "adapter": "chandra",
            "task_type": "ocr",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate OCR routing dataset for Chandra"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of entries to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args(argv)

    system_prompt = load_system_prompt()
    prompts = generate_ocr_prompts(args.count, seed=args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    for i, prompt in enumerate(prompts):
        entry_id = f"ocr_routing_{i:04d}"
        entry = generate_ocr_routing_entry(prompt, system_prompt, entry_id)
        entries.append(entry)

    with output_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Generated {len(entries)} OCR routing entries to {output_path}")
    print(f"\nSample entry:")
    print(json.dumps(entries[0], indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
