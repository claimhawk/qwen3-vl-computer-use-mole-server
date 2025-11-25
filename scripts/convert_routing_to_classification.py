#!/usr/bin/env python3
"""
Convert routing dataset to classification format for LoRA training.

Converts from routing format (with tool calls) to simple classification format
where the model outputs just the adapter name.
"""

import json
import argparse
from pathlib import Path


def convert_dataset(input_jsonl: Path, output_jsonl: Path):
    """Convert routing dataset to classification format."""

    system_prompt = (
        "You are a routing classifier for a multi-adapter system. "
        "Given a screenshot, identify which LoRA adapter should handle it. "
        "Respond with only the adapter name: calendar, claim-window, or provider-select."
    )

    converted_count = 0

    with open(input_jsonl, 'r') as f_in, open(output_jsonl, 'w') as f_out:
        for line in f_in:
            sample = json.loads(line)

            # Extract adapter label
            adapter = sample['adapter']

            # Create new conversations format for classification
            new_sample = {
                'id': sample['id'],
                'image': sample['image'],
                'conversations': [
                    {
                        'from': 'system',
                        'value': system_prompt
                    },
                    {
                        'from': 'human',
                        'value': '<image>\nWhich adapter should handle this screenshot?'
                    },
                    {
                        'from': 'gpt',
                        'value': adapter
                    }
                ],
                'metadata': {
                    'task_type': 'routing_classification',
                    'original_task': sample.get('metadata', {}).get('task_type', 'unknown'),
                    'label': sample.get('label', -1)
                }
            }

            f_out.write(json.dumps(new_sample) + '\n')
            converted_count += 1

    print(f"Converted {converted_count} samples")
    print(f"Output: {output_jsonl}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert routing dataset to classification format'
    )
    parser.add_argument(
        'dataset_dir',
        type=Path,
        help='Dataset directory (e.g., datasets/routing_20251124_150334)'
    )

    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return 1

    # Convert train and eval IN-PLACE (overwrite originals)
    for split in ['train', 'eval']:
        input_file = dataset_dir / f'{split}.jsonl'
        if not input_file.exists():
            print(f"Warning: {input_file} not found, skipping...")
            continue

        # Read original, convert to temp, then replace original
        temp_file = dataset_dir / f'{split}.jsonl.tmp'
        print(f"\nConverting {split} split...")
        convert_dataset(input_file, temp_file)

        # Replace original with converted version
        temp_file.replace(input_file)
        print(f"✅ Replaced {input_file} with classification format")

    print(f"\n✅ Classification dataset ready!")
    print(f"\nNext steps:")
    print(f"1. Train router LoRA using your existing training infrastructure")
    print(f"2. Point training script to: {dataset_dir}/train.jsonl")
    print(f"3. Model will learn to output: 'calendar', 'claim-window', or 'provider-select'")


if __name__ == '__main__':
    exit(main() or 0)
