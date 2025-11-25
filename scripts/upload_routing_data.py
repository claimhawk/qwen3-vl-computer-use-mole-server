#!/usr/bin/env python3
"""Upload routing dataset directory to Modal volume.

Uploads a local routing dataset directory (with .jsonl files and metadata.json)
to a Modal volume. This script uses the Modal CLI via subprocess to upload files.

Usage:
    python scripts/upload_routing_data.py datasets/routing_20251124_123456/
    python scripts/upload_routing_data.py datasets/routing_20251124_123456/ --volume moe-lora-data
    python scripts/upload_routing_data.py datasets/routing_20251124_123456/ --remote-dir custom-datasets

Directory structure (expected in local dataset dir):
    routing-train.jsonl
    routing-eval.jsonl
    metadata.json

These files will be uploaded to:
    {volume}:/datasets/{dataset_name}/routing-train.jsonl
    {volume}:/datasets/{dataset_name}/routing-eval.jsonl
    {volume}:/datasets/{dataset_name}/metadata.json
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Upload routing dataset to Modal volume."""
    parser = argparse.ArgumentParser(
        description="Upload routing dataset directory to Modal volume"
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Local dataset directory (e.g., datasets/routing_20251124_123456/)",
    )
    parser.add_argument(
        "--volume",
        default="moe-lora-data",
        help="Modal volume name (default: moe-lora-data)",
    )
    parser.add_argument(
        "--remote-dir",
        default="datasets",
        help="Remote directory prefix (default: datasets)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print upload commands without executing them",
    )

    args = parser.parse_args()

    # Validate local directory exists
    if not args.dataset_dir.exists():
        print(f"Error: Dataset directory not found: {args.dataset_dir}", file=sys.stderr)
        sys.exit(1)

    if not args.dataset_dir.is_dir():
        print(f"Error: Not a directory: {args.dataset_dir}", file=sys.stderr)
        sys.exit(1)

    # Get dataset name from directory
    dataset_name = args.dataset_dir.name

    # Files to upload (in order of preference)
    expected_files = [
        "routing-train.jsonl",
        "routing-eval.jsonl",
        "metadata.json",
    ]

    # Also check for alternative naming patterns
    alternative_patterns = [
        ("data.jsonl", "data.jsonl"),
        ("train.jsonl", "train.jsonl"),
        ("val.jsonl", "val.jsonl"),
    ]

    files_to_upload = []

    # First, collect the expected files if they exist
    for filename in expected_files:
        local_path = args.dataset_dir / filename
        if local_path.exists():
            files_to_upload.append((local_path, filename))

    # If no standard files found, check alternatives
    if not files_to_upload:
        for pattern, remote_name in alternative_patterns:
            local_path = args.dataset_dir / pattern
            if local_path.exists():
                files_to_upload.append((local_path, remote_name))

    if not files_to_upload:
        print(
            f"Error: No dataset files found in {args.dataset_dir}",
            file=sys.stderr,
        )
        print(
            f"Expected files: {', '.join(expected_files)}",
            file=sys.stderr,
        )
        print(
            f"Alternative files: {', '.join(p[0] for p in alternative_patterns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Print upload plan
    print("=" * 60)
    print(f"Uploading Dataset: {dataset_name}")
    print("=" * 60)
    print(f"Local directory:  {args.dataset_dir.absolute()}")
    print(f"Modal volume:     {args.volume}")
    print(f"Remote path:      {args.remote_dir}/{dataset_name}/")
    print(f"Files to upload:  {len(files_to_upload)}")
    print("=" * 60)
    print()

    # Upload each file
    failed_uploads = []

    for local_path, remote_filename in files_to_upload:
        remote_path = f"{args.remote_dir}/{dataset_name}/{remote_filename}"

        # Build the modal volume put command
        cmd = [
            "uvx",
            "modal",
            "volume",
            "put",
            "--force",
            args.volume,
            str(local_path),
            remote_path,
        ]

        print(f"Uploading {local_path.name} -> {args.volume}:{remote_path}")

        if args.dry_run:
            print(f"  [DRY RUN] Would run: {' '.join(cmd)}")
            continue

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            if result.stdout:
                print(f"  {result.stdout.strip()}")
            print(f"  ✓ Success")
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.strip() if e.stderr else str(e)
            print(f"  ✗ Failed: {error_msg}", file=sys.stderr)
            failed_uploads.append((local_path.name, error_msg))
        except FileNotFoundError:
            print(
                "  ✗ Error: 'uvx' or 'modal' command not found. "
                "Please install modal: pip install modal",
                file=sys.stderr,
            )
            sys.exit(1)

        print()

    # Summary
    print("=" * 60)
    if args.dry_run:
        print("DRY RUN: No files were uploaded")
    elif failed_uploads:
        print(f"Upload completed with {len(failed_uploads)} error(s):")
        for filename, error in failed_uploads:
            print(f"  - {filename}: {error}")
        sys.exit(1)
    else:
        print("✓ Upload complete!")
        print(f"  Uploaded {len(files_to_upload)} file(s) to {args.volume}:{args.remote_dir}/{dataset_name}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
