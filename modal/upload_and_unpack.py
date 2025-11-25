#!/usr/bin/env python3
"""
Upload and unpack routing dataset to Modal volume.

Compresses local dataset, uploads to Modal, and unpacks on the volume.

Usage:
    modal run modal/upload_and_unpack.py --dataset-dir routing_20251124_133254
"""

import tarfile
import tempfile
from pathlib import Path

import modal

app = modal.App("upload-and-unpack")

VOLUME = modal.Volume.from_name("moe-lora-data", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.12")


@app.function(
    image=image,
    volumes={"/data": VOLUME},
    timeout=3600,  # 1 hour
)
def unpack_dataset(dataset_name: str, archive_path: str):
    """Unpack uploaded dataset archive on Modal volume."""
    import tarfile
    from pathlib import Path

    print(f"\n{'='*80}")
    print("📦 Unpacking dataset on Modal volume")
    print(f"{'='*80}\n")

    # Reload volume to see latest data
    VOLUME.reload()

    archive_file = Path(archive_path)
    output_dir = Path("/data/datasets") / dataset_name

    print(f"Archive: {archive_file}")
    print(f"Output: {output_dir}")

    if not archive_file.exists():
        raise FileNotFoundError(f"Archive not found: {archive_file}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract archive
    print(f"\nExtracting archive...")
    with tarfile.open(archive_file, "r:gz") as tar:
        tar.extractall(output_dir.parent)

    print(f"✅ Dataset unpacked to {output_dir}")

    # List files to verify
    files = list(output_dir.rglob("*"))
    print(f"\nExtracted {len(files)} files:")
    for f in sorted(files)[:10]:
        rel_path = f.relative_to(output_dir)
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {rel_path} ({size_mb:.1f} MB)")
        else:
            print(f"  {rel_path}/ (directory)")

    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more")

    # Clean up archive
    archive_file.unlink()
    print(f"\n🗑️  Removed archive: {archive_file}")

    # Commit volume
    VOLUME.commit()
    print(f"✅ Volume committed")

    return {
        "dataset_name": dataset_name,
        "output_dir": str(output_dir),
        "files_extracted": len(files),
    }


@app.local_entrypoint()
def main(dataset_dir: str):
    """
    Compress local dataset, upload to Modal, and unpack.

    Args:
        dataset_dir: Name of dataset directory (e.g., "routing_20251124_133254")
    """
    import os

    print(f"\n{'='*80}")
    print("🚀 Uploading dataset to Modal")
    print(f"{'='*80}\n")

    # Get local dataset path
    local_dataset_path = Path("datasets") / dataset_dir

    if not local_dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {local_dataset_path}")

    print(f"Local dataset: {local_dataset_path.absolute()}")

    # Create temporary tar.gz archive
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
        archive_path = Path(tmp_file.name)

    print(f"Creating archive: {archive_path}")

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(local_dataset_path, arcname=dataset_dir)

    archive_size_mb = archive_path.stat().st_size / (1024 * 1024)
    print(f"✅ Archive created ({archive_size_mb:.1f} MB)")

    # Upload archive to Modal volume
    remote_archive_path = f"/data/archives/{dataset_dir}.tar.gz"

    print(f"\nUploading to Modal volume...")
    print(f"Remote path: {remote_archive_path}")

    # Use modal CLI to upload the archive
    import subprocess

    cmd = [
        "uvx", "modal", "volume", "put",
        "--force",
        "moe-lora-data",
        str(archive_path),
        remote_archive_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Upload failed: {result.stderr}")
        archive_path.unlink()
        raise RuntimeError(f"Upload failed: {result.stderr}")

    print(f"✅ Archive uploaded")

    # Clean up local archive
    archive_path.unlink()
    print(f"🗑️  Removed local archive")

    # Unpack on Modal
    print(f"\n{'='*80}")
    print("📦 Unpacking on Modal...")
    print(f"{'='*80}\n")

    result = unpack_dataset.remote(dataset_dir, remote_archive_path)

    print(f"\n{'='*80}")
    print("✅ Upload and unpack complete!")
    print(f"{'='*80}\n")
    print(f"Dataset: {result['dataset_name']}")
    print(f"Location: {result['output_dir']}")
    print(f"Files: {result['files_extracted']}")
    print(f"\nNext step:")
    print(f"  modal run modal/router_preprocess.py --dataset-dir {dataset_dir}")
