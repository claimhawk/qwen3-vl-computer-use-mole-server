#!/usr/bin/env python3
"""Deploy routing LoRA checkpoint to inference path.

Usage:
    modal run modal/deploy.py --checkpoint checkpoints/datasets/routing_20251202_185620/routing-20251202-185620/final
    modal run modal/deploy.py --latest
"""

import shutil
from pathlib import Path

import modal

# =============================================================================
# CENTRALIZED CONFIGURATION
# =============================================================================
# Volume names are loaded from config/adapters.yaml via the SDK.
# Users can customize these by editing the YAML file.

try:
    from sdk.modal_compat import get_volume_name
    MOE_VOLUME_NAME = get_volume_name("moe_data")
    INFERENCE_VOLUME_NAME = get_volume_name("inference")
except ImportError:
    # Fallback for Modal remote execution
    MOE_VOLUME_NAME = "moe-lora-data"
    INFERENCE_VOLUME_NAME = "moe-inference"

app = modal.App("deploy-routing")

# Source volume (training checkpoints)
source_volume = modal.Volume.from_name(MOE_VOLUME_NAME)

# Target volume (inference server)
inference_volume = modal.Volume.from_name(INFERENCE_VOLUME_NAME, create_if_missing=True)


@app.function(volumes={"/moe-data": source_volume}, timeout=300)
def find_latest_checkpoint() -> str | None:
    """Find the latest routing checkpoint."""
    checkpoints_base = Path("/moe-data/checkpoints/datasets")

    if not checkpoints_base.exists():
        print("ERROR: No checkpoints directory found")
        return None

    # Find all routing datasets, sorted by name (timestamp) descending
    routing_dirs = sorted(
        [d for d in checkpoints_base.iterdir() if d.name.startswith("routing_")],
        reverse=True,
    )

    if not routing_dirs:
        print("ERROR: No routing checkpoints found")
        return None

    # Get the latest dataset
    latest_dataset = routing_dirs[0]
    print(f"Latest dataset: {latest_dataset.name}")

    # Find runs in this dataset
    runs = list(latest_dataset.iterdir())
    if not runs:
        print(f"ERROR: No runs in {latest_dataset}")
        return None

    latest_run = runs[0]
    print(f"Latest run: {latest_run.name}")

    # Prefer 'final', else highest numbered checkpoint
    final = latest_run / "final"
    if final.exists():
        return str(final).replace("/moe-data/", "")

    # Find highest numbered checkpoint
    checkpoint_dirs = [c for c in latest_run.iterdir() if c.is_dir() and c.name.startswith("checkpoint-")]
    if checkpoint_dirs:
        latest = sorted(checkpoint_dirs, key=lambda x: int(x.name.split("-")[1]), reverse=True)[0]
        return str(latest).replace("/moe-data/", "")

    print("ERROR: No valid checkpoint found")
    return None


@app.function(
    volumes={
        "/moe-data": source_volume,
        "/inference": inference_volume,
    },
    timeout=300,
)
def deploy_checkpoint(checkpoint_path: str) -> bool:
    """Deploy a checkpoint from training volume to inference volume."""
    src = Path(f"/moe-data/{checkpoint_path}")
    dst = Path("/inference/routing/adapter")

    if not src.exists():
        print(f"ERROR: Checkpoint not found: {src}")
        return False

    print(f"Source: {src} (moe-lora-data volume)")
    print(f"Target: {dst} (moe-inference volume)")

    # Create parent and remove old deployment
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print("Removing old deployment...")
        shutil.rmtree(dst)

    # Copy checkpoint
    print("Copying checkpoint...")
    shutil.copytree(src, dst)

    # Commit inference volume
    inference_volume.commit()

    print(f"\n✅ Deployed successfully!")
    print(f"   Files: {[f.name for f in dst.iterdir()]}")
    return True


@app.local_entrypoint()
def main(checkpoint: str = None, latest: bool = False):
    """Deploy routing LoRA to inference path.

    Args:
        checkpoint: Path to checkpoint (relative to volume root)
        latest: If True, automatically find and deploy the latest checkpoint
    """
    if latest:
        print("Finding latest checkpoint...")
        checkpoint = find_latest_checkpoint.remote()
        if not checkpoint:
            print("❌ Could not find latest checkpoint")
            exit(1)
        print(f"Found: {checkpoint}\n")

    if not checkpoint:
        print("ERROR: Must specify --checkpoint or --latest")
        print("\nUsage:")
        print("  modal run modal/deploy.py --latest")
        print("  modal run modal/deploy.py --checkpoint <path>")
        exit(1)

    print("=" * 60)
    print("Deploying Routing LoRA")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint}")
    print()

    success = deploy_checkpoint.remote(checkpoint)
    if not success:
        print("❌ Deployment failed!")
        exit(1)

    print("\n" + "=" * 60)
    print("✅ Routing LoRA deployed to inference/routing/adapter")
    print("=" * 60)
