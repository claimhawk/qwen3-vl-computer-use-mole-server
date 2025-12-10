# Copyright (c) 2025 Tylt LLC. All rights reserved.
# Licensed for research use only. Commercial use requires a license from Tylt LLC.
# Contact: hello@claimhawk.app | See LICENSE for terms.

# Version: 2025-12-02-v8 - Eval injects system prompt to match preprocessing
"""
Qwen3-VL LoRA Fine-tuning on Modal

Train Qwen3-VL-8B-Instruct on custom computer workflow tasks using LoRA.

Usage:
    modal run modal/training.py --run-name my_training_run
    modal run modal/training.py --run-name fast_test --fast --patience 2
"""

import gzip
import json
import os
import re
import tarfile
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import modal

# =============================================================================
# CENTRALIZED CONFIGURATION
# =============================================================================
# Volume names are read from config/adapters.yaml at module level.
# System prompt is built inside the Modal function from the bundled config.
#
# NOTE: We use a simple regex-based parser here instead of yaml because
# the yaml module may not be installed locally. The actual config is
# loaded properly with pyyaml inside the Modal container.


def _read_yaml_value(content: str, key_path: list[str]) -> str:
    """Extract a value from YAML content using simple parsing."""
    import re
    # This is a simple parser for the specific structure of adapters.yaml
    lines = content.split('\n')
    in_section = [False] * len(key_path)

    for line in lines:
        if not line.strip() or line.strip().startswith('#'):
            continue

        indent = len(line) - len(line.lstrip())
        level = indent // 2

        for i, key in enumerate(key_path[:-1]):
            if level == i and line.strip().startswith(f"{key}:"):
                in_section[i] = True
                for j in range(i + 1, len(in_section)):
                    in_section[j] = False
                break

        if all(in_section[:-1]):
            final_key = key_path[-1]
            if level == len(key_path) - 1 and line.strip().startswith(f"{final_key}:"):
                match = re.search(rf'{final_key}:\s*["\']?([^"\']+)["\']?', line)
                if match:
                    return match.group(1).strip()

    raise ValueError(f"Could not find {'.'.join(key_path)} in config")


def _load_local_config_values() -> dict[str, str]:
    """Load essential config values from local adapters.yaml."""
    # Path: modal/ -> mole-trainer-server/ -> projects/ -> claimhawk/ -> config/
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "adapters.yaml"
    content = config_path.read_text()
    return {
        "moe_volume": _read_yaml_value(content, ["volumes", "moe_data", "name"]),
        "base_model": _read_yaml_value(content, ["models", "base_vlm"]),
    }


_local_config = _load_local_config_values()

MOE_VOLUME_NAME = _local_config["moe_volume"]
BASE_MODEL = _local_config["base_model"]


def _build_system_prompt() -> str:
    """Build system prompt from config. Called inside Modal where yaml is available."""
    import yaml
    with open("/config/adapters.yaml") as f:
        config = yaml.safe_load(f)
    adapter_names = "\n".join(f"- {name}" for name in sorted(config["experts"].keys()))
    return f"""You are a Mixture of Experts router. You have been trained to look at an image and a text instruction and determine what adapter to route to.

Valid adapters names:
{adapter_names}

What adapter name should handle this image and instruction?"""

# Modal App Setup
app = modal.App("moe-lora-training")

# Modal Volumes - ONE volume with folders for each thing
moe_volume = modal.Volume.from_name(MOE_VOLUME_NAME, create_if_missing=True)
checkpoints_volume = moe_volume
logs_volume = moe_volume
data_volume = moe_volume

# Training history volume (shared with lora-trainer)
HISTORY_VOLUME_NAME = "claimhawk-training-history"
history_volume = modal.Volume.from_name(HISTORY_VOLUME_NAME, create_if_missing=True)

# Docker Image with Dependencies
# Use pre-compiled flash-attn wheel to avoid 10+ minute compilation
# IMPORTANT: torch version must match the flash-attn wheel (torch2.4)
# Use cxx11abiFALSE for better compatibility with CUDA containers
flash_attn_wheel = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
)

# Bundle config file into the Modal image
# Path: modal/ -> mole-trainer-server/ -> projects/ -> claimhawk/ -> config/
_config_path = Path(__file__).parent.parent.parent.parent / "config" / "adapters.yaml"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04",
        add_python="3.11"
    )
    .pip_install(
        "torch==2.4.0",  # Must match flash-attn wheel (torch2.4)
        "torchvision==0.19.0",  # Compatible with torch 2.4.0
        flash_attn_wheel,  # Pre-compiled wheel for fast container builds
    )
    .pip_install(
        "transformers>=4.57.0",  # Qwen3-VL requirement
        "accelerate>=0.27.0",
        "peft>=0.11.0",
        "datasets>=2.14.0",
        "qwen-vl-utils",
        "tensorboard>=2.14.0",
        "huggingface-hub>=0.20.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "pyyaml",
    )
    .add_local_file(str(_config_path), "/config/adapters.yaml", copy=True)
)


# Model cache volume for pre-downloaded HF models (avoids rate limiting)
model_cache = modal.Volume.from_name("claimhawk-model-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100:4",  # 4x H100 GPUs
    timeout=86400,  # 24 hours max
    volumes={
        "/moe-data": moe_volume,  # Single volume mount with subdirectories for each purpose
        "/models": model_cache,  # Pre-cached HF models
        "/history": history_volume,  # Training history (shared with lora-trainer)
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],  # For model downloads
)
def train_qwen3vl_lora(
    run_name: str = None,
    dataset_name: str = None,  # Required: Name of the dataset to train on
    # Data parameters
    use_preprocessed: bool = True,  # Use preprocessed data (faster startup)
    test_mode: bool = False,  # Use tiny subset for end-to-end testing
    # LoRA parameters (None = auto-tune based on dataset size)
    lora_rank: int = None,
    lora_alpha: int = None,
    lora_dropout: float = 0.05,
    enable_vision_lora: bool = True,
    vision_lora_rank: int = None,
    vision_lora_alpha: int = None,
    vision_lora_dropout: float = 0.05,
    # Training parameters (None = auto-tune)
    learning_rate: float = None,
    num_epochs: int = 1000,  # Large number - early stopping determines actual end
    batch_size: int = None,
    gradient_accumulation_steps: int = None,
    eval_steps: int = 20,
    save_steps: int = 20,
    # Early stopping (None = auto-tune)
    patience: int = None,
    # Fast mode
    fast: bool = False,
    # Model
    model_name: str = BASE_MODEL,
    # Auto-tuning
    auto_tune_hyperparams: bool = True,  # Automatically tune hyperparams based on dataset size
):
    """
    Fine-tune Qwen3-VL-8B-Instruct with LoRA on custom computer workflow tasks.

    Args:
        run_name: Name for this training run (default: auto-generated timestamp)
        dataset_name: Name of the dataset to train on (required)
        use_preprocessed: Use cached tensors instead of raw data
        test_mode: Slice dataset to tiny subset for end-to-end testing
        lora_rank: LoRA rank for LLM adapters (default: 16)
        lora_alpha: LoRA alpha for LLM adapters, typically 2x rank (default: 32)
        lora_dropout: LoRA dropout rate for LLM adapters (default: 0.05)
        enable_vision_lora: Whether to attach LoRA adapters to the vision tower/projector
        vision_lora_rank: LoRA rank for vision/projector adapters (default: 8)
        vision_lora_alpha: LoRA alpha for vision/projector adapters (default: 16)
        vision_lora_dropout: LoRA dropout for vision/projector adapters (default: 0.05)
        learning_rate: Learning rate (default: 3e-5 for production, use 1e-4 for fast mode)
        num_epochs: Number of training epochs (default: 3)
        batch_size: Batch size per device (default: 2)
        gradient_accumulation_steps: Gradient accumulation steps (default: 8, effective batch: 64 with 4 GPUs)
        eval_steps: Evaluate every N steps (default: 20)
        save_steps: Save checkpoint every N steps (default: 20)
        warmup_ratio: Warmup ratio (default: 0.1)
        patience: Early stopping patience (default: 8, use 2 for fast mode)
        action_weight: Loss weight for action names (default: 2.0)
        arg_weight: Loss weight for argument values (default: 5.0)
        fast: Enable fast mode (higher LR, lower patience) (default: False)
        model_name: Hugging Face model name (default: Qwen/Qwen3-VL-8B-Instruct)
    """
    import torch
    from transformers import (
        AutoProcessor,
        Qwen3VLForConditionalGeneration,
        TrainingArguments,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset
    from PIL import Image
    import numpy as np
    from tqdm import tqdm

    # Validate required parameters
    if dataset_name is None:
        raise ValueError("dataset_name is required. Please specify --dataset-name <name>")

    # Generate run name if not provided
    if run_name is None:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n{'='*80}")
    print(f"🚀 Starting Qwen3-VL LoRA Training: {run_name}")
    print(f"   Dataset: {dataset_name}")
    print(f"{'='*80}\n")

    # ============================================================================
    # STEP 1: Load Dataset (need this early to get dataset_name)
    # ============================================================================

    print(f"\n{'='*80}")
    print("📦 STEP 1: Loading Dataset")
    print(f"{'='*80}\n")

    if use_preprocessed:
        print("🚀 Using preprocessed data (fast path)")

        # Use explicit dataset name - extract base name to match preprocessing output
        # e.g., "datasets/routing_20251125_050207" -> "routing_20251125_050207"
        preprocessed_base = Path("/moe-data/preprocessed")
        base_dataset_name = Path(dataset_name).name
        latest_dataset_dir = preprocessed_base / base_dataset_name

        if not latest_dataset_dir.exists():
            raise FileNotFoundError(
                f"Preprocessed dataset not found: {latest_dataset_dir}\n"
                f"Please run preprocessing first:\n"
                f"  uvx modal run -d modal/preprocess.py --dataset-name {dataset_name}"
            )

        # Check for metadata.json
        metadata_path = latest_dataset_dir / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found: {metadata_path}\n"
                f"Please run preprocessing first:\n"
                f"  uvx modal run -d modal/preprocess.py --dataset-name {dataset_name}"
            )

        with open(metadata_path) as f:
            metadata = json.load(f)

        train_samples = metadata["train_samples"]
        val_samples = metadata["val_samples"]

        print(f"\n📊 Preprocessed dataset size:")
        print(f"   Dataset: {dataset_name}")
        print(f"   Train samples: {train_samples}")
        print(f"   Val samples: {val_samples}")
        print(f"   Model: {metadata['model_name']}")

        # Setup paths with nested structure: /moe-data/checkpoints/{dataset_name}/{run_name}/
        output_dir = f"/moe-data/checkpoints/{dataset_name}/{run_name}"
        log_dir = f"/moe-data/tb_logs/{dataset_name}/{run_name}"

        # Create directories
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        print(f"\n📁 Paths:")
        print(f"   Checkpoints: {output_dir}")
        print(f"   Logs: {log_dir}")

        # Store paths for later use
        train_data = None  # Will load .pt files directly in STEP 4
        val_data = None

    else:
        print("📦 Using raw data (will preprocess on GPU)")

        # Check for chunked dataset
        chunks_dir = Path(f"/moe-data/data/{dataset_name}/chunks")
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
            dataset_dir = Path(f"/moe-data/data/{dataset_name}")
            reassemble_path = dataset_dir / "dataset.tar.gz"
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
                tar.extractall(path=dataset_dir)
            print("   ✅ Extraction complete")

            # Cleanup archive
            reassemble_path.unlink()
            print("   🧹 Cleaned up archive")

        # Find train and val files
        # Try both /moe-data/data/{dataset_name} and /moe-data/data/datasets/{dataset_name}
        dataset_dir = Path(f"/moe-data/data/{dataset_name}")
        if not dataset_dir.exists():
            dataset_dir = Path(f"/moe-data/data/datasets/{dataset_name}")

        train_files = list(dataset_dir.glob("train*.jsonl"))
        # Accept both val*.jsonl and eval*.jsonl
        val_files = list(dataset_dir.glob("val*.jsonl")) + list(dataset_dir.glob("eval*.jsonl"))

        if not train_files or not val_files:
            raise FileNotFoundError(f"Could not find train*.jsonl or val*.jsonl/eval*.jsonl in {dataset_dir}")

        train_path = train_files[0]
        val_path = val_files[0]

        print(f"\n✅ Dataset files:")
        print(f"   Train: {train_path.name}")
        print(f"   Val: {val_path.name}")

        # Load JSONL data
        def load_jsonl(path):
            data = []
            with open(path, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
            return data

        train_data = load_jsonl(train_path)
        val_data = load_jsonl(val_path)

        if test_mode:
            original_train = len(train_data)
            original_val = len(val_data)
            train_data = train_data[: min(64, original_train)]
            val_data = val_data[: min(16, original_val)]
            print(f"\n🧪 Test mode enabled – raw dataset sliced to {len(train_data)}/{original_train} train and {len(val_data)}/{original_val} val samples")

        print(f"\n📊 Dataset size:")
        print(f"   Train samples: {len(train_data)}")
        print(f"   Val samples: {len(val_data)}")

        # Setup paths with nested structure: /moe-data/checkpoints/{dataset_name}/{run_name}/
        output_dir = f"/moe-data/checkpoints/{dataset_name}/{run_name}"
        log_dir = f"/moe-data/tb_logs/{dataset_name}/{run_name}"

        # Create directories
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        print(f"\n📁 Paths:")
        print(f"   Checkpoints: {output_dir}")
        print(f"   Logs: {log_dir}")

    # Clear ALL old tensorboard logs (including nested run directories)
    import shutil
    tb_logs_path = Path("/moe-data/tb_logs")
    if tb_logs_path.exists():
        print("\n🧹 Clearing ALL old tensorboard logs...")
        # Delete all subdirectories in tb_logs (run_* folders)
        deleted_count = 0
        for item in tb_logs_path.iterdir():
            if item.is_dir() and not item.is_symlink():
                print(f"   Deleting {item.name}...")
                shutil.rmtree(item)
                deleted_count += 1
        print(f"✓ Deleted {deleted_count} old tensorboard run(s)")

        # Commit volume so TensorBoard web server can see the changes
        print("💾 Committing logs volume for TensorBoard...")
        logs_volume.commit()
        print("✓ Logs volume committed - TensorBoard will refresh within 30s\n")

    # ============================================================================
    # STEP 1.5: Auto-tune Hyperparameters Based on Dataset Size
    # ============================================================================

    # Determine train size based on whether we're using preprocessed or raw data
    if use_preprocessed:
        # Read metadata to get train size (use base_dataset_name to match preprocessing output)
        metadata_path = Path(f"/moe-data/preprocessed/{base_dataset_name}/metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        train_size = metadata['train_samples']
    else:
        # Use loaded train_data
        train_size = len(train_data)

    if auto_tune_hyperparams:
        print(f"\n{'='*80}")
        print("🎯 STEP 1.5: Auto-tuning Hyperparameters")
        print(f"{'='*80}\n")

        print(f"📊 Dataset size: {train_size:,} training samples\n")

        # Auto-tune LoRA ranks based on dataset size
        # Target: 100-200 samples per 1M trainable params (conservative to avoid overfitting)
        # Rough estimate: rank R creates ~2R million params for LLM, ~R million for vision
        # So total params ≈ 3R million, target is train_size / (100-200) million params
        if lora_rank is None:
            # Calculate optimal rank: train_size / target_samples_per_M_params / 3 (for 3R total)
            target_samples_per_m = 150  # Conservative target
            optimal_rank = train_size / (target_samples_per_m * 3)

            # Round to nearest power of 2 (common LoRA ranks: 2, 4, 8, 16, 32)
            import math
            lora_rank = max(2, min(32, 2 ** round(math.log2(optimal_rank))))

            print(f"🔧 Auto-tuned lora_rank: {lora_rank}")
            print(f"   (Optimal: {optimal_rank:.1f}, rounded to power of 2)")

        if lora_alpha is None:
            lora_alpha = lora_rank * 2  # Standard 2x multiplier
            print(f"🔧 Auto-tuned lora_alpha: {lora_alpha}")

        if vision_lora_rank is None:
            # Vision tower needs less capacity - use half of LLM rank
            vision_lora_rank = max(1, lora_rank // 2)
            print(f"🔧 Auto-tuned vision_lora_rank: {vision_lora_rank}")

        if vision_lora_alpha is None:
            vision_lora_alpha = vision_lora_rank * 2
            print(f"🔧 Auto-tuned vision_lora_alpha: {vision_lora_alpha}")

        # Calculate actual trainable params
        estimated_llm_params = lora_rank * 2  # Rough estimate in millions
        estimated_vision_params = vision_lora_rank  # Rough estimate in millions
        total_params = estimated_llm_params + estimated_vision_params
        samples_per_m = train_size / total_params

        print(f"\n📈 Parameter efficiency:")
        print(f"   Estimated trainable params: ~{total_params:.0f}M ({estimated_llm_params}M LLM + {estimated_vision_params}M vision)")
        print(f"   Samples per 1M params: ~{samples_per_m:.0f}")

        # Auto-tune batch size based on dataset size
        if batch_size is None:
            # Smaller datasets use smaller batches to see more gradient updates
            # Larger datasets can use larger batches for efficiency
            if train_size < 5000:
                batch_size = 1
            elif train_size < 50000:
                batch_size = 2
            else:
                batch_size = 4
            print(f"\n🔧 Auto-tuned batch_size: {batch_size}")

        if gradient_accumulation_steps is None:
            # Target effective batch size of 16-32
            target_effective_batch = 16
            gradient_accumulation_steps = max(1, target_effective_batch // batch_size)
            print(f"🔧 Auto-tuned gradient_accumulation_steps: {gradient_accumulation_steps}")
            print(f"   Effective batch size: {batch_size * gradient_accumulation_steps}")

        # Auto-tune learning rate based on rank
        if learning_rate is None:
            # Smaller ranks = fewer params = can handle higher LR
            # Base LR scales inversely with sqrt(rank) for stability
            import math
            base_lr = 1e-4
            lr_scale = math.sqrt(4 / lora_rank)  # Normalized to rank 4
            learning_rate = base_lr * lr_scale
            # Clamp to reasonable range
            learning_rate = max(2e-5, min(1e-4, learning_rate))
            print(f"\n🔧 Auto-tuned learning_rate: {learning_rate:.2e}")
            print(f"   (Base LR {base_lr:.2e} × scale {lr_scale:.2f})")

        # Set patience to 3 by default (unless explicitly overridden)
        if patience is None:
            patience = 3
            print(f"\n🔧 Patience: {patience} (default)")

        print(f"\n✓ Hyperparameters auto-tuned for {train_size:,} samples\n")

    else:
        # Use provided defaults if auto-tune is disabled
        if lora_rank is None: lora_rank = 16
        if lora_alpha is None: lora_alpha = 32
        if vision_lora_rank is None: vision_lora_rank = 8
        if vision_lora_alpha is None: vision_lora_alpha = 16
        if batch_size is None: batch_size = 2
        if gradient_accumulation_steps is None: gradient_accumulation_steps = 8
        if learning_rate is None: learning_rate = 3e-5
        if patience is None: patience = 8

    # Fast mode: 40 steps, then exit (quick training run for validation)
    max_steps_override = None
    if fast:
        print(f"\n{'='*80}")
        print("⚡ FAST MODE")
        print(f"{'='*80}\n")
        max_steps_override = 40
        eval_steps = 40  # Evaluate at end
        save_steps = 40  # Save checkpoint at end
        print(f"🔧 max_steps: {max_steps_override}")
        print(f"🔧 eval_steps: {eval_steps}")
        print(f"🔧 save_steps: {save_steps}")
        print(f"🔧 Early stopping: DISABLED (will exit after {max_steps_override} steps)")
        print(f"\nPipeline: load dataset → load model → train {max_steps_override} steps → validate → checkpoint → stats → exit\n")

    # Loss weighting for tool_call tokens (used for computer use training)
    # For routing training (simple classification), use equal weights
    base_weight = 1.0
    action_weight = 1.0
    arg_weight = 1.0

    # ALWAYS print final hyperparameter values (whether auto-tuned or provided)
    print(f"\n{'='*80}")
    print("📋 Final Hyperparameters")
    print(f"{'='*80}\n")
    print(f"LoRA Configuration:")
    print(f"   lora_rank: {lora_rank}")
    print(f"   lora_alpha: {lora_alpha}")
    print(f"   lora_dropout: {lora_dropout}")
    print(f"   vision_lora_rank: {vision_lora_rank}")
    print(f"   vision_lora_alpha: {vision_lora_alpha}")
    print(f"   vision_lora_dropout: {vision_lora_dropout}")
    print(f"\nTraining Configuration:")
    print(f"   batch_size: {batch_size}")
    print(f"   gradient_accumulation_steps: {gradient_accumulation_steps}")
    print(f"   effective_batch_size: {batch_size * gradient_accumulation_steps}")
    print(f"   learning_rate: {learning_rate}")
    print(f"   num_epochs: {num_epochs}")
    print(f"   patience: {patience}")
    print(f"\nEarly Stopping:")
    print(f"   Metric: eval_loss")
    print(f"   Min delta: 0.0005")
    print()

    # ============================================================================
    # STEP 2: Load Model and Processor
    # ============================================================================

    print(f"\n{'='*80}")
    print("🤖 STEP 2: Loading Model")
    print(f"{'='*80}\n")

    # Check for cached model (avoids HF rate limiting)
    cached_model_path = f"/models/{model_name.replace('/', '--')}"
    if os.path.exists(cached_model_path) and os.listdir(cached_model_path):
        print(f"✅ Using cached model: {cached_model_path}")
        model_path = cached_model_path
    else:
        print(f"📥 Downloading model from HF: {model_name}")
        model_path = model_name

    print(f"Loading model: {model_path}")

    # Load processor
    processor = AutoProcessor.from_pretrained(model_path)

    # Load model
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",  # Use flash attention
    )

    print(f"✅ Model loaded")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    print(f"   Dtype: {model.dtype}")

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    print("✓ Gradient checkpointing enabled")

    # ============================================================================
    # STEP 3: Apply LoRA
    # ============================================================================

    print(f"\n{'='*80}")
    print("🔧 STEP 3: Applying LoRA")
    print(f"{'='*80}\n")

    vision_discovery_pattern = re.compile(r"(vision|visual|vit|vision_tower|mm_projector|vision_proj)", re.IGNORECASE)
    vision_scope_pattern = re.compile(r"(?:^|[._])(vision|visual|vit)", re.IGNORECASE)
    projector_scope_pattern = re.compile(r"(?:^|[._])(mm_projector|projector|vision_proj)", re.IGNORECASE)

    VISION_KEYWORDS = ["vision", "visual", "vit"]
    PROJECTOR_KEYWORDS = ["mm_projector", "projector", "vision_proj"]

    LLM_TARGET_SUFFIXES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    VISION_TARGET_SUFFIXES = ["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2", "proj", "linear"]
    PROJECTOR_TARGET_SUFFIXES = ["proj", "linear"]
    TARGET_SUFFIXES = sorted(set(LLM_TARGET_SUFFIXES + VISION_TARGET_SUFFIXES + PROJECTOR_TARGET_SUFFIXES))

    def _unique(values):
        seen = set()
        ordered = []
        for item in values:
            if item in seen:
                continue
            ordered.append(item)
            seen.add(item)
        return ordered

    def _preview(name_list, max_items=20):
        if not name_list:
            return ""
        subset = name_list[:max_items]
        suffix = "" if len(name_list) <= max_items else " ..."
        return ", ".join(subset) + suffix

    module_names = [name for name, _ in model.named_modules() if name]

    discovery_hits = [name for name in module_names if vision_discovery_pattern.search(name)]
    projector_hits = [name for name in module_names if projector_scope_pattern.search(name.lower())]
    vision_hits = [
        name for name in module_names
        if vision_scope_pattern.search(name.lower()) and name not in projector_hits
    ]

    print(f"🔍 Vision/Projector module discovery: {len(discovery_hits)} hit(s)")
    if discovery_hits:
        print(f"   Preview: {_preview(discovery_hits)}")
    else:
        print("   ⚠️  No modules matched discovery regex; will fall back to suffix targeting.")

    print(f"   Vision hits: {len(vision_hits)}")
    print(f"   Projector hits: {len(projector_hits)}")
    if not projector_hits:
        print("   ⚠️  No modules matched projector-specific prefixes (mm_projector|projector|vision_proj).")
    has_projector_modules = bool(projector_hits)

    def _categorize_targets(names):
        llm_targets, vision_targets, projector_targets = [], [], []
        for name in names:
            suffix = name.split(".")[-1]
            if not suffix:
                continue
            lower_name = name.lower()
            in_projector = bool(projector_scope_pattern.search(lower_name))
            in_vision = bool(vision_scope_pattern.search(lower_name)) and not in_projector

            if in_projector and suffix in PROJECTOR_TARGET_SUFFIXES:
                projector_targets.append(name)
                continue

            if (in_projector or in_vision) and suffix in VISION_TARGET_SUFFIXES:
                vision_targets.append(name)
                continue

            if not in_projector and not in_vision and suffix in LLM_TARGET_SUFFIXES:
                llm_targets.append(name)

        return _unique(llm_targets), _unique(vision_targets), _unique(projector_targets)

    llm_module_targets, vision_module_targets, projector_module_targets = _categorize_targets(module_names)

    print("\nTarget module summary:")
    print(f"   LLM targets: {len(llm_module_targets)}")
    print(f"   Vision targets: {len(vision_module_targets)}")
    print(f"   Projector targets: {len(projector_module_targets)}")
    print(f"   Canonical target suffixes: {TARGET_SUFFIXES}")

    # Freeze base weights before attaching adapters
    for param in model.parameters():
        param.requires_grad = False
    print("✓ Base model parameters frozen")

    def _normalize_targets(resolved_targets, fallback_suffixes, scope_name):
        if resolved_targets:
            print(f"   Using {len(resolved_targets)} explicit {scope_name} target modules")
            return resolved_targets
        print(f"   ⚠️  No explicit {scope_name} modules discovered; falling back to suffixes: {fallback_suffixes}")
        return fallback_suffixes

    normalized_llm_targets = _normalize_targets(llm_module_targets, LLM_TARGET_SUFFIXES, "LLM")

    combined_vision_targets = _unique(vision_module_targets + projector_module_targets)
    normalized_vision_targets = []
    if enable_vision_lora:
        normalized_vision_targets = _normalize_targets(
            combined_vision_targets,
            sorted(set(VISION_TARGET_SUFFIXES + PROJECTOR_TARGET_SUFFIXES)),
            "vision/projector",
        )
    else:
        print("\n⚠️  Vision LoRA disabled via flag; only LLM adapters will be trained.")

    all_target_modules = _unique(normalized_llm_targets + normalized_vision_targets)
    if not all_target_modules:
        all_target_modules = normalized_llm_targets or LLM_TARGET_SUFFIXES

    rank_pattern = {}
    alpha_pattern = {}
    if enable_vision_lora and normalized_vision_targets:
        for module_name in normalized_vision_targets:
            rank_pattern[module_name] = vision_lora_rank
            alpha_pattern[module_name] = vision_lora_alpha

    print("\nApplying unified LoRA adapters...")
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=all_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        rank_pattern=rank_pattern or None,
        alpha_pattern=alpha_pattern or None,
    )
    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    def _has_token(names, tokens):
        lowered = [n.lower() for n in names]
        return any(tok in name for tok in tokens for name in lowered)

    trainable_lora_names = [name for name, p in model.named_parameters() if p.requires_grad and "lora" in name]
    if not trainable_lora_names:
        raise RuntimeError("No trainable LoRA parameters detected after adapter injection.")

    non_lora_trainables = [name for name, p in model.named_parameters() if p.requires_grad and "lora" not in name]
    if non_lora_trainables:
        sample = ", ".join(non_lora_trainables[:8])
        raise RuntimeError(f"Base weights remain trainable (expected frozen). Sample: {sample}")

    if enable_vision_lora:
        if not _has_token(trainable_lora_names, VISION_KEYWORDS):
            raise RuntimeError("Missing vision LoRA parameters (no trainable names contain vision tokens).")
        if has_projector_modules:
            if not _has_token(trainable_lora_names, PROJECTOR_KEYWORDS):
                raise RuntimeError("Missing projector LoRA parameters (no trainable names contain projector tokens).")
            print("✓ Vision and projector LoRA parameters detected.")
        else:
            print("⚠️  No projector-prefixed modules exist on this model; verified vision adapters only.")
    else:
        print("✓ Skipping vision/projector LoRA verification (disabled).")

    print(f"\n✅ LoRA applied")
    print(f"   Trainable parameters: {trainable_params / 1e6:.2f}M ({100 * trainable_params / total_params:.2f}%)")
    print(f"   Total parameters: {total_params / 1e9:.2f}B")

    # ============================================================================
    # STEP 4: Prepare Dataset
    # ============================================================================

    print(f"\n{'='*80}")
    print("📚 STEP 4: Preparing Dataset")
    print(f"{'='*80}\n")

    # Constants from official Qwen3-VL implementation
    IGNORE_INDEX = -100

    if use_preprocessed:
        print("🚀 Using preprocessed tensors (lazy loading)...")

        # Use the latest_dataset_dir that was already found in STEP 1
        preproc_base = latest_dataset_dir
        print(f"   Using preprocessed data from: {preproc_base}")

        # Find all preprocessed files
        train_pt_files = sorted((preproc_base / "train").glob("*.pt"))
        val_pt_files = sorted((preproc_base / "val").glob("*.pt"))

        if test_mode:
            original_train = len(train_pt_files)
            original_val = len(val_pt_files)
            train_pt_files = train_pt_files[: min(64, original_train)]
            val_pt_files = val_pt_files[: min(16, original_val)]
            print(f"\n🧪 Test mode enabled – preprocessed dataset sliced to {len(train_pt_files)}/{original_train} train and {len(val_pt_files)}/{original_val} val files")

        print(f"Found {len(train_pt_files)} training files")
        print(f"Found {len(val_pt_files)} validation files")

        # Create lazy-loading dataset (doesn't load into memory until needed)
        class PreprocessedDataset(torch.utils.data.Dataset):
            """Lazy-loading dataset for preprocessed .pt files"""
            def __init__(self, pt_files):
                self.pt_files = pt_files

            def __len__(self):
                return len(self.pt_files)

            def __getitem__(self, idx):
                # Load sample on-demand (not upfront)
                path = self.pt_files[idx]
                try:
                    return torch.load(path)
                except Exception as exc:
                    raise RuntimeError(f"Failed to load preprocessed sample: {path}") from exc

        train_dataset = PreprocessedDataset(train_pt_files)
        val_dataset = PreprocessedDataset(val_pt_files)

        print(f"✅ Lazy-loading dataset created (samples loaded on-demand during training)")

    else:
        print("📦 Preprocessing raw data on GPU...")

        def prepare_sample(sample):
            """
            Process each sample individually following official Qwen3-VL pattern.
            Each instance gets its own pixel_values and image_grid_thw.
            """
            from PIL import Image

            # Get image path - the path already includes the full relative path
            image_path = Path(f"/moe-data/data/{dataset_name}") / sample["image"]
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            # Get conversations in old format:
            # [{"from": "human", "value": "<image>\nquery"}, {"from": "gpt", "value": "response"}]
            old_conversations = sample["conversations"]

            # Convert to Qwen-VL's expected format:
            # [{"role": "user", "content": [{"type": "image", "image": "file://..."}, {"type": "text", "text": "query"}]}, ...]
            messages = []
            images = []
            for msg in old_conversations:
                role = "user" if msg["from"] == "human" else "assistant"
                content_list = []

                # Parse the value - check for <image> token
                value = msg["value"]
                if "<image>" in value:
                    # Add image content first
                    content_list.append({
                        "type": "image",
                        "image": f"file://{image_path}"
                    })
                    # Collect actual image for processing
                    images.append(Image.open(image_path).convert("RGB"))
                    # Remove <image> token and add remaining text
                    text = value.replace("<image>", "").strip()
                    if text:
                        content_list.append({"type": "text", "text": text})
                else:
                    # Just text content
                    content_list.append({"type": "text", "text": value})

                messages.append({"role": role, "content": content_list})

            # Process this single example with the processor
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

            # Process with processor to get vision features
            model_inputs = processor(
                text=[text],
                images=[images] if images else None,
                return_tensors="pt",
                padding=False  # Don't pad individual samples
            )

            # Return processed sample with squeezed tensors (remove batch dim)
            # Ensure all values are tensors, not lists
            result = {
                "input_ids": torch.tensor(model_inputs["input_ids"][0]) if isinstance(model_inputs["input_ids"][0], list) else model_inputs["input_ids"][0],
                "attention_mask": torch.tensor(model_inputs["attention_mask"][0]) if isinstance(model_inputs["attention_mask"][0], list) else model_inputs["attention_mask"][0],
            }

            if "pixel_values" in model_inputs:
                result["pixel_values"] = model_inputs["pixel_values"]  # Keep as is for concatenation
                result["image_grid_thw"] = model_inputs["image_grid_thw"]  # Keep as is

            return result

        # Create datasets
        print("Processing training data...")
        train_dataset = Dataset.from_list([prepare_sample(s) for s in tqdm(train_data, desc="Train")])

        print("Processing validation data...")
        val_dataset = Dataset.from_list([prepare_sample(s) for s in tqdm(val_data, desc="Val")])

    print(f"\n✅ Dataset prepared")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val: {len(val_dataset)} samples")

    # ============================================================================
    # STEP 5: Training Setup
    # ============================================================================

    print(f"\n{'='*80}")
    print("⚙️  STEP 5: Training Setup")
    print(f"{'='*80}\n")

    # Calculate total steps
    num_gpus = torch.cuda.device_count()
    effective_batch_size = batch_size * gradient_accumulation_steps * num_gpus
    steps_per_epoch = len(train_dataset) // effective_batch_size
    total_steps = steps_per_epoch * num_epochs

    # Warmup based on realistic training length with early stopping (300-500 steps)
    # Use 5% of realistic length, not theoretical total_steps
    warmup_steps = 10  # Fixed short warmup for LoRA fine-tuning

    print(f"Training Configuration:")
    print(f"   GPUs: {num_gpus}")
    print(f"   Batch size per device: {batch_size}")
    print(f"   Gradient accumulation: {gradient_accumulation_steps}")
    print(f"   Effective batch size: {effective_batch_size}")
    print(f"   Steps per epoch: {steps_per_epoch}")
    print(f"   Total epochs: {num_epochs} (early stopping will terminate)")
    print(f"   Total steps: {total_steps}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Warmup steps: {warmup_steps} (scaled: max(10, 5% of total))")
    print(f"   Early stopping patience: {patience} evaluations ({patience * eval_steps} steps)")

    # Training arguments (matches UI-TARS config)
    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps_override if max_steps_override else -1,
        num_train_epochs=num_epochs if not max_steps_override else -1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="constant_with_warmup",  # Constant LR after warmup for steep learning
        warmup_steps=warmup_steps,  # Short warmup then full speed
        weight_decay=0.01,  # Like UI-TARS
        logging_dir=log_dir,
        logging_steps=5,  # More frequent logging (like UI-TARS)
        eval_strategy="steps",  # Changed from evaluation_strategy (deprecated)
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=None,  # Keep ALL checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # Use standard eval loss
        greater_is_better=False,
        bf16=True,  # Use bfloat16 for H100
        gradient_checkpointing=True,  # Enable gradient checkpointing
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=0.3,  # Gradient clipping
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        remove_unused_columns=False,  # Important for vision models
        report_to="tensorboard",  # TensorBoard logging
        seed=420,  # Fixed seed for reproducibility
        data_seed=420,  # Fixed seed for data shuffling
    )

    # Simple early stopping callback
    from transformers import TrainerCallback

    class EarlyStoppingCallback(TrainerCallback):
        """
        Simple early stopping: stop training after `patience` evaluations with no improvement.
        """
        def __init__(self, patience, min_delta):
            self.patience = patience
            self.min_delta = min_delta
            self.best_metric = None
            self.prev_metric = None
            self.patience_counter = 0

        def on_evaluate(self, args, state, control, metrics, **kwargs):
            current_loss = metrics.get('eval_loss')
            if current_loss is None:
                return control

            # Check if we have improvement
            improved = False
            if self.best_metric is None:
                self.best_metric = current_loss
                improved = True
            elif current_loss < self.best_metric - self.min_delta:
                # Significant improvement
                self.best_metric = current_loss
                self.patience_counter = 0
                improved = True
            else:
                # No significant improvement
                self.patience_counter += 1

            # Display status with delta
            if self.prev_metric is not None:
                delta = self.prev_metric - current_loss
                delta_str = f" (Δ={delta:+.6f})" if delta != 0 else ""
            else:
                delta_str = ""

            if improved:
                print(f"📊 Eval loss improved: {current_loss:.6f} (best: {self.best_metric:.6f}){delta_str}")
                print(f"   Patience: {self.patience_counter}/{self.patience}")
            else:
                print(f"📊 No improvement: {current_loss:.6f} vs best {self.best_metric:.6f}{delta_str}")
                print(f"   Patience: {self.patience_counter}/{self.patience}")

            self.prev_metric = current_loss

            # Check if patience exceeded - STOP immediately
            if self.patience_counter >= self.patience:
                print(f"\n⏹️  Early stopping triggered: {self.patience} evaluations with no improvement")
                control.should_training_stop = True

            return control

    early_stopping = EarlyStoppingCallback(
        patience=patience,
        min_delta=0.0005,
    )

    print(f"\n📉 Early Stopping:")
    print(f"   Metric: eval_loss")
    print(f"   Min delta: 0.0005")
    print(f"   Patience: {patience} evals (then stop)\n")

    # Cost tracking callback
    class CostTrackingCallback(TrainerCallback):
        """Tracks training costs based on Modal GPU pricing and generates cost reports"""
        def __init__(self, output_dir, dataset_name, run_name, num_gpus=8):
            self.output_dir = Path(output_dir)
            self.dataset_name = dataset_name
            self.run_name = run_name
            self.num_gpus = num_gpus
            # Modal H100 pricing: $4.50/hour per GPU
            self.gpu_hourly_rate = 4.50
            self.start_time = None
            self.checkpoint_costs = []
            self.total_time_seconds = 0
            # Path to save train-results.json in the original dataset folder
            self.dataset_results_path = Path(f"/moe-data/data/{dataset_name}/train-results.json")

        def on_train_begin(self, args, state, control, **kwargs):
            """Record training start time"""
            self.start_time = time.time()
            print(f"\n💰 Cost Tracking Initialized")
            print(f"   GPU Config: {self.num_gpus}x H100")
            print(f"   Rate: ${self.gpu_hourly_rate}/hour per GPU")
            print(f"   Total Rate: ${self.gpu_hourly_rate * self.num_gpus}/hour\n")
            return control

        def on_save(self, args, state, control, **kwargs):
            """Calculate cost for this checkpoint"""
            if self.start_time is None:
                return control

            current_time = time.time()
            elapsed_seconds = current_time - self.start_time
            elapsed_hours = elapsed_seconds / 3600

            # Calculate cumulative cost from training start
            cumulative_cost = elapsed_hours * self.gpu_hourly_rate * self.num_gpus
            self.total_time_seconds = elapsed_seconds

            self.checkpoint_costs.append({
                'step': state.global_step,
                'elapsed_seconds': elapsed_seconds,
                'elapsed_hours': elapsed_hours,
                'cumulative_cost_usd': cumulative_cost
            })

            print(f"\n💰 Checkpoint {state.global_step} Cost:")
            print(f"   Time: {elapsed_hours:.2f} hours ({elapsed_seconds/60:.1f} minutes)")
            print(f"   Cumulative Cost: ${self.checkpoint_costs[-1]['cumulative_cost_usd']:.4f}\n")

            return control

        def on_train_end(self, args, state, control, **kwargs):
            """Generate final cost report"""
            if self.start_time is None:
                return control

            final_time = time.time()
            total_seconds = final_time - self.start_time
            total_hours = total_seconds / 3600
            total_cost = total_hours * self.gpu_hourly_rate * self.num_gpus

            # Generate detailed cost report
            report = {
                'run_metadata': {
                    'dataset': self.dataset_name,
                    'run_name': self.run_name,
                    'completion_time': datetime.now().isoformat(),
                    'total_steps': state.global_step,
                },
                'hardware': {
                    'gpu_count': self.num_gpus,
                    'gpu_type': 'H100',
                    'hourly_rate_per_gpu': self.gpu_hourly_rate,
                    'total_hourly_rate': self.gpu_hourly_rate * self.num_gpus,
                },
                'time': {
                    'total_seconds': total_seconds,
                    'total_minutes': total_seconds / 60,
                    'total_hours': total_hours,
                },
                'cost': {
                    'total_usd': total_cost,
                    'cost_per_step': total_cost / state.global_step if state.global_step > 0 else 0,
                },
                'checkpoints': self.checkpoint_costs,
            }

            # Save cost report as JSON to checkpoint dir
            report_path = self.output_dir / "cost_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)

            # Also save train-results.json to the original dataset folder
            # This makes cost/training info available alongside the dataset
            try:
                with open(self.dataset_results_path, 'w') as f:
                    json.dump(report, f, indent=2)
                print(f"📄 Train results saved to dataset: {self.dataset_results_path}")
            except Exception as e:
                print(f"⚠️  Warning: Could not save train-results.json to dataset: {e}")

            # Print final cost report
            print(f"\n{'='*80}")
            print("💰 TRAINING COST REPORT")
            print(f"{'='*80}\n")
            print(f"Hardware: {self.num_gpus}x H100 @ ${self.gpu_hourly_rate}/hour/GPU")
            print(f"Total Time: {total_hours:.2f} hours ({total_seconds/60:.1f} minutes)")
            print(f"Total Cost: ${total_cost:.4f}")
            print(f"Cost per Step: ${total_cost/state.global_step:.6f}")
            print(f"\nCheckpoint Breakdown:")
            for ckpt in self.checkpoint_costs:
                print(f"  Step {ckpt['step']:4d}: ${ckpt['cumulative_cost_usd']:.4f} "
                      f"({ckpt['elapsed_hours']:.2f}h)")
            print(f"\n📄 Detailed report saved to: {report_path}")
            print(f"{'='*80}\n")

            return control

    # Calculate num GPUs for cost tracking
    num_gpus = torch.cuda.device_count()
    cost_tracker = CostTrackingCallback(output_dir, dataset_name, run_name, num_gpus)

    # Volume commit callback: Commit checkpoints to persistent storage after each save
    class VolumeCommitCallback(TrainerCallback):
        """Commits Modal volumes after each checkpoint save"""
        def __init__(self):
            # Reference the same Modal volumes by name (uses centralized config)
            self.checkpoint_volume = modal.Volume.from_name(MOE_VOLUME_NAME)
            self.logs_volume = modal.Volume.from_name(MOE_VOLUME_NAME)
            self.data_volume = modal.Volume.from_name(MOE_VOLUME_NAME)

        def on_save(self, args, state, control, **kwargs):
            # Commit volumes after each checkpoint save
            print(f"\n💾 Committing checkpoint {state.global_step} to Modal volume...")
            self.checkpoint_volume.commit()
            self.logs_volume.commit()
            print(f"✓ Checkpoint {state.global_step} persisted to volume\n")
            return control

        def on_train_end(self, args, state, control, **kwargs):
            # Commit data volume to persist train-results.json
            print(f"\n💾 Committing train-results.json to data volume...")
            self.data_volume.commit()
            print(f"✓ Train results persisted to data volume\n")
            return control

    volume_commit_callback = VolumeCommitCallback()

    class EvalExampleLogger(TrainerCallback):
        """Logs decoded eval examples to TensorBoard during evaluation."""
        def __init__(self, tokenizer, eval_dataset, log_dir, max_examples=3, max_chars=2000):
            from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
            self.tokenizer = tokenizer
            self.eval_dataset = eval_dataset
            self.max_examples = max_examples
            self.max_chars = max_chars
            self.writer = _SummaryWriter(os.path.join(log_dir, "eval_examples"))

        @staticmethod
        def _tensor_to_list(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().tolist()
            if isinstance(value, list):
                return value
            if isinstance(value, tuple):
                return list(value)
            if isinstance(value, int):
                return [int(value)]
            try:
                return list(value)
            except TypeError:
                return [value]

        def _decode_prompt(self, token_ids):
            text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
            if len(text) > self.max_chars:
                return text[: self.max_chars] + " …"
            return text

        def on_evaluate(self, args, state, control, metrics, model=None, **kwargs):
            if not self.eval_dataset:
                return control

            num_examples = min(self.max_examples, len(self.eval_dataset))
            for idx in range(num_examples):
                sample = self.eval_dataset[idx]
                input_ids = self._tensor_to_list(sample["input_ids"])
                prompt_text = self._decode_prompt(input_ids)

                labels = sample.get("labels")
                if labels is not None:
                    label_values = self._tensor_to_list(labels)
                    label_ids = [token_id for token_id in label_values if token_id != IGNORE_INDEX]
                    target_text = self._decode_prompt(label_ids) if label_ids else "(empty)"
                else:
                    target_text = "(missing labels)"

                log_text = f"**Prompt**\n{prompt_text}\n\n**Target**\n{target_text}"
                tag = f"eval_examples/example_{idx}"
                self.writer.add_text(tag, log_text, state.global_step)

            return control

        def on_train_end(self, args, state, control, **kwargs):
            self.writer.flush()
            self.writer.close()
            return control

    # ============================================================================
    # STEP 6: Train
    # ============================================================================

    print(f"\n{'='*80}")
    print("🏋️  STEP 6: Training")
    print(f"{'='*80}\n")

    # Collator following official Qwen3-VL pattern
    @dataclass
    class DataCollatorForSupervisedDataset:
        """Collate examples for supervised fine-tuning."""
        processor: Any

        def __call__(self, instances):
            """
            Batch collation following official Qwen3-VL pattern.
            Each instance is already processed - just concatenate tensors.
            """
            # Convert to tensors if they're lists (HuggingFace Dataset may serialize tensors)
            input_ids = [
                torch.tensor(instance["input_ids"]) if isinstance(instance["input_ids"], list) else instance["input_ids"]
                for instance in instances
            ]
            # Use router_attention_mask if available (masks user text tokens for image-only routing)
            # Falls back to attention_mask for backward compatibility
            attention_mask_key = "router_attention_mask" if "router_attention_mask" in instances[0] else "attention_mask"
            attention_mask = [
                torch.tensor(instance[attention_mask_key]) if isinstance(instance[attention_mask_key], list) else instance[attention_mask_key]
                for instance in instances
            ]
            labels = [
                torch.tensor(instance["labels"]) if isinstance(instance["labels"], list) else instance["labels"]
                for instance in instances
            ]

            # Pad sequences
            input_ids_padded = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.processor.tokenizer.pad_token_id
            )
            attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
                attention_mask,
                batch_first=True,
                padding_value=0
            )
            labels_padded = torch.nn.utils.rnn.pad_sequence(
                labels,
                batch_first=True,
                padding_value=IGNORE_INDEX
            )

            # Concatenate vision tensors if present
            pixel_values = None
            image_grid_thw = None

            images_list = [
                torch.tensor(instance["pixel_values"], dtype=torch.float32) if isinstance(instance["pixel_values"], list) else instance["pixel_values"]
                for instance in instances if "pixel_values" in instance
            ]
            if images_list:
                # Concatenate along dim 0 (batch dim)
                try:
                    pixel_values = torch.cat(images_list, dim=0)
                except RuntimeError as err:
                    shapes = [tuple(t.shape) for t in images_list]
                    raise RuntimeError(f"Failed to concatenate pixel_values; shapes={shapes}") from err

                # Concatenate grid metadata
                grid_list = [
                    torch.tensor(instance["image_grid_thw"], dtype=torch.long) if isinstance(instance["image_grid_thw"], list) else instance["image_grid_thw"]
                    for instance in instances if "image_grid_thw" in instance
                ]
                try:
                    image_grid_thw = torch.cat(grid_list, dim=0)
                except RuntimeError as err:
                    shapes = [tuple(t.shape) for t in grid_list]
                    raise RuntimeError(f"Failed to concatenate image_grid_thw; shapes={shapes}") from err

            # Uniform loss weights: 1.0 for all non-ignored tokens
            loss_weights = (labels_padded != -100).float()

            # Build batch dict
            batch = {
                "input_ids": input_ids_padded,
                "attention_mask": attention_mask_padded,
                "labels": labels_padded,
                "loss_weights": loss_weights,
            }

            if pixel_values is not None:
                batch["pixel_values"] = pixel_values
                batch["image_grid_thw"] = image_grid_thw

            return batch

    data_collator = DataCollatorForSupervisedDataset(processor=processor)

    eval_example_logger = EvalExampleLogger(
        tokenizer=processor.tokenizer,
        eval_dataset=val_dataset,
        log_dir=log_dir,
    )

    def _move_batch_to_device(batch, device):
        moved = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(device)
            else:
                moved[key] = value
        return moved

    def run_vision_lora_smoke_test():
        """Run a one-batch grad check to confirm vision LoRA connectivity."""
        if not enable_vision_lora:
            print("Skipping vision LoRA smoke test (vision adapters disabled).")
            return

        dataset_length = len(train_dataset)
        if dataset_length == 0:
            raise RuntimeError("Vision LoRA smoke test requires at least one training sample.")

        vision_sample = None
        probe_window = min(dataset_length, 32)
        for idx in range(probe_window):
            candidate = train_dataset[idx]
            if isinstance(candidate, dict) and "pixel_values" in candidate:
                vision_sample = candidate
                break

        if vision_sample is None:
            raise RuntimeError(
                "Vision LoRA smoke test could not find a sample with pixel_values "
                f"within the first {probe_window} training items."
            )

        batch = data_collator([vision_sample])
        target_param = next((p for p in model.parameters() if p.requires_grad), None)
        if target_param is None:
            raise RuntimeError("Vision LoRA smoke test could not locate a trainable parameter for device placement.")
        device = target_param.device
        batch_on_device = _move_batch_to_device(batch, device)

        model_state = model.training
        model.train()
        model.zero_grad(set_to_none=True)

        autocast_enabled = device.type == "cuda"
        autocast_ctx = (
            torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled)
            if hasattr(torch, "autocast")
            else nullcontext()
        )
        with autocast_ctx:
            outputs = model(**batch_on_device)
            loss = outputs.loss if hasattr(outputs, "loss") else None

        if loss is None:
            raise RuntimeError("Vision LoRA smoke test failed to compute a loss value.")

        loss.backward()

        def _collect_params(tokens):
            collector = []
            for name, param in model.named_parameters():
                if not (param.requires_grad and "lora" in name):
                    continue
                name_l = name.lower()
                if any(tok in name_l for tok in tokens):
                    collector.append(param)
            return collector

        vision_params = _collect_params(VISION_KEYWORDS + PROJECTOR_KEYWORDS)
        if not vision_params:
            raise RuntimeError("Vision LoRA smoke test found no trainable vision/projector parameters.")

        has_grad = any(param.grad is not None and torch.any(param.grad != 0) for param in vision_params)
        if not has_grad:
            raise RuntimeError("No gradients detected on vision LoRA parameters during smoke test.")

        print("✓ Vision LoRA smoke test produced gradients on vision/projector adapters.")
        model.zero_grad(set_to_none=True)
        if not model_state:
            model.eval()

    run_vision_lora_smoke_test()

    # Custom Trainer with weighted loss support
    class WeightedLossTrainer(Trainer):
        """Custom Trainer that supports per-token loss weighting with component logging."""

        def __init__(self, *args, **kwargs):
            # Extract our custom kwargs
            self.action_weight_value = kwargs.pop('action_weight', 1.5)
            self.arg_weight_value = kwargs.pop('arg_weight', 2.0)
            # Now call parent with remaining kwargs
            super().__init__(*args, **kwargs)
            # Track loss components (separate arg from other)
            self.loss_components = {'action': [], 'arg': [], 'base': []}

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            """
            Compute weighted loss if loss_weights are present in the batch.
            Track individual loss components for monitoring.
            Otherwise, fall back to standard loss computation.

            Args:
                model: The model being trained
                inputs: Input batch dictionary
                return_outputs: Whether to return model outputs along with loss
                num_items_in_batch: Number of items in batch (newer transformers versions)
            """
            # Extract loss weights if present
            loss_weights = inputs.pop("loss_weights", None)

            # Get model outputs
            outputs = model(**inputs)

            # Compute loss with component tracking
            if loss_weights is not None and "labels" in inputs:
                # Graduated weighted loss with component tracking
                logits = outputs.logits
                labels = inputs["labels"]

                # Shift logits and labels for causal LM
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                shift_weights = loss_weights[..., 1:].contiguous()

                # Flatten the tokens
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=IGNORE_INDEX)
                shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                shift_labels = shift_labels.view(-1)
                shift_weights = shift_weights.view(-1)

                # Compute per-token loss
                per_token_loss = loss_fct(shift_logits, shift_labels)

                # Track component losses for monitoring (action, arg, base)
                # Action tokens: those with weight equal to action_weight
                action_mask = torch.isclose(shift_weights, torch.tensor(self.action_weight_value, device=shift_weights.device), rtol=1e-5)
                # Arg tokens: those with weight equal to arg_weight (coordinates!)
                arg_mask = torch.isclose(shift_weights, torch.tensor(self.arg_weight_value, device=shift_weights.device), rtol=1e-5)
                # Base tokens: those with non-zero weight but not action or arg
                base_mask = (shift_weights > 0) & ~action_mask & ~arg_mask

                if action_mask.any():
                    action_loss = per_token_loss[action_mask].mean().item()
                    self.loss_components['action'].append(action_loss)

                if arg_mask.any():
                    arg_loss = per_token_loss[arg_mask].mean().item()
                    self.loss_components['arg'].append(arg_loss)

                if base_mask.any():
                    base_loss = per_token_loss[base_mask].mean().item()
                    self.loss_components['base'].append(base_loss)

                # Apply weights and average over valid tokens
                weighted_loss = per_token_loss * shift_weights
                loss = weighted_loss.sum() / (shift_weights.sum() + 1e-8)
            else:
                # Standard loss (from model)
                loss = outputs.loss

            return (loss, outputs) if return_outputs else loss

        def log(self, logs, start_time=None):
            """Override log method."""
            # Clear any accumulated loss components (not used for routing training)
            self.loss_components['action'].clear()
            self.loss_components['arg'].clear()
            self.loss_components['base'].clear()

            # Call parent log method with start_time if provided
            if start_time is not None:
                super().log(logs, start_time)
            else:
                super().log(logs)

    # Create trainer
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,  # Official Qwen3-VL collator
        callbacks=[early_stopping, cost_tracker, volume_commit_callback, eval_example_logger],
        # Pass weight values for loss component tracking
        action_weight=action_weight,
        arg_weight=arg_weight,
    )

    # Train
    print("Starting training...")
    train_result = trainer.train()

    print(f"\n✅ Training complete!")
    print(f"   Train loss: {train_result.training_loss:.4f}")
    print(f"   Total steps: {train_result.global_step}")

    # ============================================================================
    # STEP 7: Save Final Model
    # ============================================================================

    print(f"\n{'='*80}")
    print("💾 STEP 7: Saving Model")
    print(f"{'='*80}\n")

    # Save final model
    final_output_dir = f"{output_dir}/final"
    trainer.save_model(final_output_dir)

    print(f"✅ Model saved to: {final_output_dir}")

    def validate_checkpoint_artifacts(checkpoint_dir: str):
        checkpoint_dir = Path(checkpoint_dir)
        adapter_path = checkpoint_dir / "adapter_model.safetensors"
        config_path = checkpoint_dir / "adapter_config.json"
        issues = []

        if not adapter_path.exists():
            issues.append(f"Missing adapter weights: {adapter_path}")
        if not config_path.exists():
            issues.append(f"Missing adapter config: {config_path}")

        target_modules = []
        if config_path.exists():
            with open(config_path) as cfg_file:
                cfg = json.load(cfg_file)
            target_modules = cfg.get("target_modules") or []

        def _has_suffix(modules, suffixes):
            for module in modules:
                for suffix in suffixes:
                    if module.endswith(suffix) or f".{suffix}" in module:
                        return True
            return False

        if target_modules:
            if not _has_suffix(target_modules, LLM_TARGET_SUFFIXES):
                issues.append("Adapter config missing LLM target modules.")
            if enable_vision_lora and not _has_suffix(target_modules, VISION_TARGET_SUFFIXES + PROJECTOR_TARGET_SUFFIXES):
                issues.append("Adapter config missing vision/projector target modules.")

        llm_weights = False
        vision_weights = False
        if adapter_path.exists():
            from safetensors.torch import safe_open
            with safe_open(str(adapter_path), framework="pt", device="cpu") as f:
                for key in f.keys():
                    if ".lora_" not in key:
                        continue
                    if any(f".{suffix}.lora_" in key for suffix in LLM_TARGET_SUFFIXES):
                        llm_weights = True
                    if ".visual." in key:
                        vision_weights = True
                    if llm_weights and (vision_weights or not enable_vision_lora):
                        break
        if not llm_weights:
            issues.append("Adapter weights missing LLM tensors.")
        if enable_vision_lora and not vision_weights:
            issues.append("Adapter weights missing vision tensors.")

        if issues:
            error_msg = "\n - ".join(["Checkpoint validation failed:"] + issues)
            raise RuntimeError(error_msg)
        print("✅ Checkpoint validation passed (LLM + vision adapters present)")

    validate_checkpoint_artifacts(final_output_dir)

    # Commit volumes
    checkpoints_volume.commit()
    logs_volume.commit()

    print(f"\n✅ Volumes committed")

    # ============================================================================
    # STEP 8: Auto-Evaluate on Held-Out Eval Data
    # ============================================================================

    print(f"\n{'='*80}")
    print("📊 STEP 8: Running Evaluation on Held-Out Data")
    print(f"{'='*80}\n")

    eval_result = None
    # Valid adapters and system prompt - loaded from bundled config/adapters.yaml
    import yaml
    with open("/config/adapters.yaml") as f:
        config = yaml.safe_load(f)
    VALID_ADAPTERS = set(config["experts"].keys())
    eval_system_prompt = _build_system_prompt()

    try:
        from qwen_vl_utils import process_vision_info

        # Load eval data (held-out, NOT validation)
        eval_path = Path(f"/moe-data/{dataset_name}/eval.jsonl")
        if not eval_path.exists():
            eval_path = Path(f"/moe-data/datasets/{base_dataset_name}/eval.jsonl")

        if not eval_path.exists():
            print(f"⚠️  Eval data not found: {eval_path}")
            print("   Skipping evaluation...")
        else:
            eval_data = []
            with open(eval_path) as f:
                for line in f:
                    if line.strip():
                        eval_data.append(json.loads(line))

            eval_data = eval_data[:100]  # Use all 100 held-out samples
            print(f"Eval samples: {len(eval_data)}")

            # Load best checkpoint (the one the trainer loaded at the end)
            print("Using best checkpoint (loaded by trainer)...")
            model.eval()

            # Run evaluation
            results = {
                "correct": 0,
                "total": 0,
                "per_class": {adapter: {"correct": 0, "total": 0} for adapter in VALID_ADAPTERS},
            }

            images_dir = eval_path.parent / "images"

            for sample in tqdm(eval_data, desc="Evaluating"):
                expected_adapter = sample["metadata"]["adapter"]

                # Build prompt - always inject system prompt to match preprocessing
                messages = [{"role": "system", "content": eval_system_prompt}]

                for conv in sample["conversations"]:
                    # Skip any system prompts in the data - we use our own
                    if conv["from"] == "system":
                        continue
                    elif conv["from"] == "human":
                        content = []
                        value = conv["value"]
                        if "<image>" in value:
                            img_name = Path(sample["image"]).name
                            image_path = images_dir / img_name
                            if image_path.exists():
                                content.append({"type": "image", "image": f"file://{image_path}"})
                            value = value.replace("<image>", "").strip()
                        if value:
                            content.append({"type": "text", "text": value})
                        messages.append({"role": "user", "content": content})

                try:
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    image_inputs, video_inputs = process_vision_info(messages)

                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        return_tensors="pt",
                        padding=True,
                    )
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=50,
                            do_sample=False,
                            pad_token_id=processor.tokenizer.pad_token_id,
                        )

                    input_len = inputs["input_ids"].shape[1]
                    generated = processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
                    predicted_adapter = generated.lower().strip()

                    is_correct = predicted_adapter == expected_adapter

                    results["total"] += 1
                    results["per_class"][expected_adapter]["total"] += 1

                    if is_correct:
                        results["correct"] += 1
                        results["per_class"][expected_adapter]["correct"] += 1

                except Exception as e:
                    print(f"Error: {e}")
                    results["total"] += 1
                    results["per_class"][expected_adapter]["total"] += 1

            # Print results
            overall_acc = results["correct"] / results["total"] * 100 if results["total"] > 0 else 0

            print(f"\n{'='*70}")
            print("EVAL RESULTS")
            print(f"{'='*70}")
            print(f"Overall accuracy: {results['correct']}/{results['total']} ({overall_acc:.1f}%)")

            per_class_acc = {}
            for adapter in VALID_ADAPTERS:
                stats = results["per_class"][adapter]
                if stats["total"] > 0:
                    acc = stats["correct"] / stats["total"] * 100
                    per_class_acc[adapter] = acc
                    print(f"  {adapter:20s}: {stats['correct']:3d}/{stats['total']:3d} ({acc:.1f}%)")
                else:
                    per_class_acc[adapter] = None
                    print(f"  {adapter:20s}: N/A (0 samples)")

            # Build and save eval report
            eval_report = {
                "run_name": run_name,
                "dataset_name": dataset_name,
                "checkpoint": final_output_dir,
                "eval_samples": len(eval_data),
                "timestamp": datetime.now().isoformat(),
                "accuracy": {
                    "overall": overall_acc,
                    "per_class": per_class_acc,
                },
                "results": {
                    "correct": results["correct"],
                    "total": results["total"],
                    "per_class": results["per_class"],
                },
            }

            # Save eval-report.json to the dataset folder
            report_path = eval_path.parent / "eval-report.json"
            with open(report_path, "w") as f:
                json.dump(eval_report, f, indent=2)
            print(f"\n📄 Eval report saved to: {report_path}")

            # Commit volume
            moe_volume.commit()

            eval_result = eval_report

    except Exception as e:
        print(f"⚠️  Evaluation failed: {e}")
        print("   You can run evaluation manually:")
        print(f"   ./scripts/eval.sh --run-name {run_name} --dataset-name {dataset_name}")

    # ============================================================================
    # STEP 9: Summary
    # ============================================================================

    print(f"\n{'='*80}")
    print("🎉 TRAINING & EVALUATION COMPLETE!")
    print(f"{'='*80}\n")

    print(f"Run name: {run_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Checkpoints: {output_dir}")
    print(f"Logs: {log_dir}")
    print(f"Train loss: {train_result.training_loss:.4f}")
    print(f"Total steps: {train_result.global_step}")
    if eval_result:
        print(f"\nEval accuracy: {eval_result['accuracy']['overall']:.1f}%")
        print(f"Eval report: /moe-data/datasets/{base_dataset_name}/eval-report.json")
    print(f"\nTo use this model for inference:")
    print(f"  modal run modal/inference.py --checkpoint-name {run_name}/final")
    print(f"\nTo download eval report locally:")
    print(f"  uvx modal volume get {MOE_VOLUME_NAME} datasets/{base_dataset_name}/eval-report.json datasets/{base_dataset_name}/")

    # ============================================================================
    # Save comprehensive train-results.json for dashboard
    # ============================================================================

    train_results = {
        "type": "router",  # mole-trainer-server produces router models
        "dataset_name": dataset_name,
        "run_name": run_name,
        "completion_time": datetime.now().isoformat(),
        "dataset": {
            "train_samples": train_samples,
            "val_samples": val_samples,
            "total_samples": train_samples + val_samples,
        },
        "training": {
            "total_steps": train_result.global_step,
            "train_loss": train_result.training_loss,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
        },
        "eval": eval_result if eval_result else {},
        "output_dir": output_dir,
        "log_dir": log_dir,
    }

    train_results_path = Path(output_dir) / "train-results.json"
    with open(train_results_path, "w") as f:
        json.dump(train_results, f, indent=2)

    print(f"\n📄 Train results saved to: {train_results_path}")

    # Commit volume with train results
    moe_volume.commit()

    # ============================================================================
    # Save to training history (persistent across dataset cleanups)
    # ============================================================================

    print(f"\n{'='*80}")
    print("📊 STEP 10: Saving Training History")
    print(f"{'='*80}\n")

    try:
        # Read cost report for training time and cost
        cost_report_path = Path(output_dir) / "cost_report.json"
        training_time_hours = None
        total_cost_usd = None
        if cost_report_path.exists():
            with open(cost_report_path) as f:
                cost_report = json.load(f)
                training_time_hours = cost_report.get("time", {}).get("total_hours")
                total_cost_usd = cost_report.get("cost", {}).get("total_usd")

        # Build history record
        history_metrics = {
            "val_loss": train_result.training_loss,  # Final training loss as proxy
            "eval_accuracy": eval_result['accuracy']['overall'] if eval_result else None,
            "per_class_accuracy": eval_result['accuracy']['per_class'] if eval_result else {},
            "cost_usd": total_cost_usd,
            "training_time_hours": training_time_hours,
            "total_steps": train_result.global_step,
            "early_stopped": early_stopping.patience_counter >= patience,
            "dataset_size": {
                "train": train_samples,
                "val": val_samples,
                "eval": len(eval_data) if 'eval_data' in dir() else 100,
            },
            "hyperparams": {
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "vision_lora_rank": vision_lora_rank,
                "vision_lora_alpha": vision_lora_alpha,
            },
        }

        # Save to history volume (inline since we can't import from same app)
        run_id = f"{dataset_name}__{run_name}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        history_record = {
            "run_id": run_id,
            "type": "router",
            "dataset_name": dataset_name,
            "run_name": run_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "val_loss": history_metrics.get("val_loss"),
                "eval_accuracy": history_metrics.get("eval_accuracy"),
                "per_class_accuracy": history_metrics.get("per_class_accuracy", {}),
                "cost_usd": history_metrics.get("cost_usd"),
                "training_time_hours": history_metrics.get("training_time_hours"),
                "total_steps": history_metrics.get("total_steps"),
                "early_stopped": history_metrics.get("early_stopped", False),
            },
            "dataset_size": history_metrics.get("dataset_size", {}),
            "hyperparams": history_metrics.get("hyperparams", {}),
        }

        # Router history goes in /history/router
        router_dir = Path("/history/router")
        router_dir.mkdir(parents=True, exist_ok=True)

        # Save individual record
        record_path = router_dir / f"{run_id}.json"
        with open(record_path, "w") as f:
            json.dump(history_record, f, indent=2)

        # Append to JSONL log
        log_path = router_dir / "runs.jsonl"
        with open(log_path, "a") as f:
            f.write(json.dumps(history_record) + "\n")

        # Commit history volume
        history_volume.commit()

        print(f"✅ Training history saved: {run_id}")
        print(f"   Eval Accuracy: {history_record['metrics']['eval_accuracy']:.1f}%" if history_record['metrics']['eval_accuracy'] else "   Eval Accuracy: N/A")
        if history_record['metrics']['cost_usd']:
            print(f"   Cost: ${history_record['metrics']['cost_usd']:.2f}")
        print(f"   View history: uvx modal run modal/history.py")

    except Exception as e:
        print(f"⚠️  Warning: Could not save training history: {e}")
        import traceback
        traceback.print_exc()

    result = {
        "run_name": run_name,
        "dataset_name": dataset_name,
        "output_dir": output_dir,
        "train_loss": train_result.training_loss,
        "total_steps": train_result.global_step,
    }
    if eval_result:
        result["eval_accuracy"] = eval_result['accuracy']['overall']
        result["eval_report_path"] = f"/moe-data/datasets/{base_dataset_name}/eval-report.json"

    return result


# TensorBoard is served separately via projects/tensorboard-server
# Run: modal deploy projects/tensorboard-server/modal/tensorboard.py
# This keeps TensorBoard stable across training restarts.


@app.local_entrypoint()
def main(
    dataset_name: str,
    run_name: str = None,
    use_preprocessed: bool = True,
    test_mode: bool = False,
    lora_rank: int = None,  # None = auto-tune
    lora_alpha: int = None,  # None = auto-tune
    lora_dropout: float = 0.05,
    enable_vision_lora: bool = True,
    vision_lora_rank: int = None,  # None = auto-tune
    vision_lora_alpha: int = None,  # None = auto-tune
    vision_lora_dropout: float = 0.05,
    batch_size: int = None,  # None = auto-tune
    gradient_accumulation_steps: int = None,  # None = auto-tune
    learning_rate: float = None,  # None = auto-tune
    patience: int = None,  # None = auto-tune
    fast: bool = False,
):
    """
    Local entrypoint for running training.

    Usage:
        modal run modal/training.py --dataset-name calendar_20251115_042041
        modal run modal/training.py --dataset-name my_dataset --run-name my_run
        modal run modal/training.py --dataset-name my_dataset --fast --patience 2
        modal run modal/training.py --dataset-name my_dataset --use-preprocessed=False  # Use raw data
    """
    result = train_qwen3vl_lora.remote(
        run_name=run_name,
        dataset_name=dataset_name,
        use_preprocessed=use_preprocessed,
        test_mode=test_mode,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        enable_vision_lora=enable_vision_lora,
        vision_lora_rank=vision_lora_rank,
        vision_lora_alpha=vision_lora_alpha,
        vision_lora_dropout=vision_lora_dropout,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        patience=patience,
        fast=fast,
    )

    print(f"\n{'='*80}")
    print("Training job submitted to Modal!")
    print(f"{'='*80}\n")
    print(f"Results: {result}")
