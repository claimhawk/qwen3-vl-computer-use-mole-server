#!/usr/bin/env python3
# Copyright (c) 2025 Tylt LLC. All rights reserved.
# Licensed for research use only. Commercial use requires a license from Tylt LLC.
# Contact: hello@claimhawk.app | See LICENSE for terms.

"""
Router Training from Linked Expert Datasets

Trains the router model using preprocessed data directly from expert datasets.
Labels are remapped at runtime - no reprocessing needed.

Usage:
    modal run modal/train_router.py --manifest router_linked_20251209_204654
"""

import json
import os
import random
from datetime import datetime
from pathlib import Path

import modal

# Volume names
MOE_VOLUME_NAME = "moe-lora-data"
EXPERT_VOLUME_NAME = "claimhawk-lora-training"
MODEL_CACHE_NAME = "claimhawk-model-cache"
BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

app = modal.App("router-training-linked")

moe_volume = modal.Volume.from_name(MOE_VOLUME_NAME, create_if_missing=True)
expert_volume = modal.Volume.from_name(EXPERT_VOLUME_NAME, create_if_missing=False)
model_cache = modal.Volume.from_name(MODEL_CACHE_NAME, create_if_missing=True)

# Pre-compiled flash-attn wheel (matches training.py)
flash_attn_wheel = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04",
        add_python="3.11"
    )
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        flash_attn_wheel,
    )
    .pip_install(
        "transformers>=4.57.0",
        "accelerate>=0.26.0",
        "peft>=0.14.0",
        "datasets>=2.16.0",
        "tensorboard>=2.15.0",
        "qwen-vl-utils",
        "Pillow>=10.0.0",
        "tqdm>=4.65.0",
    )
)


@app.function(
    image=image,
    gpu="H100",  # Use H100 for more memory headroom
    timeout=14400,  # 4 hours
    volumes={
        "/moe-data": moe_volume,
        "/expert-data": expert_volume,
        "/models": model_cache,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train_router(
    manifest_name: str,
    run_name: str = None,
    num_epochs: int = 3,
    batch_size: int = 1,  # Reduced to 1 to avoid OOM with variable-length sequences
    learning_rate: float = 2e-4,
    lora_rank: int = 64,
    max_samples_per_expert: int = None,
    seed: int = 42,
):
    """Train router model from linked expert datasets."""
    import torch
    from transformers import (
        AutoProcessor,
        Qwen3VLForConditionalGeneration,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback,
    )
    from peft import LoraConfig, get_peft_model
    from tqdm import tqdm

    random.seed(seed)
    torch.manual_seed(seed)

    # Reload volumes
    moe_volume.reload()
    expert_volume.reload()

    # Generate run name
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"router_{timestamp}"

    print(f"\n{'='*70}")
    print("Router Training from Expert Datasets")
    print(f"{'='*70}")
    print(f"Manifest: {manifest_name}")
    print(f"Run: {run_name}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"LoRA rank: {lora_rank}")

    # Load manifest
    manifest_path = Path(f"/moe-data/router_manifests/{manifest_name}.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"\nLoaded manifest: {manifest['name']}")
    print(f"Experts: {list(manifest['experts'].keys())}")

    # Build file list with labels
    train_files = []  # List of (pt_path, label)
    val_files = []

    for expert_name, info in manifest["experts"].items():
        label = info["label"]
        train_dir = Path(f"/expert-data/{info['train_path']}")
        val_dir = Path(f"/expert-data/{info['val_path']}")

        # Get held-out test indices (these should NOT be used for training)
        test_indices = set(info.get("test_indices", []))

        # Use train_count from manifest (set by max_records), or CLI override
        train_limit = info.get("train_count")
        val_limit = info.get("val_count")
        if max_samples_per_expert:
            train_limit = max_samples_per_expert
            val_limit = max_samples_per_expert // 4

        # Get training files, excluding test indices
        all_train_files = sorted(train_dir.glob("sample_*.pt"))
        expert_train_files = [
            f for i, f in enumerate(all_train_files) if i not in test_indices
        ]

        if train_limit and len(expert_train_files) > train_limit:
            random.shuffle(expert_train_files)
            expert_train_files = expert_train_files[:train_limit]

        for f in expert_train_files:
            train_files.append((f, label, expert_name))

        # Get validation files
        if val_dir.exists():
            expert_val_files = sorted(val_dir.glob("sample_*.pt"))
            if val_limit and len(expert_val_files) > val_limit:
                random.shuffle(expert_val_files)
                expert_val_files = expert_val_files[:val_limit]

            for f in expert_val_files:
                val_files.append((f, label, expert_name))

        held_out = len(test_indices)
        train_count = len([x for x in train_files if x[2] == expert_name])
        print(f"  {expert_name}: {train_count} train (held out {held_out} for test)")

    random.shuffle(train_files)
    random.shuffle(val_files)

    print(f"\nTotal train: {len(train_files)}")
    print(f"Total val: {len(val_files)}")

    # Create dataset class that remaps labels
    class LinkedRouterDataset(torch.utils.data.Dataset):
        """Dataset that loads expert .pt files and remaps labels to routing labels."""

        def __init__(self, files_with_labels, tokenizer):
            self.files = files_with_labels
            self.tokenizer = tokenizer
            # Pre-tokenize all possible labels
            self.label_tokens = {}
            for label in range(7):  # Labels 0-6
                tokens = tokenizer.encode(str(label), add_special_tokens=False)
                tokens.append(151645)  # <|im_end|>
                self.label_tokens[label] = torch.tensor(tokens, dtype=torch.int64)

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            pt_path, label, expert_name = self.files[idx]

            # Load the expert sample
            sample = torch.load(pt_path, weights_only=False)

            input_ids = sample["input_ids"]
            attention_mask = sample["attention_mask"]
            pixel_values = sample["pixel_values"]
            image_grid_thw = sample["image_grid_thw"]
            original_labels = sample["labels"]

            # Find where the label starts (first non -100)
            label_start = None
            for i, val in enumerate(original_labels.tolist()):
                if val != -100:
                    label_start = i
                    break

            if label_start is None:
                # Fallback: use full sequence
                label_start = len(input_ids) - 10

            # Get new label tokens
            new_label = self.label_tokens[label]

            # Truncate input to before label, append new label
            new_input_ids = torch.cat([input_ids[:label_start], new_label])
            new_attention_mask = torch.cat([
                attention_mask[:label_start],
                torch.ones(len(new_label), dtype=torch.int64)
            ])

            # Create new labels (-100 for input, actual tokens for label)
            new_labels = torch.full((len(new_input_ids),), -100, dtype=torch.int64)
            new_labels[-len(new_label):] = new_label

            return {
                "input_ids": new_input_ids,
                "attention_mask": new_attention_mask,
                "labels": new_labels,
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            }

    # Load model
    print(f"\n{'='*70}")
    print("Loading Model")
    print(f"{'='*70}")

    cached_model_path = f"/models/{BASE_MODEL.replace('/', '--')}"
    if Path(cached_model_path).exists():
        print(f"Using cached model: {cached_model_path}")
        model_path = cached_model_path
    else:
        print(f"Downloading from HuggingFace: {BASE_MODEL}")
        model_path = BASE_MODEL

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Create datasets
    train_dataset = LinkedRouterDataset(train_files, processor.tokenizer)
    val_dataset = LinkedRouterDataset(val_files, processor.tokenizer)

    # Data collator
    class DataCollatorForRouter:
        def __init__(self, processor):
            self.processor = processor

        def __call__(self, instances):
            input_ids = [inst["input_ids"] for inst in instances]
            attention_mask = [inst["attention_mask"] for inst in instances]
            labels = [inst["labels"] for inst in instances]
            pixel_values = [inst["pixel_values"] for inst in instances]
            image_grid_thw = [inst["image_grid_thw"] for inst in instances]

            # Pad sequences
            max_len = max(len(ids) for ids in input_ids)

            padded_input_ids = []
            padded_attention_mask = []
            padded_labels = []

            for ids, mask, lab in zip(input_ids, attention_mask, labels):
                pad_len = max_len - len(ids)
                padded_input_ids.append(torch.cat([ids, torch.zeros(pad_len, dtype=torch.int64)]))
                padded_attention_mask.append(torch.cat([mask, torch.zeros(pad_len, dtype=torch.int64)]))
                padded_labels.append(torch.cat([lab, torch.full((pad_len,), -100, dtype=torch.int64)]))

            return {
                "input_ids": torch.stack(padded_input_ids),
                "attention_mask": torch.stack(padded_attention_mask),
                "labels": torch.stack(padded_labels),
                "pixel_values": torch.cat(pixel_values, dim=0),
                "image_grid_thw": torch.cat(image_grid_thw, dim=0),
            }

    # Training arguments
    output_dir = f"/moe-data/checkpoints/router_linked/{run_name}"
    log_dir = f"/moe-data/tb_logs/{run_name}"  # Use tb_logs for TensorBoard visibility
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=log_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=8,  # Increased to compensate for batch_size=1
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=20,  # Eval every 20 steps
        save_strategy="steps",
        save_steps=20,  # Save every 20 steps to match eval
        save_total_limit=5,  # Keep more checkpoints for early stopping
        bf16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,  # Lower eval_loss is better
    )

    # Create trainer with early stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForRouter(processor),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # Train
    print(f"\n{'='*70}")
    print("Starting Training")
    print(f"{'='*70}")

    trainer.train()

    # Save final model
    final_path = Path(output_dir) / "final"
    trainer.save_model(final_path)
    print(f"\nModel saved to: {final_path}")

    # Final evaluation on validation set
    print(f"\n{'='*70}")
    print("Final Evaluation")
    print(f"{'='*70}")

    final_metrics = trainer.evaluate()
    print(f"Final eval_loss: {final_metrics.get('eval_loss', 'N/A')}")

    # Save metrics to JSON
    metrics_path = Path(output_dir) / "final_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    # Commit volume
    moe_volume.commit()

    return {
        "run_name": run_name,
        "output_dir": output_dir,
        "train_samples": len(train_files),
        "val_samples": len(val_files),
        "final_eval_loss": final_metrics.get("eval_loss"),
    }


@app.local_entrypoint()
def main(
    manifest: str,
    run_name: str = None,
    epochs: int = 3,
    batch_size: int = 1,  # Reduced to avoid OOM with variable-length sequences
    lr: float = 2e-4,
    lora_rank: int = 64,
    max_per_expert: int = None,
):
    """Train router from linked expert datasets."""
    result = train_router.remote(
        manifest_name=manifest,
        run_name=run_name,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        lora_rank=lora_rank,
        max_samples_per_expert=max_per_expert,
    )

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Run: {result['run_name']}")
    print(f"Output: {result['output_dir']}")
    print(f"Train samples: {result['train_samples']}")
    print(f"Val samples: {result['val_samples']}")
