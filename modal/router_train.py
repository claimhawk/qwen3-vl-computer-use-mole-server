"""Modal entrypoint to train a routing head and run gated inference with LoRAs.

Assumptions:
- Base model weights and LoRA checkpoints are available in mounted Modal volumes.
- Routing data is stored in /datasets with train.jsonl and val.jsonl files.
- Each dataset has `adapter` labels indicating which expert to route to.
- Inference uses top-1 gating to select the active LoRA and generate a response.

Dataset Structure:
- Datasets are organized in timestamped directories: /datasets/routing_YYYYMMDD_HHMMSS/
- Each directory contains train.jsonl and val.jsonl files
- Use --dataset-dir to specify the dataset directory
- Legacy --data-path and --eval-data-path flags are still supported
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import modal

# Heavy deps are imported lazily inside functions to avoid local install requirements.
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch  # noqa: F401
    from peft import PeftModel  # noqa: F401
    from torch import nn  # noqa: F401
    from torch.utils.data import DataLoader, Dataset  # noqa: F401
    from transformers import AutoTokenizer  # noqa: F401
    from transformers import Qwen3VLForConditionalGeneration  # noqa: F401
    from transformers import GenerationConfig  # noqa: F401

# Modal configuration mirrors existing training stack (CUDA keyring, torch, HF deps).
CUDA_KEYRING_URL = "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb"

app = modal.App("moe-router-train")

# Match Qwen3-VL training stack (torch 2.4 + flash-attn wheel + newer transformers).
flash_attn_wheel = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
)

IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("wget", "gnupg", "build-essential")
    .run_commands(
        f"wget {CUDA_KEYRING_URL}",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "rm cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        "apt-get install -y cuda-toolkit-12-6",
    )
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        flash_attn_wheel,
    )
    .pip_install(
        "transformers>=4.57.0",
        "accelerate>=0.27.0",
        "peft>=0.11.0",
        "qwen-vl-utils",
        "huggingface-hub>=0.20.0",
        "Pillow>=10.0.0",
        "numpy<2",
        "tqdm>=4.65.0",
        "tensorboard>=2.14.0",
    )
)

# Defaults: override via env when invoking the modal function.
DEFAULT_BASE_MODEL = os.environ.get("BASE_MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")
DEFAULT_TRAIN_DATA = os.environ.get("TRAIN_DATA", "/datasets/routing_latest/train.jsonl")
DEFAULT_VAL_DATA = os.environ.get("VAL_DATA", "/datasets/routing_latest/val.jsonl")
DEFAULT_ROUTER_OUT = os.environ.get("ROUTER_OUT", "/output/router-head.pt")
# LoRA defaults point to the checkpoints volume mount.
DEFAULT_LORA_CAL = os.environ.get("LORA_CALENDAR", "/checkpoints/calendar-tasks")
DEFAULT_LORA_CLAIM = os.environ.get("LORA_CLAIM", "/checkpoints/claim-window")
DEFAULT_LORA_PROVIDER = os.environ.get("LORA_PROVIDER", "/checkpoints/provider-select")
DEFAULT_LORA_PROVIDER = os.environ.get("LORA_PROVIDER", "/checkpoints/provider-select")

# Volumes: datasets (routing JSONL), checkpoints (LoRAs), data (legacy), output (router head + tensorboard logs).
VOLUMES = {
    "/data": modal.Volume.from_name("moe-lora-data", create_if_missing=False),
    "/checkpoints": modal.Volume.from_name("moe-lora-checkpoints", create_if_missing=False),
    "/output": modal.Volume.from_name("routing-output", create_if_missing=True),
}


@dataclass
class RoutingExample:
    text: str
    label: int


def _lazy_imports():
    import torch  # noqa: F401
    from peft import PeftModel  # noqa: F401
    from torch import nn  # noqa: F401
    from torch.utils.data import DataLoader, Dataset  # noqa: F401
    from transformers import AutoTokenizer  # noqa: F401
    from transformers import Qwen3VLForConditionalGeneration  # noqa: F401
    from transformers import GenerationConfig  # noqa: F401

    return (torch, PeftModel, nn, DataLoader, Dataset, Qwen3VLForConditionalGeneration, AutoTokenizer, GenerationConfig)


def get_hidden_size(model) -> int:
    """Extract hidden_size from a Qwen3-VL model, handling nested config structures.

    Qwen3VLConfig doesn't expose hidden_size at the top level. The hidden size
    for the text encoder is on the inner language model's config.

    Args:
        model: A Qwen3VLForConditionalGeneration model instance.

    Returns:
        The hidden size of the text encoder.

    Raises:
        AttributeError: If hidden_size cannot be found in any expected location.
    """
    # Try direct config attribute (standard transformers pattern)
    if hasattr(model.config, "hidden_size"):
        return model.config.hidden_size

    # Qwen3-VL: hidden_size is on the inner language model's config
    # The model has a `.model` attribute which is the language model backbone
    if hasattr(model, "model") and hasattr(model.model, "config"):
        if hasattr(model.model.config, "hidden_size"):
            return model.model.config.hidden_size

    # Alternative: check for text_config nested structure (some VLMs use this)
    if hasattr(model.config, "text_config") and hasattr(model.config.text_config, "hidden_size"):
        return model.config.text_config.hidden_size

    # Fallback: inspect embed_tokens to infer hidden dimension
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens.weight.shape[1]

    raise AttributeError(
        f"Cannot find hidden_size in model config. "
        f"Config type: {type(model.config).__name__}. "
        f"Available config attrs: {list(model.config.__dict__.keys())}"
    )


def build_preprocessed_routing_dataset(preprocessed_dir: Path, split: str = "train"):
    """Load preprocessed routing dataset from .pt files.

    Args:
        preprocessed_dir: Directory containing preprocessed data (e.g., /data/preprocessed/routing_xxx)
        split: Either "train" or "val"

    Returns:
        Dataset that loads preprocessed tensors from disk
    """
    torch, _, _, _, Dataset, _, _, _ = _lazy_imports()

    class _PreprocessedRoutingDataset(Dataset):
        def __init__(self, preprocessed_dir: Path, split: str) -> None:
            import random

            self.split_dir = preprocessed_dir / split
            if not self.split_dir.exists():
                raise FileNotFoundError(f"Preprocessed {split} directory not found: {self.split_dir}")

            # Find all .pt files
            self.sample_files = sorted(self.split_dir.glob("sample_*.pt"))
            if not self.sample_files:
                raise ValueError(f"No preprocessed samples found in {self.split_dir}")

            # Shuffle file paths during init for training (lazy loading pattern)
            if split == "train":
                random.shuffle(self.sample_files)

            print(f"  Loaded {len(self.sample_files)} preprocessed {split} samples from {self.split_dir}")

        def __len__(self) -> int:  # type: ignore[override]
            return len(self.sample_files)

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
            # Load preprocessed tensor from disk
            sample_path = self.sample_files[idx]
            sample = torch.load(sample_path, map_location="cpu", weights_only=False)
            return sample

    return _PreprocessedRoutingDataset(preprocessed_dir, split)


def adapters_from_data(path: Path) -> list[str]:
    """Extract unique adapter labels from a routing JSONL."""
    adapters: list[str] = []
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            adapter = record.get("adapter")
            if adapter and adapter not in seen:
                seen.add(adapter)
                adapters.append(adapter)
    if not adapters:
        adapters = ["calendar", "claim-window"]
    return adapters


def build_router_head(hidden_size: int, num_labels: int):
    torch, _, nn, _, _, _, _, _ = _lazy_imports()

    class _RouterHead(nn.Module):
        def __init__(self, hidden_size: int, num_labels: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_labels),
            )
            # Initialize weights with small values for stable training
            self._init_weights()

        def _init_weights(self) -> None:
            """Initialize weights with small std for stable classification training."""
            for module in self.net.modules():
                if isinstance(module, nn.Linear):
                    # std=0.02 works well in float32 for classification heads
                    module.weight.data.normal_(mean=0.0, std=0.02)
                    if module.bias is not None:
                        module.bias.data.zero_()

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.net(x)

    return _RouterHead(hidden_size, num_labels)


def build_routing_model(base_model_id: str, num_labels: int, device):
    torch, _, nn, _, _, Qwen3VLForConditionalGeneration, _, _ = _lazy_imports()
    RouterHead = build_router_head  # noqa: N806

    class _RoutingModel(nn.Module):
        def __init__(self, base_model_id: str, num_labels: int, device: torch.device) -> None:
            super().__init__()
            # Load encoder with bfloat16 and device_map to fit in GPU memory
            self.encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                base_model_id,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.encoder.requires_grad_(False)
            hidden_size = get_hidden_size(self.encoder)
            self.head = RouterHead(hidden_size, num_labels)
            self.device = device
            # Keep head in float32 for numerical stability (encoder outputs are bfloat16)
            self.head.to(device=device)

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            pixel_values: torch.Tensor | None = None,
            image_grid_thw: torch.Tensor | None = None,
            labels: torch.Tensor | None = None,
        ) -> torch.Tensor:  # type: ignore[override]
            # Build encoder inputs (vision + text)
            encoder_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "output_hidden_states": True,
            }
            if pixel_values is not None:
                encoder_inputs["pixel_values"] = pixel_values
            if image_grid_thw is not None:
                encoder_inputs["image_grid_thw"] = image_grid_thw

            outputs = self.encoder(**encoder_inputs)
            hidden = outputs.hidden_states[-1][:, 0, :]  # CLS/first token
            # Convert from bfloat16 to float32 for numerical stability in router head
            hidden = hidden.to(torch.float32)
            logits = self.head(hidden)
            return logits

    return _RoutingModel(base_model_id, num_labels, device)


def collate_preprocessed_fn(batch: list[dict[str, "torch.Tensor"]]):
    """Collate function for preprocessed tensors.

    Each sample in the batch is a dict with:
        - input_ids: Tensor of shape (seq_len,)
        - attention_mask: Tensor of shape (seq_len,)
        - label: Tensor scalar
        - pixel_values: Optional Tensor
        - image_grid_thw: Optional Tensor

    Returns:
        Batched tensors with padding applied to sequences
    """
    torch, _, _, _, _, _, _, _ = _lazy_imports()
    from torch.nn.utils.rnn import pad_sequence

    # Extract fields from batch
    input_ids_list = [sample["input_ids"] for sample in batch]
    attention_mask_list = [sample["attention_mask"] for sample in batch]
    labels = torch.stack([sample["label"] for sample in batch])

    # Pad sequences (padding_value=0 for both input_ids and attention_mask)
    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)

    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    # Handle optional vision fields
    if "pixel_values" in batch[0]:
        # Stack pixel values (requires batch_size=1 since images have varying patch counts)
        result["pixel_values"] = torch.stack([sample["pixel_values"] for sample in batch])

    if "image_grid_thw" in batch[0]:
        image_grid_thw_list = [sample["image_grid_thw"].squeeze(0) for sample in batch]
        result["image_grid_thw"] = torch.stack(image_grid_thw_list)

    return result


class RouterTrainer:
    """Custom Trainer for router head using HuggingFace Trainer API.

    This matches the structure of the LoRA training approach.
    """
    def __init__(self):
        from transformers import Trainer as HFTrainer
        self.HFTrainer = HFTrainer

    def create_trainer_class(self, model):
        """Create custom Trainer with compute_loss method."""
        torch, _, _, _, _, _, _, _ = _lazy_imports()
        HFTrainer = self.HFTrainer

        class _CustomRouterTrainer(HFTrainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                """Compute cross-entropy loss for router classification."""
                labels = inputs.pop("labels")

                # Forward pass through routing model
                logits = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw"),
                )

                # Compute cross-entropy loss
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)

                return (loss, {"logits": logits}) if return_outputs else loss

        return _CustomRouterTrainer


def train_router(
    preprocessed_dir: Path,
    base_model_id: str,
    num_labels: int,
    output_path: Path,
    log_dir: str = "/output/tensorboard",
    batch_size: int = 8,
    epochs: int = 100,
    lr: float = 1e-4,
    eval_steps: int = 20,
    patience: int = 3,
    seed: int = 42,
) -> None:
    """Train router head with preprocessed vision+text data using HF Trainer.

    Args:
        preprocessed_dir: Path to preprocessed routing dataset directory.
        base_model_id: HuggingFace model ID.
        num_labels: Number of routing labels (e.g., 3 for calendar/claim-window/provider-select).
        output_path: Where to save the router head.
        log_dir: Directory for TensorBoard logs (within /output volume).
        batch_size: Batch size for training.
        epochs: Maximum number of epochs (early stopping may end sooner).
        lr: Initial learning rate.
        eval_steps: Evaluate every N training steps (default 20).
        patience: Number of evals without improvement before early stopping (default 3).
        seed: Random seed.
    """
    torch, _, _, _, _, _, _, _ = _lazy_imports()
    from transformers import TrainingArguments, EarlyStoppingCallback
    from datetime import datetime
    import os

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reload the volume to see latest preprocessed data
    print("Reloading volume to access latest preprocessed data...")
    VOLUMES["/data"].reload()
    print("Volume reloaded successfully")

    # Load adapter names from dataset metadata
    dataset_name = Path(preprocessed_dir).name
    dataset_metadata_path = Path("/data/datasets") / dataset_name / "metadata.json"

    print(f"Loading adapter names from dataset metadata: {dataset_metadata_path}")

    if not dataset_metadata_path.exists():
        raise FileNotFoundError(f"Dataset metadata not found: {dataset_metadata_path}")

    import json
    with open(dataset_metadata_path, "r") as f:
        dataset_metadata = json.load(f)

    # Extract adapter names sorted by label (0, 1, 2, ...)
    adapters_dict = dataset_metadata["adapters"]
    adapter_names = [name for name, info in sorted(adapters_dict.items(), key=lambda x: x[1]["label"])]
    print(f"Loaded adapter names: {adapter_names}")

    # Verify num_labels matches
    if len(adapter_names) != num_labels:
        raise ValueError(
            f"Mismatch: Expected {len(adapter_names)} classes but num_labels={num_labels}"
        )

    # Load preprocessed datasets (lazy loading with shuffled file paths)
    train_dataset = build_preprocessed_routing_dataset(preprocessed_dir, split="train")
    eval_dataset = build_preprocessed_routing_dataset(preprocessed_dir, split="val")

    print(f"\n{'='*60}")
    print(f"Router Training Configuration")
    print(f"{'='*60}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Eval samples: {len(eval_dataset)}")
    print(f"  Num labels: {num_labels}")
    print(f"  Max epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Eval steps: {eval_steps}")
    print(f"  Patience: {patience}")
    print(f"{'='*60}\n")

    # Build routing model
    model = build_routing_model(base_model_id, num_labels=num_labels, device=device)

    # Setup output directory for HF Trainer checkpoints
    run_name = f"router_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    training_output_dir = Path(log_dir) / run_name

    # Configure TrainingArguments (matches LoRA training pattern)
    training_args = TrainingArguments(
        output_dir=str(training_output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        logging_dir=str(training_output_dir / "tensorboard"),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        seed=seed,
        bf16=True,  # Use bfloat16 to match encoder
        dataloader_num_workers=0,  # Modal volume I/O doesn't benefit from multiprocessing
        remove_unused_columns=False,  # Keep all fields from dataset
        report_to=["tensorboard"],
    )

    # Create custom trainer class
    router_trainer = RouterTrainer()
    trainer_class = router_trainer.create_trainer_class(model)

    # Initialize trainer with datasets and early stopping
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_preprocessed_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )

    # Train
    print("\n🚀 Starting training with HuggingFace Trainer...")
    trainer.train()

    print("\n✅ Training complete!")

    # Save best router head checkpoint
    print(f"\nSaving best router head to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract best metrics
    best_metrics = trainer.state.best_metric
    best_loss = best_metrics if best_metrics is not None else float("inf")

    # Save router head state dict with metadata
    torch.save(
        {
            "state_dict": model.head.state_dict(),
            "adapters": adapter_names,
            "base_model_id": base_model_id,
            "tokenizer_name": base_model_id,
            "best_loss": best_loss,
        },
        output_path,
    )
    print(f"✅ Saved best router head to {output_path}")
    print(f"   Best eval loss: {best_loss:.4f}")


def load_router_head(path: Path, hidden_size: int, device, dtype=None):
    torch, _, _, _, _, _, _, _ = _lazy_imports()
    # weights_only=False needed because checkpoint contains non-tensor data (adapters list)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    adapters = checkpoint["adapters"]
    head = build_router_head(hidden_size, num_labels=len(adapters))
    head.load_state_dict(checkpoint["state_dict"])
    # Move to device with correct dtype (bfloat16 to match encoder outputs)
    dtype = dtype or torch.bfloat16
    head.to(device=device, dtype=dtype)
    head.eval()
    return head, adapters


def load_base_with_loras(
    base_model_id: str,
    lora_paths: dict[str, str],
    dtype=None,
) -> "PeftModel":
    torch, PeftModel, _, _, _, Qwen3VLForConditionalGeneration, _, _ = _lazy_imports()
    dtype = dtype or torch.float16
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_id, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    )
    # Load first adapter directly, then add others.
    first_name, first_path = next(iter(lora_paths.items()))
    model = PeftModel.from_pretrained(base, first_path, adapter_name=first_name)
    for name, path in list(lora_paths.items())[1:]:
        model.load_adapter(path, adapter_name=name)
    return model


def route_and_generate(
    prompt: str,
    base_model_id: str,
    router_path: Path,
    lora_paths: dict[str, str],
    generation_config=None,
) -> tuple[str, str]:
    torch, PeftModel, _, _, _, Qwen3VLForConditionalGeneration, AutoTokenizer, GenerationConfigCls = _lazy_imports()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    encoder.eval()

    with torch.no_grad():
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        outputs = encoder(**encoded, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][:, 0, :]  # First token of last layer
        hidden_size = get_hidden_size(encoder)
        head, adapters = load_router_head(router_path, hidden_size=hidden_size, device=device)
        logits = head(hidden)
        pred_idx = int(logits.argmax(dim=-1).item())
        adapter_name = adapters[pred_idx]

    lora_model = load_base_with_loras(base_model_id, lora_paths)
    lora_model.set_adapter(adapter_name)
    gen_config = generation_config or GenerationConfigCls(max_new_tokens=64)

    inputs = tokenizer(prompt, return_tensors="pt").to(lora_model.device)
    with torch.no_grad():
        output_ids = lora_model.generate(**inputs, generation_config=gen_config)
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return adapter_name, response


def evaluate_router(
    data_path: Path,
    base_model_id: str,
    router_path: Path,
    adapters: list[str],
) -> dict[str, Any]:
    """Evaluate routing accuracy on labeled JSONL."""
    torch, _, _, DataLoader, _, Qwen3VLForConditionalGeneration, AutoTokenizer, _ = _lazy_imports()
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    dataset = build_routing_dataset(data_path, adapters)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=lambda b: collate_fn(tokenizer, b))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    encoder.eval()
    hidden_size = get_hidden_size(encoder)
    head, adapters_saved = load_router_head(router_path, hidden_size=hidden_size, device=device)

    correct = 0
    total = 0
    conf: dict[tuple[str, str], int] = {}
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden = outputs.hidden_states[-1][:, 0, :]  # First token of last layer
            logits = head(hidden)
            preds = logits.argmax(dim=-1).cpu()
            for label_idx, pred_idx in zip(labels, preds):
                total += 1
                gold = adapters[label_idx]
                pred = adapters_saved[pred_idx]
                if gold == pred:
                    correct += 1
                conf[(gold, pred)] = conf.get((gold, pred), 0) + 1
    accuracy = correct / total if total else 0.0
    # Convert tuple keys to strings for JSON serialization
    conf_json = {f"{gold}->{pred}": count for (gold, pred), count in conf.items()}
    return {"accuracy": accuracy, "total": total, "correct": correct, "confusion": conf_json}


@app.function(
    image=IMAGE,
    gpu="A10G",
    timeout=60 * 60 * 2,  # 2 hours for longer training with early stopping
    volumes=VOLUMES,
)
def run(
    mode: str = "train",
    preprocessed_dir: str | None = None,
    dataset_dir: str | None = None,  # Deprecated: use preprocessed_dir
    data_path: str | None = None,  # Deprecated: use preprocessed_dir
    eval_data_path: str | None = None,  # Deprecated: use preprocessed_dir
    base_model_id: str = DEFAULT_BASE_MODEL,
    router_out: str = DEFAULT_ROUTER_OUT,
    lora_calendar: str = DEFAULT_LORA_CAL,
    lora_claim: str = DEFAULT_LORA_CLAIM,
    lora_provider: str = DEFAULT_LORA_PROVIDER,
    prompt: str | None = None,
    epochs: int = 100,
    batch_size: int = 8,
    lr: float = 1e-4,
    patience: int = 5,
    min_delta: float = 0.001,
    num_labels: int = 3,  # Number of routing labels (calendar, claim-window, provider-select)
) -> str:
    """Train router head with vision+text or run routed inference.

    Volumes: /data (preprocessed data + raw datasets), /checkpoints (LoRA checkpoints), /output (router head).

    Args:
        mode: "train", "eval", or "infer"
        preprocessed_dir: Path to preprocessed dataset directory (e.g., /data/preprocessed/routing_20251124_123456/)
        dataset_dir: DEPRECATED - Path to raw dataset directory
        data_path: DEPRECATED - Explicit path to training/eval JSONL
        eval_data_path: DEPRECATED - Explicit path to eval JSONL
        base_model_id: HuggingFace model ID
        router_out: Path to save router head checkpoint
        lora_calendar: Path to calendar LoRA checkpoint
        lora_claim: Path to claim-window LoRA checkpoint
        lora_provider: Path to provider-select LoRA checkpoint
        prompt: Prompt for inference mode
        epochs: Max epochs (early stopping will end sooner)
        batch_size: Training batch size
        lr: Learning rate
        patience: Epochs without improvement before stopping
        min_delta: Minimum loss improvement to count as progress
        num_labels: Number of routing labels (default 3)
    """
    # For training mode, require preprocessed_dir
    if mode == "train":
        if preprocessed_dir is None:
            raise ValueError(
                "preprocessed_dir is required for training. "
                "Run modal/router_preprocess.py first to preprocess your dataset."
            )

        print(f"Training with preprocessed data: {preprocessed_dir}")

        train_router(
            preprocessed_dir=Path(preprocessed_dir),
            base_model_id=base_model_id,
            num_labels=num_labels,
            output_path=Path(router_out),
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            patience=patience,
        )
        return "train_complete"

    if mode == "eval":
        metrics = evaluate_router(
            data_path=Path(data_path),
            base_model_id=base_model_id,
            router_path=Path(router_out),
            adapters=adapters,
        )
        out_path = Path("/output/router-eval.json")
        out_path.write_text(json.dumps(metrics, indent=2))
        print(f"Eval metrics: {metrics}")
        return "eval_complete"

    if mode == "infer":
        if prompt is None:
            raise ValueError("prompt is required for infer mode")
        adapter, resp = route_and_generate(
            prompt=prompt,
            base_model_id=base_model_id,
            router_path=Path(router_out),
            lora_paths=lora_paths,
        )
        print(f"[router] selected adapter={adapter}")
        print(f"[model] response: {resp}")
        return adapter

    raise ValueError(f"Unknown mode: {mode}")


@app.function(
    image=IMAGE,
    volumes=VOLUMES,
)
@modal.web_server(8080, startup_timeout=60)
def tensorboard():
    """Serve TensorBoard web UI for monitoring router training."""
    import subprocess
    import sys

    cmd = [
        sys.executable, "-m", "tensorboard.main",
        "--logdir=/output/tensorboard",
        "--host=0.0.0.0",
        "--port=8080",
        "--reload_interval=30",
    ]

    subprocess.Popen(cmd)


@app.local_entrypoint()
def main(
    mode: str = "train",
    prompt: str = "Click December 3 in the calendar",
    dataset_dir: str | None = None,
    data_path: str | None = None,
    eval_data_path: str | None = None,
    epochs: int = 100,
    batch_size: int = 8,
    lr: float = 1e-4,
    patience: int = 5,
    min_delta: float = 0.001,
):
    """Local test harness: runs the Modal function with dataset directory support.

    Args:
        mode: "train", "eval", or "infer"
        dataset_dir: Path to dataset directory with train.jsonl and val.jsonl
        data_path: Explicit path to training JSONL (overrides dataset_dir)
        eval_data_path: Explicit path to eval JSONL (overrides dataset_dir)
        epochs: Max epochs (early stopping will end sooner)
        batch_size: Training batch size
        lr: Learning rate
        patience: Epochs without improvement before stopping
        min_delta: Minimum loss improvement to count as progress
    """
    result = run.remote(
        mode=mode,
        prompt=prompt,
        dataset_dir=dataset_dir,
        data_path=data_path,
        eval_data_path=eval_data_path,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        min_delta=min_delta,
    )
    print(result)
