# Router as LoRA Adapter - Implementation Guide

## Overview

**The router is just another LoRA adapter** that outputs adapter names instead of tool calls.

```
Stack of independent layers:
├── Base: Qwen3-VL-8B-Instruct
├── Layer 1 (Router LoRA): Outputs "calendar" | "claim-window" | "provider-select"
├── Layer 2 (Expert LoRAs): calendar, claim-window, provider-select
```

## Current Implementation

### Volumes (Modal)
- `moe-lora-data`: Routing datasets, checkpoints
- `moe-lora-checkpoints`: Task LoRA adapters
- `claimhawk-training-data`: Source training data

### Scripts
```bash
scripts/generate_dataset.sh  # Generate routing dataset on Modal
scripts/preprocess.sh        # Preprocess for training
scripts/train.sh             # Train routing LoRA
scripts/eval.sh              # Evaluate accuracy
```

### Modal Modules
```bash
modal/generate_routing.py    # Dataset generation from source volumes
modal/preprocess.py          # Preprocessing
modal/training.py            # LoRA training with SFTTrainer
modal/eval.py                # Evaluation
modal/stacked_inference.py   # Full pipeline: route → task LoRA → output
```

## Workflow

### 1. Generate Routing Dataset

```bash
./scripts/generate_dataset.sh
```

This runs `modal/generate_routing.py` which:
- Reads from `claimhawk-training-data` volume (source datasets)
- Creates balanced train/val/eval splits
- Saves to `moe-lora-data` volume
- Downloads locally for inspection

**Output**: `datasets/routing_YYYYMMDD_HHMMSS/`
```
├── train.jsonl   # Training data
├── val.jsonl     # Validation (early stopping)
├── eval.jsonl    # Held-out evaluation
├── images/       # Referenced screenshots
└── metadata.json
```

**Sample format**:
```json
{
  "id": "claim-window_00447",
  "image": "images/claim-window_00447.png",
  "conversations": [
    {"from": "system", "value": "You are a routing classifier..."},
    {"from": "human", "value": "<image>\nWhich adapter should handle this?"},
    {"from": "gpt", "value": "claim-window"}
  ],
  "metadata": {"adapter": "claim-window", "label": 1}
}
```

### 2. Preprocess Dataset

```bash
./scripts/preprocess.sh --dataset-name datasets/routing_YYYYMMDD_HHMMSS
```

### 3. Train Router LoRA

```bash
./scripts/train.sh --dataset-name datasets/routing_YYYYMMDD_HHMMSS --run-name my-routing-lora
```

This runs `modal/training.py` using SFTTrainer to fine-tune a LoRA adapter.

### 4. Evaluate

```bash
./scripts/eval.sh --run-name my-routing-lora --dataset-name datasets/routing_YYYYMMDD_HHMMSS
```

## Stacked Inference

The full pipeline runs in `modal/stacked_inference.py`:

```python
# 1. Load routing LoRA
routing_model = PeftModel.from_pretrained(base_model, routing_lora_path)

# 2. Route: get adapter name
adapter_name = generate(routing_model, image, prompt)  # "calendar"

# 3. Load fresh base + task LoRA (fresh model to avoid adapter conflicts)
task_base = Qwen3VLForConditionalGeneration.from_pretrained(...)
task_model = PeftModel.from_pretrained(task_base, task_lora_path)

# 4. Execute task
result = generate(task_model, image, task_prompt)  # <tool_call>...</tool_call>
```

**Key insight**: Load a fresh base model for task inference to avoid PeftModel conflicts.

### Running Stacked Inference

```bash
modal run modal/stacked_inference.py \
  --routing-run routing-20251125-052946 \
  --routing-dataset datasets/routing_20251125_052946 \
  --max-samples 10
```

## Task LoRA Paths

Defined in `config/loras.json`:
```json
{
  "loras": {
    "calendar": {"volume": "moe-lora-checkpoints", "path": "calendar-tasks"},
    "claim-window": {"volume": "moe-lora-checkpoints", "path": "claim-window"},
    "provider-select": {"volume": "moe-lora-checkpoints", "path": "provider-select"}
  }
}
```

The inference code auto-detects `/final` subfolder for backwards compatibility.

## Adding New Experts

### 1. Train the new task LoRA
Use your existing training pipeline to create the new adapter.

### 2. Upload to Modal volume
```bash
./scripts/save_lora.sh new-adapter /path/to/checkpoint
```

### 3. Regenerate routing dataset with 4 classes
Update `modal/generate_routing.py` ADAPTER_DATASETS to include new source.

### 4. Retrain router
```bash
./scripts/generate_dataset.sh
./scripts/preprocess.sh --dataset-name datasets/routing_NEW
./scripts/train.sh --dataset-name datasets/routing_NEW --run-name routing-4way
```

### 5. Update config/loras.json
Add the new adapter path.

## Results

Current performance (routing_20251125_052946):
- **Routing accuracy**: 100% (9/9 on sample)
- **Stacked inference**: 100% (all samples produce valid `<tool_call>` output)

**Last updated**: 2025-11-25
