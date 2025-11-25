# Vision-Enabled Router Training Workflow

This document describes the complete workflow for training a vision+text router for LoRA adapter selection.

## Architecture

```
[Image + Text Prompt] → Qwen3-VL Encoder (frozen) → Hidden States → Router Head (trainable) → [Label 0,1,2]
```

The router sees the SAME multimodal input (image + text) as the LoRA adapters, ensuring consistent routing decisions.

## Labels

- `0`: calendar
- `1`: claim-window
- `2`: provider-select

## Workflow Steps

### 1. Generate Routing Dataset

Create a routing dataset from source adapter datasets. This preserves the full multimodal format (conversations with images).

```bash
python scripts/generate_routing_data.py \
    --calendar-dataset /path/to/calendar/train.jsonl \
    --claim-window-dataset /path/to/claim-window/train.jsonl \
    --provider-select-dataset /path/to/provider-select/train.jsonl \
    --train-samples 100 \
    --eval-samples 100
```

**Output**: `./datasets/routing_YYYYMMDD_HHMMSS/`
- `train.jsonl` - 80% of training samples
- `val.jsonl` - 20% of training samples
- `metadata.json` - Dataset generation info

**Dataset Format** (preserves full multimodal format):
```json
{
  "id": "...",
  "image": "path/to/image.png",
  "conversations": [
    {"from": "human", "value": "<image>\nClick March 29 in the calendar"}
  ],
  "metadata": {...},
  "label": 0,
  "adapter": "calendar"
}
```

### 2. Upload to Modal

Upload the dataset to Modal's `moe-lora-data` volume.

```bash
python scripts/upload_routing_data.py routing_20251124_115228
```

**Output**: Dataset uploaded to `/datasets/routing_20251124_115228/` on Modal volume

### 3. Preprocess Dataset

Preprocess the dataset on Modal CPU instances. This:
- Loads raw JSONL + images
- Processes through Qwen3-VL processor (tokenize + load images)
- Saves preprocessed tensors as `.pt` files

```bash
modal run modal/router_preprocess.py --dataset-dir routing_20251124_115228
```

**Output**: Preprocessed tensors in `/data/preprocessed/routing_20251124_115228/`
- `train/sample_000000.pt`, `train/sample_000001.pt`, ...
- `val/sample_000000.pt`, `val/sample_000001.pt`, ...
- `metadata.json`

**Preprocessed Sample Format**:
```python
{
    "input_ids": Tensor[seq_len],           # Tokenized text
    "attention_mask": Tensor[seq_len],      # Attention mask
    "label": Tensor[],                      # Routing label (0, 1, or 2)
    "pixel_values": Tensor[...],            # Processed image tensor
    "image_grid_thw": Tensor[...]           # Image grid info
}
```

### 4. Train Router

Train the router head on preprocessed vision+text data.

```bash
modal run modal/router_train.py::run \
    --mode train \
    --preprocessed-dir /data/preprocessed/routing_20251124_115228 \
    --base-model-id Qwen/Qwen3-VL-8B-Instruct \
    --router-out /output/router-head.pt \
    --epochs 100 \
    --batch-size 8 \
    --lr 1e-4 \
    --patience 5 \
    --num-labels 3
```

**Training Details**:
- Encoder: Frozen Qwen3-VL (bfloat16)
- Router Head: Trainable linear layers (bfloat16)
- Optimizer: AdamW with LR reduction on plateau
- Early Stopping: Patience = 5 evals (eval every 20 steps)
- No label smoothing (only 3 classes)

**Output**:
- Router head checkpoint: `/output/router-head.pt`
- TensorBoard logs: `/output/tensorboard/router_YYYYMMDD_HHMMSS/`

## Key Differences from Text-Only Approach

### Before (Text-Only)
```python
# Generator extracted text only
{"text": "Click March 29 in the calendar", "label": 0}

# Training used text tokenization only
model(input_ids=..., attention_mask=...)
```

### After (Vision+Text)
```python
# Generator preserves full multimodal format
{
    "conversations": [...],
    "image": "path/to/image.png",
    "label": 0,
    "adapter": "calendar"
}

# Preprocessing creates vision+text tensors
{
    "input_ids": ...,
    "attention_mask": ...,
    "pixel_values": ...,  # NEW: Image tensor
    "image_grid_thw": ..., # NEW: Image grid info
    "label": ...
}

# Training uses full multimodal input
model(
    input_ids=...,
    attention_mask=...,
    pixel_values=...,      # NEW
    image_grid_thw=...     # NEW
)
```

## Performance Benefits

1. **Preprocessing on CPU**: Expensive image loading and tokenization happens once on cheap CPU instances
2. **Fast GPU Training**: Training loads preprocessed `.pt` files directly, no on-the-fly processing
3. **Reusable**: Preprocessed data can be reused across multiple training runs

## File Locations

### Local
- Generator: `scripts/generate_routing_data.py`
- Uploader: `scripts/upload_routing_data.py`
- Datasets: `./datasets/routing_YYYYMMDD_HHMMSS/`

### Modal Volumes
- Raw datasets: `moe-lora-data:/datasets/routing_YYYYMMDD_HHMMSS/`
- Preprocessed: `moe-lora-data:/preprocessed/routing_YYYYMMDD_HHMMSS/`
- Router output: `routing-output:/router-head.pt`
- LoRA checkpoints: `moe-lora-checkpoints:/`

### Modal Scripts
- Preprocessor: `modal/router_preprocess.py`
- Trainer: `modal/router_train.py`
