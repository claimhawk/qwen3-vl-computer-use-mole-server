# Routing Dataset Workflow

Complete guide to generating, uploading, and training with routing datasets for the MoE LoRA router.

## Table of Contents
1. [Overview](#overview)
2. [Dataset Structure](#dataset-structure)
3. [Generation Workflow](#generation-workflow)
4. [Configuration Options](#configuration-options)
5. [Testing](#testing)
6. [Modal Integration](#modal-integration)
7. [Training with Routing Datasets](#training-with-routing-datasets)

---

## Overview

### What are Routing Datasets?

Routing datasets are classification datasets that train the MoE router to select the correct LoRA adapter for a given input. Each sample contains:
- **Text**: User input extracted from conversations
- **Label**: Integer label mapping to an adapter (0=calendar, 1=claim-window, 2=provider-select)
- **Adapter**: Human-readable adapter name for the label
- **Source ID**: Reference to the original sample

### Why Do We Need Them?

The MoE (Mixture of Experts) router learns to route inputs to specialized LoRA adapters:
- **Calendar adapter** (label 0): Handles calendar UI interactions (clicking dates, buttons)
- **Claim-window adapter** (label 1): Handles claim form UI tasks (editing fields, scrolling)
- **Provider-select adapter** (label 2): Handles provider dropdown selection tasks

By training on balanced samples from each adapter's dataset, the router learns to distinguish between task types and select the appropriate expert.

---

## Dataset Structure

### Directory Layout

```
datasets/
└── routing_20251124_123456/
    ├── data.jsonl        # All training samples combined (train + val)
    ├── train.jsonl       # 80% of training samples (default split)
    ├── val.jsonl         # 20% of training samples (for validation during training)
    └── metadata.json     # Dataset generation metadata
```

**Timestamp format**: `YYYYMMDD_HHMMSS` (automatically generated)

### File Formats

#### data.jsonl / train.jsonl / val.jsonl

Each line is a JSON object with routing information:

```jsonl
{"text": "Click December 3 in the calendar", "label": 0, "adapter": "calendar", "source_id": "cal_123"}
{"text": "Edit the claim window date field", "label": 1, "adapter": "claim-window", "source_id": "claim_456"}
{"text": "Select provider from dropdown", "label": 2, "adapter": "provider-select", "source_id": "prov_789"}
```

**Fields**:
- `text`: User input text (extracted from conversation format)
- `label`: Integer label (0, 1, or 2)
- `adapter`: Adapter name string
- `source_id`: Original sample ID from source dataset

#### metadata.json

Contains dataset generation information:

```json
{
  "dataset_name": "routing",
  "created_at": "2025-11-24T12:34:56.789012",
  "seed": 42,
  "train_val_split": 0.8,
  "adapters": {
    "calendar": {
      "label": 0,
      "source_dataset": "/path/to/calendar/train.jsonl"
    },
    "claim-window": {
      "label": 1,
      "source_dataset": "/path/to/claim-window/train.jsonl"
    },
    "provider-select": {
      "label": 2,
      "source_dataset": "/path/to/provider-select/train.jsonl"
    }
  },
  "samples": {
    "train": 240,
    "val": 60,
    "eval": 0,
    "total": 300
  },
  "label_distribution_train": {
    "calendar": 80,
    "claim-window": 80,
    "provider-select": 80
  },
  "label_distribution_val": {
    "calendar": 20,
    "claim-window": 20,
    "provider-select": 20
  }
}
```

### Label Mapping

**Fixed mapping** (defined in `scripts/generate_routing_data.py`):

```python
ADAPTER_LABELS = {
    "calendar": 0,
    "claim-window": 1,
    "provider-select": 2,
}
```

This mapping must remain consistent across generation, training, and inference.

---

## Generation Workflow

### Step 1: Generate Dataset Locally

Use `scripts/generate_routing_data.py` to create a routing dataset from source adapter datasets.

#### Basic Usage

```bash
python scripts/generate_routing_data.py \
    --calendar-dataset /Users/michaeloneal/development/claimhawk/generators/calendar/datasets/mike-im-day-clicks-system-prompt-8B_20251120_180854/train.jsonl \
    --claim-window-dataset /path/to/claim-window/train.jsonl \
    --provider-select-dataset /path/to/provider-select/train.jsonl \
    --prefix routing \
    --train-samples 100 \
    --eval-samples 100
```

#### Full Example with All Options

```bash
python scripts/generate_routing_data.py \
    --calendar-dataset /Users/michaeloneal/development/claimhawk/generators/calendar/datasets/mike-im-day-clicks-system-prompt-8B_20251120_180854/train.jsonl \
    --claim-window-dataset /Users/michaeloneal/development/claimhawk/generators/claim-window/datasets/edit-claim-window-prod_20251122_234849/train.jsonl \
    --provider-select-dataset /Users/michaeloneal/development/claimhawk/generators/provider-select/datasets/select-provider-dropdown_20251123_120000/train.jsonl \
    --prefix routing \
    --train-samples 200 \
    --eval-samples 50 \
    --train-val-split 0.8 \
    --output-dir datasets/routing_custom \
    --seed 42
```

**Output**:
```
======================================================================
Generating Routing Dataset: routing
======================================================================
Output directory: datasets/routing_20251124_143022
Train samples per adapter: 200
Eval samples per adapter: 50
Train/val split: 80%/20%
Random seed: 42
======================================================================

Loading calendar (label=0)...
  calendar              -> train= 200, eval= 50
Loading claim-window (label=1)...
  claim-window          -> train= 200, eval= 50
Loading provider-select (label=2)...
  provider-select       -> train= 200, eval= 50

Total samples:
  Training: 480
  Validation: 120
  Eval: 150

  Saved 600 samples to data.jsonl
  Saved 480 samples to train.jsonl
  Saved 120 samples to val.jsonl
  Saved metadata to metadata.json

======================================================================
Dataset generation complete!
======================================================================
```

### Step 2: Upload to Modal

Use `scripts/upload_routing_data.py` to upload the generated dataset to Modal's volume.

#### Basic Usage

```bash
python scripts/upload_routing_data.py datasets/routing_20251124_143022/
```

#### With Custom Volume/Path

```bash
python scripts/upload_routing_data.py datasets/routing_20251124_143022/ \
    --volume moe-lora-data \
    --remote-dir datasets
```

#### Dry Run (Preview Upload)

```bash
python scripts/upload_routing_data.py datasets/routing_20251124_143022/ --dry-run
```

**Output**:
```
============================================================
Uploading Dataset: routing_20251124_143022
============================================================
Local directory:  /Users/michaeloneal/development/claimhawk/moe-lora/datasets/routing_20251124_143022
Modal volume:     moe-lora-data
Remote path:      datasets/routing_20251124_143022/
Files to upload:  3
============================================================

Uploading train.jsonl -> moe-lora-data:datasets/routing_20251124_143022/train.jsonl
  ✓ Success

Uploading val.jsonl -> moe-lora-data:datasets/routing_20251124_143022/val.jsonl
  ✓ Success

Uploading metadata.json -> moe-lora-data:datasets/routing_20251124_143022/metadata.json
  ✓ Success

============================================================
✓ Upload complete!
  Uploaded 3 file(s) to moe-lora-data:datasets/routing_20251124_143022/
============================================================
```

### Step 3: Train Router

Use `modal/router_train.py` with the `--dataset-dir` flag to train the router.

#### Using Dataset Directory

```bash
modal run modal/router_train.py --dataset-dir /datasets/routing_20251124_143022
```

#### With Training Parameters

```bash
modal run modal/router_train.py \
    --dataset-dir /datasets/routing_20251124_143022 \
    --epochs 100 \
    --batch-size 8 \
    --lr 1e-4 \
    --patience 5 \
    --min-delta 0.001
```

#### Legacy: Using Explicit Paths

```bash
modal run modal/router_train.py \
    --data-path /datasets/routing_20251124_143022/train.jsonl \
    --eval-data-path /datasets/routing_20251124_143022/val.jsonl
```

**Output**:
```
============================================================
Router Training Configuration
============================================================
  Train samples: 480
  Eval samples: 120
  Max epochs: 100
  Batch size: 8
  Initial learning rate: 0.0001
  Eval every N steps: 20
  Patience (evals): 5
  Min delta: 0.001
  LR reduction factor: 0.5
  Max LR reductions: 5
  Min LR: 5e-08
============================================================

TensorBoard logs: /output/tensorboard/router_20251124_143500

Step    20: eval_loss=0.2145 eval_acc=0.9250 | ✓ improved
Step    40: eval_loss=0.1823 eval_acc=0.9417 | ✓ improved
Step    60: eval_loss=0.1654 eval_acc=0.9583 | ✓ improved
...
🔽 Reducing learning rate: 1.00e-04 → 5.00e-05
   Reduction 1/5 | Resetting patience

Step   200: eval_loss=0.0912 eval_acc=0.9750 | ✓ improved
...
⏹️  Stopping: Hit max LR reductions (5/5)
   Best eval_loss: 0.0856, Best eval_acc: 0.9833

✅ Router head saved to /output/router-head.pt
```

---

## Configuration Options

### Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--calendar-dataset` | Required | Path to calendar adapter training JSONL |
| `--claim-window-dataset` | Required | Path to claim-window adapter training JSONL |
| `--provider-select-dataset` | Required | Path to provider-select adapter training JSONL |
| `--prefix` | `routing` | Prefix for output directory name |
| `--train-samples` | `100` | Number of training samples per adapter |
| `--eval-samples` | `100` | Number of eval samples per adapter |
| `--train-val-split` | `0.8` | Train/validation split ratio (0.0-1.0) |
| `--output-dir` | Auto | Output directory (default: `datasets/{prefix}_{timestamp}`) |
| `--seed` | `42` | Random seed for reproducibility |
| `--test` | `False` | Run smoke test with 1 sample per adapter |

### Upload Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset_dir` | Required | Local dataset directory to upload |
| `--volume` | `moe-lora-data` | Modal volume name |
| `--remote-dir` | `datasets` | Remote directory prefix in volume |
| `--dry-run` | `False` | Preview upload without executing |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset-dir` | None | Dataset directory with train.jsonl and val.jsonl |
| `--data-path` | `/datasets/routing_latest/train.jsonl` | Explicit training data path (legacy) |
| `--eval-data-path` | `/datasets/routing_latest/val.jsonl` | Explicit eval data path (legacy) |
| `--epochs` | `100` | Maximum number of training epochs |
| `--batch-size` | `8` | Training batch size |
| `--lr` | `1e-4` | Initial learning rate |
| `--patience` | `5` | Epochs without improvement before LR reduction |
| `--min-delta` | `0.001` | Minimum improvement to count as progress |

---

## Testing

### Smoke Test

Run a quick test with minimal samples to verify the pipeline:

```bash
# Generate test dataset (1 sample per adapter)
python scripts/generate_routing_data.py --test
```

**What it does**:
- Sets `--train-samples 1` and `--eval-samples 1`
- Sets `--prefix routing_test`
- Uses default calendar dataset path (can override with explicit paths)
- Creates `datasets/routing_test_{timestamp}/`

**Output**:
```
Running in TEST mode (1 sample per adapter)
======================================================================
Generating Routing Dataset: routing_test
======================================================================
Output directory: datasets/routing_test_20251124_150000
Train samples per adapter: 1
Eval samples per adapter: 1
Train/val split: 80%/20%
Random seed: 42
======================================================================

Loading calendar (label=0)...
  calendar              -> train=   1, eval=  1
Loading claim-window (label=1)...
  claim-window          -> train=   1, eval=  1
Loading provider-select (label=2)...
  provider-select       -> train=   1, eval=  1

Total samples:
  Training: 2
  Validation: 1
  Eval: 3

  Saved 3 samples to data.jsonl
  Saved 2 samples to train.jsonl
  Saved 1 samples to val.jsonl
  Saved metadata to metadata.json
```

### Verification Checklist

After generating a dataset, verify:

1. **Files exist**:
   ```bash
   ls -lh datasets/routing_test_*/
   # Should show: data.jsonl, train.jsonl, val.jsonl, metadata.json
   ```

2. **Sample counts are correct**:
   ```bash
   wc -l datasets/routing_test_*/*.jsonl
   # data.jsonl: train + val samples
   # train.jsonl: ~80% of train_samples * num_adapters
   # val.jsonl: ~20% of train_samples * num_adapters
   ```

3. **Labels are valid**:
   ```bash
   # Check all labels are 0, 1, or 2
   jq .label datasets/routing_test_*/train.jsonl | sort -u
   # Output: 0, 1, 2
   ```

4. **Balanced distribution**:
   ```bash
   # Count samples per adapter
   jq -r .adapter datasets/routing_test_*/train.jsonl | sort | uniq -c
   # Should show roughly equal counts for each adapter
   ```

5. **Metadata is correct**:
   ```bash
   jq . datasets/routing_test_*/metadata.json
   # Verify adapters, sample counts, source paths
   ```

### Upload Test

Test the upload without actually uploading:

```bash
python scripts/upload_routing_data.py datasets/routing_test_*/ --dry-run
```

Verify the output shows correct paths and files.

---

## Modal Integration

### Volume Structure

The Modal training environment mounts volumes with this structure:

```
/datasets/                              # Modal volume: moe-lora-data
├── routing_20251124_143022/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── metadata.json
├── routing_20251123_090000/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── metadata.json
└── routing_latest/                     # Symlink or latest dataset
    ├── train.jsonl
    └── val.jsonl

/checkpoints/                           # Modal volume: moe-lora-checkpoints
├── calendar-tasks/                     # LoRA adapter checkpoints
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── claim-window/
└── provider-select/

/output/                                # Modal volume: routing-output
├── router-head.pt                      # Trained router checkpoint
└── tensorboard/                        # TensorBoard logs
    └── router_20251124_143500/
```

### How the Trainer Finds Data

The `modal/router_train.py` script resolves data paths with this priority:

1. **Explicit paths** (`--data-path` and `--eval-data-path`): Use these exact paths
2. **Dataset directory** (`--dataset-dir`): Look for `train.jsonl` and `val.jsonl` inside
3. **Default paths**: Use `/datasets/routing_latest/train.jsonl` and `val.jsonl`

**Example**: Using dataset directory (recommended):

```python
# In modal/router_train.py
if data_path is None:
    if dataset_dir:
        data_path = str(Path(dataset_dir) / "train.jsonl")
    else:
        data_path = DEFAULT_TRAIN_DATA  # /datasets/routing_latest/train.jsonl

if eval_data_path is None:
    if dataset_dir:
        eval_data_path = str(Path(dataset_dir) / "val.jsonl")
    else:
        eval_data_path = DEFAULT_VAL_DATA  # /datasets/routing_latest/val.jsonl
```

### Legacy Path Support

For backward compatibility, the trainer still supports:

```bash
modal run modal/router_train.py \
    --data-path /data/routing-train.jsonl \
    --eval-data-path /data/routing-eval.jsonl
```

This uses the older volume mount at `/data` (volume: `claimhawk-training-data`).

### Environment Variables

You can override defaults via environment variables:

```bash
# Set default paths
export TRAIN_DATA=/datasets/my_routing/train.jsonl
export VAL_DATA=/datasets/my_routing/val.jsonl
export BASE_MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
export ROUTER_OUT=/output/router-custom.pt

# Run training
modal run modal/router_train.py
```

---

## Training with Routing Datasets

### Full Training Example

```bash
# 1. Generate dataset locally
python scripts/generate_routing_data.py \
    --calendar-dataset /Users/michaeloneal/development/claimhawk/generators/calendar/datasets/mike-im-day-clicks-system-prompt-8B_20251120_180854/train.jsonl \
    --claim-window-dataset /path/to/claim-window/train.jsonl \
    --provider-select-dataset /path/to/provider-select/train.jsonl \
    --train-samples 500 \
    --eval-samples 100

# Note the output directory: datasets/routing_20251124_160000/

# 2. Upload to Modal
python scripts/upload_routing_data.py datasets/routing_20251124_160000/

# 3. Train router on Modal
modal run modal/router_train.py \
    --dataset-dir /datasets/routing_20251124_160000 \
    --epochs 200 \
    --batch-size 16 \
    --lr 1e-4 \
    --patience 5

# 4. Monitor training (optional)
modal app logs moe-router-train

# 5. Serve TensorBoard (optional)
modal serve modal/router_train.py::tensorboard
```

### Monitoring Training

**View logs in real-time**:
```bash
modal app logs moe-router-train --follow
```

**Serve TensorBoard**:
```bash
modal serve modal/router_train.py::tensorboard
# Open the provided URL in your browser
```

**Download trained router**:
```bash
modal volume get routing-output router-head.pt ./router-head.pt
```

### Evaluation Mode

After training, evaluate the router on held-out data:

```bash
modal run modal/router_train.py \
    --mode eval \
    --dataset-dir /datasets/routing_20251124_160000
```

**Output**: Saves metrics to `/output/router-eval.json`:
```json
{
  "accuracy": 0.9833,
  "total": 120,
  "correct": 118,
  "confusion": {
    "calendar->calendar": 40,
    "claim-window->claim-window": 39,
    "claim-window->calendar": 1,
    "provider-select->provider-select": 39
  }
}
```

### Inference Mode

Test the router with a sample prompt:

```bash
modal run modal/router_train.py \
    --mode infer \
    --prompt "Click December 3 in the calendar"
```

**Output**:
```
[router] selected adapter=calendar
[model] response: Clicking December 3rd...
```

---

## Best Practices

### Sample Counts

- **Minimum**: 50 samples per adapter (150 total) for basic functionality
- **Recommended**: 200-500 samples per adapter (600-1500 total) for good performance
- **Large**: 1000+ samples per adapter (3000+ total) for production use

### Train/Val Split

- **Default**: 0.8 (80% train, 20% validation)
- Use 0.9 for small datasets (< 100 samples per adapter)
- Use 0.7-0.8 for large datasets (> 500 samples per adapter)

### Reproducibility

Always set a seed for reproducible dataset generation:

```bash
python scripts/generate_routing_data.py \
    --seed 42 \
    ...
```

### Dataset Versioning

Use descriptive prefixes and keep the timestamp:

```bash
# Good: Indicates dataset purpose and version
--prefix routing_v2_balanced_500
# Output: datasets/routing_v2_balanced_500_20251124_160000/

# Okay: Simple prefix with auto timestamp
--prefix routing
# Output: datasets/routing_20251124_160000/

# Avoid: No timestamp makes it hard to track versions
--output-dir datasets/routing/
```

### Monitoring

- Always enable validation data (`--eval-data-path` or use `val.jsonl` in dataset-dir)
- Monitor TensorBoard for training curves
- Check for overfitting (train acc >> val acc)
- Watch for LR reductions - too many may indicate poor initialization or data quality

---

## Troubleshooting

### "No samples loaded from dataset"

**Cause**: Source dataset doesn't have the expected conversation format.

**Solution**: Verify the source JSONL has `conversations` field with `from: human` entries:
```bash
head -1 /path/to/source/train.jsonl | jq .conversations
```

### "Dataset directory not found"

**Cause**: Path to dataset is incorrect or dataset wasn't generated.

**Solution**: Check the path and verify files exist:
```bash
ls -lh datasets/routing_*/
```

### "uvx or modal command not found"

**Cause**: Modal CLI not installed.

**Solution**: Install Modal:
```bash
pip install modal
modal setup  # Authenticate with Modal
```

### Upload fails silently

**Cause**: Modal volume doesn't exist or authentication expired.

**Solution**:
```bash
# Re-authenticate
modal token set --token-id YOUR_TOKEN_ID --token-secret YOUR_SECRET

# Verify volume exists
modal volume list
```

### Training OOM (Out of Memory)

**Cause**: Batch size too large for GPU memory.

**Solution**: Reduce batch size:
```bash
modal run modal/router_train.py \
    --batch-size 4 \
    ...
```

### Low routing accuracy (< 80%)

**Causes**:
- Imbalanced dataset (some adapters have many more samples)
- Poor quality source data (text extraction fails)
- Task overlap (similar tasks in different adapters)

**Solutions**:
- Verify label distribution in `metadata.json`
- Increase sample count per adapter
- Check that source datasets are high-quality
- Review confusion matrix to identify problematic pairs

---

## Quick Reference

### Common Commands

```bash
# Generate test dataset
python scripts/generate_routing_data.py --test

# Generate production dataset
python scripts/generate_routing_data.py \
    --calendar-dataset /path/to/calendar/train.jsonl \
    --claim-window-dataset /path/to/claim/train.jsonl \
    --provider-select-dataset /path/to/provider/train.jsonl \
    --train-samples 500

# Upload dataset
python scripts/upload_routing_data.py datasets/routing_YYYYMMDD_HHMMSS/

# Train router
modal run modal/router_train.py --dataset-dir /datasets/routing_YYYYMMDD_HHMMSS

# Evaluate router
modal run modal/router_train.py --mode eval --dataset-dir /datasets/routing_YYYYMMDD_HHMMSS

# Test inference
modal run modal/router_train.py --mode infer --prompt "Click the calendar"
```

### File Locations

| File | Location |
|------|----------|
| Generation script | `scripts/generate_routing_data.py` |
| Upload script | `scripts/upload_routing_data.py` |
| Training script | `modal/router_train.py` |
| Local datasets | `./datasets/routing_*/` |
| Modal datasets | `/datasets/routing_*/` (volume: moe-lora-data) |
| Modal checkpoints | `/checkpoints/` (volume: moe-lora-checkpoints) |
| Trained router | `/output/router-head.pt` (volume: routing-output) |
| TensorBoard logs | `/output/tensorboard/` (volume: routing-output) |

---

## Related Documentation

- [Router Data & Eval Plan](router-data-eval-plan.md) - Original planning document
- [Router Train Blocker](router-train-blocker.md) - Known issues and blockers
- [MoE GPT Notes](moe-gpt.md) - MoE architecture background

---

**Last updated**: 2025-11-24
