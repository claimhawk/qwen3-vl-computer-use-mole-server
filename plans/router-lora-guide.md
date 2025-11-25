# Router as LoRA Adapter - Implementation Guide

## Overview

**The router is just another LoRA adapter** that outputs adapter names instead of tool calls.

```
Stack of independent layers:
├── Base: Qwen3-VL-8B-Instruct
├── Layer 1 (Router LoRA): Outputs "calendar" | "claim-window" | "provider-select"
├── Layer 2 (Expert LoRAs): calendar, claim-window, provider-select, [chandra-ocr]
```

## Benefits

✅ Uses **existing LoRA training infrastructure** (zero new code!)
✅ Router is just another adapter (same patterns as other experts)
✅ Truly modular: layers don't interfere with each other
✅ Easy to extend: add chandra-ocr by retraining router with 4th class
✅ Simple inference: load router → classify → swap to expert

## Dataset Preparation

### 1. Generate routing dataset

Already done:
```bash
datasets/routing_20251124_150334/
├── train.jsonl (5,131 samples - 2,836 calendar, 1,200 claim-window, 1,200 provider-select)
├── eval.jsonl (921 samples)
└── images/
```

### 2. Convert to classification format

```bash
python3 scripts/convert_routing_to_classification.py datasets/routing_20251124_150334
```

Output:
```bash
datasets/routing_20251124_150334/
├── train-classification.jsonl  # Model learns to output adapter names
├── eval-classification.jsonl
└── ...
```

**Format example:**
```json
{
  "id": "claim-window_00447",
  "image": "images/claim-window_00447.jpg",
  "conversations": [
    {
      "from": "system",
      "value": "You are a routing classifier for a multi-adapter system..."
    },
    {
      "from": "human",
      "value": "<image>\nWhich adapter should handle this screenshot?"
    },
    {
      "from": "gpt",
      "value": "provider-select"
    }
  ]
}
```

## Training the Router LoRA

Use your **existing LoRA training script** (the one you used for calendar/claim-window):

```bash
# Point to your existing calendar training script location
cd /path/to/trainers/calendar/qwenvl/train

# Train router LoRA (same command as other adapters!)
python train_qwen.py \
  --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
  --data_path /path/to/moe-lora/datasets/routing_20251124_150334/train-classification.jsonl \
  --eval_data_path /path/to/moe-lora/datasets/routing_20251124_150334/eval-classification.jsonl \
  --output_dir ./checkpoints/router-classifier \
  --num_train_epochs 10 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --evaluation_strategy "epoch" \
  --save_strategy "epoch" \
  --save_total_limit 3 \
  --learning_rate 2e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --bf16 True \
  --tf32 True \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --report_to "tensorboard"
```

**Note**: Adjust paths and hyperparameters to match your existing training setup.

## Inference

### Architecture

```python
# Modular layer stack - each layer is independent
base_model = "Qwen/Qwen3-VL-8B-Instruct"
router_lora = "checkpoints/router-classifier"
expert_loras = {
    "calendar": "checkpoints/calendar-tasks",
    "claim-window": "checkpoints/claim-window",
    "provider-select": "checkpoints/provider-select",
    # Future: "chandra-ocr": "checkpoints/chandra-ocr"
}
```

### Two-Stage Inference

```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

# Load base model
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

# Stage 1: Load router LoRA and classify
model = PeftModel.from_pretrained(model, "checkpoints/router-classifier")
model.eval()

routing_prompt = [
    {"role": "system", "content": "You are a routing classifier..."},
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Which adapter should handle this screenshot?"}
    ]}
]

inputs = processor(text=routing_prompt, images=[image], return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=20)
adapter_name = processor.decode(output[0], skip_special_tokens=True).strip()
# Returns: "calendar" or "claim-window" or "provider-select"

# Stage 2: Unload router, load expert, run task
model = model.unload()  # Remove router LoRA
model = PeftModel.from_pretrained(model, f"checkpoints/{adapter_name}")

# Now run the actual task with the expert adapter
task_prompt = [{"role": "user", "content": [...]}]  # Your task prompt
inputs = processor(text=task_prompt, images=[image], return_tensors="pt").to(model.device)
output = model.generate(**inputs)
result = processor.decode(output[0])
```

### Optimized Single-Load Inference

```python
# For production: keep base model loaded, swap adapters
from peft import set_adapter

# Load base once
model = Qwen3VLForConditionalGeneration.from_pretrained(...)

# Load all adapters
model = PeftModel.from_pretrained(model, "checkpoints/router-classifier", adapter_name="router")
model.load_adapter("checkpoints/calendar-tasks", adapter_name="calendar")
model.load_adapter("checkpoints/claim-window", adapter_name="claim-window")
model.load_adapter("checkpoints/provider-select", adapter_name="provider-select")

# Classify with router
model.set_adapter("router")
adapter_name = classify(image)  # Returns adapter name

# Execute with expert
model.set_adapter(adapter_name)
result = execute_task(image, prompt)
```

## Adding New Experts (e.g., chandra-ocr)

### Step 1: Regenerate routing dataset with 4 classes

```bash
./scripts/generate_dataset.sh \
  --train-images-calendar 27 \
  --train-images-claim-window 3000 \
  --train-images-provider-select 3000 \
  --train-images-chandra-ocr 3000 \  # NEW
  --eval-images 7 \
  --train-val-split 0.98 \
  --calendar-dataset ... \
  --claim-window-dataset ... \
  --provider-select-dataset ... \
  --chandra-ocr-dataset ...  # NEW
```

### Step 2: Convert and retrain router

```bash
python3 scripts/convert_routing_to_classification.py datasets/routing_NEW

# Retrain router (same command as before, just new dataset)
python train_qwen.py --data_path datasets/routing_NEW/train-classification.jsonl ...
```

**Result**: Router now outputs 4 classes: `"calendar" | "claim-window" | "provider-select" | "chandra-ocr"`

### Step 3: Update inference

```python
expert_loras = {
    "calendar": "checkpoints/calendar-tasks",
    "claim-window": "checkpoints/claim-window",
    "provider-select": "checkpoints/provider-select",
    "chandra-ocr": "checkpoints/chandra-ocr",  # NEW
}

# Everything else stays the same!
```

## Comparison with Previous Approach

| Aspect | Previous (Router Head) | New (Router LoRA) |
|--------|----------------------|-------------------|
| **Code complexity** | 500+ lines custom training | 0 new lines (reuses existing) |
| **Training time** | Hours on GPU with preprocessing | Same as other LoRAs (~30 min) |
| **Precision issues** | Multiple bfloat16/float32 bugs | None (standard LoRA training) |
| **Batch handling** | Variable-length tensor issues | Standard batching works |
| **Extensibility** | Add class, retrain head | Add class, retrain LoRA |
| **Architecture** | Custom MLP head on frozen encoder | Standard LoRA adapter |
| **Inference** | Load base + router head | Load base + router LoRA |
| **Modularity** | Router coupled to base | Router is independent layer |

## Validation

### Expected Training Metrics

- **Initial loss**: ~1.1 (random 3-class classification)
- **Final loss**: ~0.1-0.3 (after 5-10 epochs)
- **Eval accuracy**: >95% (balanced dataset)

### Testing

```python
# Test router on validation set
model.set_adapter("router")

correct = 0
total = 0

for sample in validation_set:
    predicted = classify(sample["image"])
    actual = sample["conversations"][-1]["value"]  # Ground truth adapter name
    if predicted == actual:
        correct += 1
    total += 1

accuracy = correct / total
print(f"Router accuracy: {accuracy:.2%}")
```

Expected: >95% accuracy

## Summary

**Router as LoRA** is the simplest, most maintainable approach:

1. ✅ Zero new infrastructure (reuses existing LoRA training)
2. ✅ True modularity (each layer is independent)
3. ✅ Easy to extend (just retrain router with new class)
4. ✅ No precision/batching issues (standard LoRA code handles it)
5. ✅ Matches your existing patterns (calendar, claim-window, etc.)

**Next steps:**
1. Train router LoRA with existing training script
2. Test classification accuracy on validation set
3. Integrate into inference pipeline
4. Add chandra-ocr when ready (retrain router + load 4th expert)
