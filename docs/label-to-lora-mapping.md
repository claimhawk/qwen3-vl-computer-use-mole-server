# Label-to-LoRA Mapping: How Routing Works

## The Complete Flow

Here's how a prediction label gets mapped to the correct LoRA weights:

### 1. Training Phase: Creating the Mapping

```python
# During dataset generation (scripts/generate_routing_data.py)
ADAPTER_LABELS = {
    "calendar": 0,
    "claim-window": 1,
    "provider-select": 2,
}

# Each sample gets labeled:
{
    "conversations": [...],
    "image": "...",
    "label": 0,           # Integer label
    "adapter": "calendar" # String name
}
```

### 2. Training Phase: Saving the Mapping

**ISSUE IDENTIFIED**: The current training code needs to save the adapter names!

The router checkpoint needs to include:
```python
# modal/router_train.py:555
torch.save({
    "state_dict": final_state_dict,      # Router head weights
    "adapters": ["calendar", "claim-window", "provider-select"],  # THIS IS THE KEY MAPPING
    "base_model_id": base_model_id,
    "best_loss": best_loss,
    "best_acc": best_acc,
}, output_path)
```

**The `adapters` list is critical**: It maps label indices to adapter names:
- Index 0 → "calendar"
- Index 1 → "claim-window"
- Index 2 → "provider-select"

### 3. Inference Phase: Using the Mapping

**Step 3.1: Load Router and Get Adapter Names**
```python
# modal/router_train.py:570-581
def load_router_head(path, hidden_size, device):
    checkpoint = torch.load(path)
    adapters = checkpoint["adapters"]  # Load the mapping: ["calendar", "claim-window", "provider-select"]
    head = build_router_head(hidden_size, num_labels=len(adapters))
    head.load_state_dict(checkpoint["state_dict"])
    return head, adapters  # Returns both the model AND the mapping
```

**Step 3.2: Get Prediction from Router**
```python
# modal/router_train.py:621-629 (route_and_generate function)
# 1. Encode the image+text through Qwen3-VL
outputs = encoder(input_ids, attention_mask, pixel_values, image_grid_thw, output_hidden_states=True)
hidden = outputs.hidden_states[-1][:, 0, :]  # Extract hidden state

# 2. Router head predicts a label index
head, adapters = load_router_head(router_path, ...)  # adapters = ["calendar", "claim-window", "provider-select"]
logits = head(hidden)                                 # logits shape: [batch, 3]
pred_idx = int(logits.argmax(dim=-1).item())        # pred_idx = 0, 1, or 2

# 3. Map the index to adapter name using the saved list
adapter_name = adapters[pred_idx]                    # e.g., adapters[0] = "calendar"
```

**Step 3.3: Load and Use the Selected LoRA**
```python
# modal/router_train.py:631-639
# Load all LoRAs into a multi-adapter model
lora_paths = {
    "calendar": "/checkpoints/calendar-tasks",
    "claim-window": "/checkpoints/claim-window",
    "provider-select": "/checkpoints/provider-select",
}
lora_model = load_base_with_loras(base_model_id, lora_paths)

# Switch to the selected adapter (using the string name)
lora_model.set_adapter(adapter_name)  # e.g., set_adapter("calendar")

# Generate with the selected LoRA
output_ids = lora_model.generate(**inputs)
```

## The Critical Mapping Structure

```
Training Dataset           Router Output          LoRA Selection
===============           ==============          ==============
label: 0          →       pred_idx: 0     →      adapters[0] = "calendar"          → /checkpoints/calendar-tasks
label: 1          →       pred_idx: 1     →      adapters[1] = "claim-window"      → /checkpoints/claim-window
label: 2          →       pred_idx: 2     →      adapters[2] = "provider-select"   → /checkpoints/provider-select
```

## Current Issue

The refactored `train_router()` function now takes `num_labels` instead of `adapters`, but we still need to know which adapter names correspond to which labels!

**Problem**: We need to derive or pass the adapter names list to save in the checkpoint.

**Solution Options**:

### Option 1: Read from Preprocessed Metadata
The preprocessing saves metadata with class names:
```python
# modal/router_preprocess.py:metadata
{
    "num_classes": 3,
    "class_names": ["calendar", "claim-window", "provider-select"]
}
```

We can load this during training and save it in the router checkpoint.

### Option 2: Pass as Parameter
Add `--adapter-names` parameter to the training function:
```bash
modal run modal/router_train.py::run \
    --preprocessed-dir /data/preprocessed/routing_xxx \
    --adapter-names calendar claim-window provider-select
```

### Option 3: Standard Ordering
Since we always have the same three adapters, hardcode them in order:
```python
STANDARD_ADAPTER_ORDER = ["calendar", "claim-window", "provider-select"]
```

**Recommended**: Option 1 (read from preprocessed metadata) - most robust and automatic.

## Example Inference Flow

```python
# User prompt: "Click December 3 in the calendar"
# With image showing a calendar interface

# 1. Router processes image+text
hidden_states = encoder(image, text)
logits = router_head(hidden_states)  # logits = [5.2, -1.3, 0.8]  (calendar has highest score)

# 2. Get prediction
pred_idx = argmax(logits) = 0

# 3. Map to adapter name
adapter_name = adapters[0] = "calendar"

# 4. Load LoRA and generate
lora_model.set_adapter("calendar")  # Activates calendar-specific weights
response = lora_model.generate(image, text)  # Uses calendar LoRA to generate action
```

## Summary

The label-to-LoRA mapping works through a simple list index:
1. **Training**: Labels are integers (0, 1, 2) with corresponding adapter names
2. **Saving**: The adapter names list is saved in the router checkpoint: `["calendar", "claim-window", "provider-select"]`
3. **Loading**: When loading the router, we get both the model AND the adapter names list
4. **Inference**: Router predicts an index → Use list to get name → Use name to activate LoRA

**Key insight**: The router checkpoint doesn't just save model weights—it saves the mapping from indices to adapter names, which is essential for selecting the right LoRA at inference time.
