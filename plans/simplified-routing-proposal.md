# Simplified Routing Proposal: Lightweight Alternatives to Router Head Training

## Problem Statement

Current approach (router_train.py) has proven fragile:
- 500+ lines of complex training code with multiple precision issues (bfloat16/float32)
- Multi-stage pipeline: dataset generation, preprocessing, Modal GPU training
- Multiple failure modes: numerical instability, variable-length tensor handling, collate function issues
- Hours of training time on GPU for what is essentially a 3-class classification problem

## Proposed Alternatives

### Option 1: Vision Embeddings + Lightweight Classifier (RECOMMENDED)

Replace Qwen3-VL encoder with lightweight vision model embeddings + simple classifier.

**Architecture:**
```python
# Extract embeddings from pretrained vision model
embeddings = dinov2_model(image)  # 384-dim, ~5ms inference

# Train tiny classifier (logistic regression or 1-layer MLP)
adapter_id = lightweight_classifier(embeddings)  # 3 classes
```

**Benefits:**
- ✅ No complex training infrastructure (trains in seconds on CPU)
- ✅ ~50 lines of code vs 500+ lines
- ✅ No precision issues (simple float32 sklearn or PyTorch)
- ✅ Still maintains pure selection (no adapter dilution)
- ✅ Fast inference (~5-10ms vs current ~100ms+ preprocessing)
- ✅ Easy to add chandra-ocr later (just add 4th class)

**Implementation:**
1. Use DINOv2, CLIP, or SigLIP for image embeddings (all have HuggingFace models)
2. Extract embeddings for all routing dataset images (~5k images, takes ~30 seconds)
3. Train sklearn LogisticRegression or simple PyTorch Linear layer
4. Save classifier weights (< 1MB)
5. At inference: embed image → classify → load adapter

**Training code (complete):**
```python
from transformers import AutoModel, AutoProcessor
from sklearn.linear_model import LogisticRegression
import pickle

# Extract embeddings
model = AutoModel.from_pretrained("facebook/dinov2-base")
processor = AutoProcessor.from_pretrained("facebook/dinov2-base")

embeddings = []
labels = []
for sample in dataset:
    img = Image.open(sample["image"])
    inputs = processor(images=img, return_tensors="pt")
    emb = model(**inputs).last_hidden_state[:, 0, :].detach().numpy()
    embeddings.append(emb)
    labels.append(sample["adapter"])

# Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(embeddings, labels)
pickle.dump(clf, open("router.pkl", "wb"))
```

### Option 2: Prompt-Based Routing (Zero Training)

Use Qwen3-VL itself to classify the image.

**Architecture:**
```python
routing_prompt = "Which screen type is this image showing: (A) calendar navigation, (B) claim window editing, or (C) provider selection? Answer with just the letter."
response = qwen_model.generate(image, routing_prompt)
adapter_name = parse_response(response)  # "A" -> "calendar"
```

**Benefits:**
- ✅ Zero training required
- ✅ Leverages existing model understanding
- ✅ No additional infrastructure

**Tradeoffs:**
- ⚠️ Adds 1-2 seconds latency per routing decision
- ⚠️ May not be as accurate as trained classifier

### Option 3: Keep Current Approach But Simplify

If you want to stick with Qwen3-VL encoder + MLP:

**Simplifications:**
1. Remove HuggingFace Trainer, use simple PyTorch training loop (50 lines)
2. Skip preprocessing, load images on-the-fly
3. Use float32 everywhere, no mixed precision
4. Batch size = 1 always (no collate complexity)
5. Train on CPU or small GPU (doesn't need A100)

**Benefits:**
- ✅ More control over training
- ✅ Easier to debug
- ⚠️ Still ~200 lines of code
- ⚠️ Still needs GPU for reasonable speed

## Comparison Matrix

| Approach | Code Lines | Training Time | Inference Latency | Accuracy | Extensibility | Complexity |
|----------|-----------|---------------|-------------------|----------|---------------|------------|
| **Option 1: DINOv2 + LogReg** | ~50 | ~30 sec | ~5ms | High | Easy (add class) | Low |
| Option 2: Prompt-based | ~20 | 0 | ~1-2s | Medium-High | Easy | Very Low |
| Option 3: Simplified current | ~200 | ~30 min | ~100ms | High | Easy | Medium |
| Current Implementation | ~500 | ~1-2 hours | ~100ms | High | Easy | Very High |

## Recommendation

**Implement Option 1 (DINOv2 + Logistic Regression)** because:

1. **Simplicity**: Minimal code, standard ML pipeline
2. **Speed**: Fast training and inference
3. **Maintainability**: No complex infrastructure, easy to debug
4. **No dilution risk**: Pure adapter selection maintained
5. **Extensibility**: Adding chandra-ocr is trivial (retrain takes 30 seconds)
6. **Production-ready**: Stable, well-understood components

## Implementation Plan (Option 1)

### Phase 1: Extract Embeddings (1 hour)
1. Choose embedding model (DINOv2-base or CLIP-ViT-B/16)
2. Write embedding extraction script
3. Process all routing dataset images
4. Save embeddings + labels to disk

### Phase 2: Train Classifier (30 minutes)
1. Load embeddings + labels
2. Split train/val
3. Train LogisticRegression or simple MLP
4. Evaluate accuracy
5. Save classifier weights

### Phase 3: Integration (2 hours)
1. Add classifier to inference code
2. Load adapter based on classification
3. Test end-to-end
4. Compare accuracy with manual labels

### Phase 4: Deploy (1 hour)
1. Package classifier weights
2. Update Modal serving code
3. Add fallback to manual selection
4. Monitor accuracy

**Total estimated time: ~4 hours** (vs current ~weeks of debugging)

## Code Diff Preview

**Current:** 500+ lines across router_train.py, router_preprocess.py, generate_routing_data.py

**New:** Single file, ~100 lines total:

```python
# modal/router_classify.py
import modal
from sklearn.linear_model import LogisticRegression
import pickle

app = modal.App("router-classifier")
volume = modal.Volume.from_name("moe-lora-data")

@app.function(volumes={"/data": volume})
def train_router(dataset_dir: str):
    # Extract embeddings
    embeddings, labels = extract_embeddings(dataset_dir)

    # Train classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(embeddings, labels)

    # Save
    with open("/data/router.pkl", "wb") as f:
        pickle.dump(clf, f)

    # Evaluate
    accuracy = clf.score(val_embeddings, val_labels)
    print(f"Validation accuracy: {accuracy:.3f}")

@app.function(gpu="T4")
def extract_embeddings(dataset_dir: str):
    model = AutoModel.from_pretrained("facebook/dinov2-base")
    processor = AutoProcessor.from_pretrained("facebook/dinov2-base")
    # ... extraction logic
```

## Migration Path

1. **No immediate action needed**: Let current training complete to establish baseline accuracy
2. **Parallel implementation**: Build Option 1 alongside existing code
3. **A/B test**: Compare accuracy of both approaches
4. **Gradual migration**: Switch to new approach once validated
5. **Deprecate old code**: Remove router_train.py once new approach proven

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Embeddings not discriminative enough | Try multiple models (DINOv2, CLIP, SigLIP); use ensemble |
| Accuracy lower than current approach | Validate against current baseline; fine-tune embedding model if needed |
| Inference slower than expected | Precompute embeddings where possible; use smaller model (DINOv2-small) |
| Model updates require retraining | Training is 30 seconds, not a blocker |

## Success Metrics

- ✅ Training completes in < 5 minutes (vs hours)
- ✅ Inference latency < 20ms (vs 100ms+)
- ✅ Accuracy > 95% on validation set
- ✅ Code < 150 lines (vs 500+)
- ✅ Zero precision/numerical issues
- ✅ Works with batch_size > 1

## Next Steps

1. **Decision**: Choose which option to pursue
2. **If Option 1**: Prototype embedding extraction this week
3. **Validation**: Compare accuracy with current approach baseline
4. **Production**: Deploy if accuracy acceptable (>90%)
