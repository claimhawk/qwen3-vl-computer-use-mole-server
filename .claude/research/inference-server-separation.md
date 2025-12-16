# Research: Inference Server Separation & LoRA Merge-Down Strategy

**Date**: 2025-12-11
**Author**: Claude (Research Phase)
**Status**: Complete

---

## 1. Executive Summary

This research investigates separating the inference server from the mole-trainer-server into its own git submodule, and implementing a "merge-down" strategy for hot workflows where LoRA weights are baked into the base model for improved inference performance.

### Key Findings

1. **Current Architecture**: Inference and training are tightly coupled in `mole-trainer-server/modal/stacked_inference.py`
2. **Separation Benefits**: Independent deployment, scaling, versioning, and clearer ownership
3. **Merge-Down Benefits**: Eliminates LoRA matmul overhead, simplifies quantization, ~10-20% faster inference
4. **Merge-Down Costs**: New checkpoint per retrain, ~16GB storage per merged model, loss of dynamic adapter switching

---

## 2. Current State Analysis

### 2.1 File Ownership in mole-trainer-server

| Category | Files | Purpose |
|----------|-------|---------|
| **Training** | `generate.py`, `generate_routing.py`, `preprocess_router.py`, `train_router.py`, `link_router_data.py`, `training.py`, `preprocess.py`, `preprocess_ocr.py`, `eval_router.py`, `history.py` | Dataset generation, preprocessing, training, evaluation |
| **Inference** | `stacked_inference.py`, `test.py` | MoE routing + task inference |
| **Shared** | `deploy.py` | Moves checkpoints from training → inference volume |
| **Config** | `config/dataset.yaml`, `config/loras.json` | Training configs (not needed for inference) |
| **SDK** | `sdk/adapters.py`, `sdk/modal_compat.py` | Shared across both (parent repo) |

### 2.2 Current Inference Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     MoEInferenceServer                          │
│                   (stacked_inference.py)                        │
├─────────────────────────────────────────────────────────────────┤
│ Models Loaded at Startup:                                       │
│   1. Qwen3-VL-8B-Instruct (base)          ~16GB VRAM           │
│   2. Router LoRA                          ~100MB                │
│   3. Task LoRAs (7x)                      ~700MB total          │
│   4. Chandra OCR                          ~2GB                  │
│   Total VRAM: ~20-25GB                                          │
├─────────────────────────────────────────────────────────────────┤
│ Inference Flow:                                                 │
│   Request → Router LoRA → Integer Label → set_adapter() → Task  │
├─────────────────────────────────────────────────────────────────┤
│ HTTP Endpoint: POST /infer_web                                  │
│   Body: {"image_b64": "...", "prompt": "...", "expert": N}      │
│   Response: {"task_output": "...", "routed_adapter": "..."}     │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Current LoRA Loading Pattern

```python
# First adapter - create PeftModel
self.task_model = PeftModel.from_pretrained(task_base, lora_path, adapter_name=name)

# Subsequent adapters - add to existing model
self.task_model.load_adapter(lora_path, adapter_name=name)

# At inference - switch adapters
self.task_model.set_adapter(routed_adapter)
```

**Pros of Current Approach**:
- Single base model in VRAM (~48GB saved vs. separate base per adapter)
- Dynamic adapter switching at runtime
- Easy to add/remove adapters without restart

**Cons of Current Approach**:
- Extra LoRA matmul per forward pass
- Cannot easily quantize (base + adapter complicates quantization)
- All adapters must be compatible with same base model version

---

## 3. Separation Analysis

### 3.1 Why Separate Inference from Training?

| Aspect | Current (Combined) | Proposed (Separated) |
|--------|-------------------|---------------------|
| **Deployment** | Single deploy affects both | Independent deploys |
| **Scaling** | GPU shared for train+infer | Dedicated inference GPUs |
| **Versioning** | Single version for all | Inference can lag training |
| **Testing** | Must test all together | Isolated inference testing |
| **Dependencies** | Training deps in inference image | Minimal inference deps |
| **Team Ownership** | Shared code ownership | Clear boundaries |
| **CI/CD** | Single pipeline | Parallel pipelines |

### 3.2 Proposed Submodule Structure

```
claimhawk/projects/
├── mole-trainer-server/          # Training only (existing, trimmed)
│   ├── modal/
│   │   ├── generate.py
│   │   ├── generate_routing.py
│   │   ├── preprocess_router.py
│   │   ├── train_router.py
│   │   ├── link_router_data.py
│   │   ├── training.py
│   │   ├── preprocess.py
│   │   ├── preprocess_ocr.py
│   │   ├── eval_router.py
│   │   └── history.py
│   ├── scripts/
│   │   ├── generate.sh
│   │   ├── train.sh
│   │   └── eval.sh
│   └── config/
│       ├── dataset.yaml
│       └── loras.json
│
└── moe-inference-server/         # NEW: Inference only
    ├── modal/
    │   ├── inference.py          # Core MoE inference server
    │   ├── merged_inference.py   # Merged model servers (new)
    │   └── deploy.py             # Deploy checkpoints
    ├── scripts/
    │   ├── deploy.sh
    │   ├── deploy-merged.sh
    │   ├── merge.sh              # Merge LoRA weights
    │   └── test.sh
    ├── config/
    │   └── inference.yaml        # Inference-specific config
    ├── pyproject.toml
    └── README.md
```

### 3.3 Shared Dependencies

Both submodules will depend on:
- **Parent SDK** (`sdk/adapters.py`, `sdk/modal_compat.py`)
- **Central config** (`config/adapters.yaml`)
- **Inference volume** (`moe-inference`) - shared read-only for inference, write for deploy

### 3.4 Git Submodule Setup

```bash
# In claimhawk root
git submodule add <repo-url> projects/moe-inference-server

# .gitmodules entry
[submodule "projects/moe-inference-server"]
    path = projects/moe-inference-server
    url = git@github.com:tylt/moe-inference-server.git
```

---

## 4. Merge-Down Strategy Analysis

### 4.1 What is "Merge-Down"?

LoRA (Low-Rank Adaptation) stores delta weights: `W' = W + BA` where B and A are low-rank matrices. "Merging" computes `W' = W + BA` once and saves the result, eliminating the BA computation at inference.

### 4.2 PEFT Merge API

```python
from peft import PeftModel

# Load base + LoRA
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
peft_model = PeftModel.from_pretrained(base_model, "/path/to/lora")

# Merge and unload
merged_model = peft_model.merge_and_unload(safe_merge=True)

# Save merged model
merged_model.save_pretrained("/path/to/merged")
```

**Key Parameters**:
- `safe_merge=True`: Checks for NaN values (recommended)
- `adapter_names`: List of adapter names to merge (default: all active)

### 4.3 Merge-Down Benefits

| Benefit | Details |
|---------|---------|
| **Performance** | No extra LoRA matmul per forward pass (~10-20% faster) |
| **Quantization** | Can apply GPTQ/AWQ/GGUF to merged model directly |
| **Simplicity** | Standard model loading, no PEFT required |
| **Deployment** | Single checkpoint, no adapter files |
| **Memory** | Slightly less VRAM (no adapter overhead) |

### 4.4 Merge-Down Costs

| Cost | Details |
|------|---------|
| **Storage** | ~16GB per merged model (vs ~100MB per LoRA) |
| **Retraining** | Must regenerate merged checkpoint on LoRA update |
| **Flexibility** | Cannot switch adapters at runtime |
| **Freshness** | Merged models lag behind latest training |
| **Base Updates** | Must re-merge if base model changes |

### 4.5 When to Use Merge-Down

**Good Candidates**:
- High-traffic adapters (calendar, claim-window)
- Stable adapters (not frequently retrained)
- Latency-critical workflows
- Production deployment targets

**Poor Candidates**:
- Experimental adapters (frequently updated)
- Low-traffic adapters
- Development/testing environments
- Adapters that need A/B testing

### 4.6 Hybrid Architecture Proposal

Run **both** merged and dynamic servers:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Production Inference                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────┐     ┌─────────────────────┐           │
│  │  MergedCalendarSvr  │     │  MergedClaimSvr     │           │
│  │  (calendar merged)  │     │  (claim-window)     │           │
│  │  ~16GB VRAM each    │     │                     │           │
│  └─────────────────────┘     └─────────────────────┘           │
│                                                                 │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              MoEInferenceServer (fallback)          │       │
│  │  Router + all dynamic LoRAs + Chandra OCR           │       │
│  │  Handles: desktop, appointment, login, chart, ocr   │       │
│  │  ~25GB VRAM                                          │       │
│  └─────────────────────────────────────────────────────┘       │
│                                                                 │
│  Routing Logic:                                                 │
│    if adapter in merged_servers: route to merged               │
│    else: route to MoEInferenceServer                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.7 Merge Automation

```bash
# scripts/merge.sh
#!/bin/bash
ADAPTER_NAME=$1
CHECKPOINT_PATH=$2
OUTPUT_PATH=$3

modal run modal/merge.py \
  --adapter-name "$ADAPTER_NAME" \
  --checkpoint-path "$CHECKPOINT_PATH" \
  --output-path "$OUTPUT_PATH" \
  --safe-merge
```

```python
# modal/merge.py
def merge_lora(
    adapter_name: str,
    checkpoint_path: str,
    output_path: str,
    safe_merge: bool = True
) -> str:
    """Merge LoRA weights into base model and save."""
    from transformers import Qwen3VLForConditionalGeneration
    from peft import PeftModel

    # Load base
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # Load adapter
    peft_model = PeftModel.from_pretrained(base, checkpoint_path)

    # Merge
    merged = peft_model.merge_and_unload(safe_merge=safe_merge)

    # Save
    merged.save_pretrained(output_path)

    return output_path
```

---

## 5. One-Server-Per-Adapter Architecture

### 5.1 Motivation

For merged models, running one server per adapter provides:
- **Isolation**: Failure in one adapter doesn't affect others
- **Scaling**: Independent scaling per workflow demand
- **Updates**: Rolling updates without full system restart
- **Monitoring**: Per-adapter metrics and alerts

### 5.2 Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Modal App: moe-inference                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    API Gateway                            │  │
│  │  Routes requests to appropriate server based on:          │  │
│  │    1. Explicit expert param → direct route                │  │
│  │    2. Router classification → merged or dynamic           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                     │
│         ┌─────────────────┼─────────────────┐                  │
│         ▼                 ▼                 ▼                  │
│  ┌────────────┐   ┌────────────┐   ┌────────────────────┐     │
│  │ Merged:    │   │ Merged:    │   │ Dynamic MoE:       │     │
│  │ calendar   │   │ claim      │   │ router + all LoRAs │     │
│  │ H100       │   │ H100       │   │ H100               │     │
│  └────────────┘   └────────────┘   └────────────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Modal Class Per Adapter

```python
# modal/merged_inference.py

def create_merged_server(adapter_name: str, merged_path: str):
    """Factory to create a merged model server class."""

    @app.cls(
        image=image,
        gpu="H100",
        timeout=600,
        scaledown_window=300,
        volumes={"/inference": inference_volume},
    )
    class MergedServer:
        adapter_name = adapter_name
        model_path = merged_path

        @modal.enter()
        def load_model(self):
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.processor = AutoProcessor.from_pretrained(BASE_MODEL)
            self.model.eval()

        @modal.method()
        def infer(self, image_b64: str, prompt: str) -> dict:
            # Standard Qwen inference (no LoRA switching)
            ...

    return MergedServer

# Create server classes for merged adapters
CalendarServer = create_merged_server("calendar", "/inference/merged/calendar")
ClaimWindowServer = create_merged_server("claim-window", "/inference/merged/claim-window")
```

### 5.4 Volume Structure with Merged Models

```
/inference/
├── routing/adapter/              # Router LoRA (always dynamic)
├── loras/                        # Dynamic LoRA adapters
│   ├── calendar/adapter/
│   ├── claim-window/adapter/
│   ├── desktop/adapter/
│   └── ...
├── merged/                       # NEW: Merged full models
│   ├── calendar/
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   └── merge_metadata.json
│   └── claim-window/
│       └── ...
└── config/
    └── deployment.json           # Which adapters are merged vs dynamic
```

### 5.5 Deployment Configuration

```yaml
# config/deployment.yaml
adapters:
  calendar:
    mode: merged           # Use merged model
    gpu: H100
    min_replicas: 1
    max_replicas: 3

  claim-window:
    mode: merged
    gpu: H100
    min_replicas: 1
    max_replicas: 2

  desktop:
    mode: dynamic          # Use MoE server with LoRA switching

  appointment:
    mode: dynamic

  login-window:
    mode: dynamic

  chart-screen:
    mode: dynamic

  ocr:
    mode: chandra          # Special OCR model

router:
  mode: dynamic            # Router always uses LoRA
```

---

## 6. Quantization Considerations

### 6.1 Quantization Options for Merged Models

| Method | Bits | VRAM | Speed | Quality | Ease |
|--------|------|------|-------|---------|------|
| **BF16** | 16 | ~16GB | Fast | Best | Easy |
| **INT8** | 8 | ~8GB | Faster | Good | Medium |
| **GPTQ** | 4 | ~4GB | Fastest | Good | Harder |
| **AWQ** | 4 | ~4GB | Fastest | Good | Medium |
| **GGUF** | 4-8 | ~4-8GB | Varies | Good | Easy |

### 6.2 Quantization Workflow

```python
# Only possible with merged models (not LoRA)
from transformers import GPTQConfig

# Load merged model
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "/inference/merged/calendar",
    torch_dtype=torch.bfloat16,
)

# Apply GPTQ quantization
quantization_config = GPTQConfig(bits=4, dataset="c4", tokenizer=tokenizer)
quantized = quantize(model, quantization_config)
quantized.save_pretrained("/inference/quantized/calendar-gptq4")
```

### 6.3 Quantization Tradeoffs

**For VLMs like Qwen3-VL**:
- Vision encoder often sensitive to quantization
- Text decoder more tolerant
- Recommend INT8 for balance of speed/quality
- GPTQ/AWQ for extreme memory savings

---

## 7. Migration Path

### 7.1 Phase 1: Separation (No Merge)

1. Create `moe-inference-server` submodule
2. Move `stacked_inference.py` → `modal/inference.py`
3. Move `deploy.py` → `modal/deploy.py`
4. Move `test.py` → `scripts/test.py`
5. Update imports and paths
6. Test independently
7. Remove from `mole-trainer-server`

### 7.2 Phase 2: Add Merge Capability

1. Add `modal/merge.py` - merge LoRA into base
2. Add `scripts/merge.sh` - merge automation
3. Add `config/deployment.yaml` - merged vs dynamic config
4. Test merged model serving

### 7.3 Phase 3: One-Server-Per-Adapter

1. Add `modal/merged_inference.py` - factory for merged servers
2. Add API gateway for routing
3. Configure per-adapter scaling
4. Monitor and optimize

### 7.4 Phase 4: Quantization (Optional)

1. Add `modal/quantize.py` - quantization pipeline
2. Test INT8 merged models
3. Evaluate GPTQ/AWQ if memory constrained

---

## 8. Risk Analysis

### 8.1 Separation Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Broken imports after separation | Medium | High | Thorough import auditing |
| SDK version mismatch | Low | Medium | Pin SDK version |
| Deploy coordination | Medium | Medium | Document deploy order |
| Volume permission issues | Low | High | Test volume access early |

### 8.2 Merge-Down Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| NaN weights during merge | Low | High | Use `safe_merge=True` |
| Quality regression | Medium | High | A/B test before promotion |
| Storage explosion | Medium | Medium | Only merge hot adapters |
| Stale merged models | High | Medium | Automate merge pipeline |

### 8.3 One-Per-Adapter Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Cost increase | High | Medium | Only for hot workflows |
| Routing complexity | Medium | Medium | Clear routing logic |
| Cold start latency | Medium | High | Keep min replicas warm |
| Inconsistent updates | Medium | Medium | Coordinated deploy |

---

## 9. Alternatives Considered

### 9.1 Keep Combined (Status Quo)

**Pros**: Simple, working, no migration effort
**Cons**: Scaling limited, tight coupling, complex image
**Decision**: Reject - separation benefits outweigh costs

### 9.2 Lambda-Style Inference (No Warm Server)

**Pros**: Simpler deployment, pay-per-use
**Cons**: 60+ second cold starts unacceptable
**Decision**: Reject - latency critical

### 9.3 Single Mega-Merged Model (All Adapters)

**Pros**: Single checkpoint, simple
**Cons**: Cannot merge multiple LoRAs cleanly, huge file
**Decision**: Reject - technically infeasible

### 9.4 vLLM with LoRA Support

**Pros**: Production-grade serving, efficient batching
**Cons**: VLM support limited, complex setup
**Decision**: Consider for Phase 4+

---

## 10. Open Questions

1. **Which adapters to merge first?** → Recommend: calendar, claim-window (highest traffic)
2. **Merge automation trigger?** → On successful eval > 95% accuracy
3. **Rollback strategy for bad merge?** → Keep 2 previous versions
4. **Quantization priority?** → Phase 4 after merge stability proven
5. **Cost allocation per-adapter servers?** → Track Modal billing by app

---

## 11. Recommendations

1. **Proceed with separation** - Clear benefits, manageable risk
2. **Start merge-down with 1-2 adapters** - Calendar and claim-window
3. **Keep dynamic server as fallback** - For low-traffic and experimental
4. **Automate merge pipeline** - On deploy, optionally merge if marked hot
5. **Defer quantization** - Focus on merge-down first, quantize later if needed

---

## 12. References

- [PEFT Model Merging Guide](https://huggingface.co/docs/peft/en/developer_guides/model_merging)
- [HuggingFace merge_and_unload Discussion](https://discuss.huggingface.co/t/help-with-merging-lora-weights-back-into-base-model/40968)
- [PEFT GitHub - Merging Issues](https://github.com/huggingface/peft/issues/1836)
- [Modal GPU Scaling](https://modal.com/docs/guide/gpu)

Sources:
- [Help with merging LoRA weights back into base model](https://discuss.huggingface.co/t/help-with-merging-lora-weights-back-into-base-model/40968)
- [Model merging - PEFT Documentation](https://huggingface.co/docs/peft/en/developer_guides/model_merging)
- [Different results when merging LORA weights](https://github.com/huggingface/peft/issues/1836)
