# Implementation Plan: Inference Server Separation & Merge-Down

**Date**: 2025-12-11
**Author**: Claude (Planning Phase)
**Status**: Ready for Review
**Related Research**: `.claude/research/inference-server-separation.md`

---

## Overview

This plan details the implementation steps for:
1. Separating the inference server into a new `moe-inference-server` git submodule
2. Implementing LoRA merge-down capability for hot workflows
3. Supporting one-server-per-adapter architecture for merged models

---

## Phase 1: Git Submodule Creation & Code Separation

### Step 1.1: Create New Repository

**Actions**:
1. Create new GitHub repository: `tylt/moe-inference-server`
2. Initialize with standard structure:
   ```
   moe-inference-server/
   ├── .github/
   │   └── workflows/
   │       └── ci.yml
   ├── modal/
   │   └── __init__.py
   ├── scripts/
   ├── config/
   ├── .gitignore
   ├── pyproject.toml
   ├── README.md
   ├── CLAUDE.md
   └── CODE_QUALITY.md
   ```

**Files to Create**:

```toml
# pyproject.toml
[project]
name = "moe-inference-server"
version = "0.1.0"
description = "MoE inference server for ClaimHawk VLM adapters"
requires-python = ">=3.11"
dependencies = [
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "ruff>=0.1.0",
    "mypy>=1.0.0",
    "types-PyYAML>=6.0.0",
]

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
```

### Step 1.2: Migrate Inference Code

**Source Files to Move** (from `mole-trainer-server`):
| Source | Destination | Notes |
|--------|-------------|-------|
| `modal/stacked_inference.py` | `modal/inference.py` | Rename, update imports |
| `modal/deploy.py` | `modal/deploy.py` | Keep for adapter deployment |
| `modal/test.py` | `scripts/test_inference.py` | Move to scripts |

**Files to Keep in `mole-trainer-server`**:
- All training files (`train_*.py`, `preprocess_*.py`, `generate_*.py`)
- `eval_router.py` (training evaluation)
- `history.py` (training metrics)
- `config/dataset.yaml`, `config/loras.json` (training configs)

### Step 1.3: Update Imports and Paths

**Changes in `modal/inference.py`**:

```python
# OLD (stacked_inference.py)
app = modal.App("stacked-lora-inference")

# NEW (inference.py)
app = modal.App("moe-inference-server")
```

```python
# Update SDK imports - no changes needed, SDK is in parent repo
try:
    from sdk.modal_compat import (
        get_volume_name,
        get_router_inference_path,
        get_base_vlm,
        get_ocr_model,
    )
except ImportError:
    # Fallbacks remain the same
    ...
```

### Step 1.4: Add Submodule to ClaimHawk

**Commands**:
```bash
cd /Users/michaeloneal/development/claimhawk

# Add submodule
git submodule add git@github.com:tylt/moe-inference-server.git projects/moe-inference-server

# Update setup script
echo 'git submodule update --init projects/moe-inference-server' >> scripts/setup.sh
```

**Update `.gitmodules`**:
```ini
[submodule "projects/moe-inference-server"]
    path = projects/moe-inference-server
    url = git@github.com:tylt/moe-inference-server.git
```

### Step 1.5: Remove Inference Code from mole-trainer-server

**Files to Remove**:
- `modal/stacked_inference.py`
- `modal/test.py`

**Files to Update**:
- `CLAUDE.md` - Remove inference documentation
- `README.md` - Point to inference server for inference docs
- `scripts/` - Remove any inference-related scripts

### Step 1.6: Create Deployment Scripts

**`scripts/deploy.sh`**:
```bash
#!/bin/bash
# Deploy the inference server to Modal
set -euo pipefail

echo "Deploying MoE Inference Server..."
modal deploy modal/inference.py

echo "Deployment complete!"
echo "HTTP endpoint: https://your-workspace--moe-inference-server.modal.run/infer_web"
```

**`scripts/test.sh`**:
```bash
#!/bin/bash
# Test the inference server
set -euo pipefail

modal run modal/inference.py
```

---

## Phase 2: LoRA Merge-Down Implementation

### Step 2.1: Create Merge Module

**`modal/merge.py`**:
```python
#!/usr/bin/env python3
# Copyright (c) 2025 Tylt LLC. All rights reserved.
"""Merge LoRA weights into base model for optimized inference.

Usage:
    modal run modal/merge.py --adapter-name calendar --checkpoint latest
    modal run modal/merge.py --adapter-name claim-window --checkpoint final
"""

import json
from pathlib import Path
from datetime import datetime

import modal

# Configuration
try:
    from sdk.modal_compat import get_volume_name, get_base_vlm
    INFERENCE_VOLUME_NAME = get_volume_name("inference")
    BASE_MODEL = get_base_vlm()
except ImportError:
    INFERENCE_VOLUME_NAME = "moe-inference"
    BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

app = modal.App("moe-merge")

inference_volume = modal.Volume.from_name(
    INFERENCE_VOLUME_NAME, create_if_missing=False
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        extra_options="--index-url https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers>=4.57.0",
        "accelerate>=0.26.0",
        "peft>=0.14.0",
        "safetensors",
    )
)


@app.function(
    image=image,
    gpu="H100",
    timeout=1800,  # 30 minutes for merge
    volumes={"/inference": inference_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def merge_lora(
    adapter_name: str,
    checkpoint_path: str | None = None,
    safe_merge: bool = True,
) -> dict:
    """Merge LoRA adapter weights into base model.

    Args:
        adapter_name: Name of adapter (calendar, claim-window, etc.)
        checkpoint_path: Path to LoRA checkpoint. If None, uses deployed adapter.
        safe_merge: Check for NaN values during merge (recommended).

    Returns:
        Dict with merge status and output path.
    """
    import torch
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    inference_volume.reload()

    # Determine source path
    if checkpoint_path is None:
        lora_path = Path(f"/inference/loras/{adapter_name}/adapter")
    else:
        lora_path = Path(checkpoint_path)

    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA not found: {lora_path}")

    output_path = Path(f"/inference/merged/{adapter_name}")
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Merging {adapter_name}")
    print(f"  Source: {lora_path}")
    print(f"  Output: {output_path}")
    print(f"  Safe merge: {safe_merge}")

    # Load base model
    print("Loading base model...")
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load processor (needed for saving)
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # Load LoRA adapter
    print("Loading LoRA adapter...")
    peft_model = PeftModel.from_pretrained(base_model, str(lora_path))

    # Merge weights
    print("Merging weights...")
    merged_model = peft_model.merge_and_unload(safe_merge=safe_merge)

    # Verify merge
    print("Verifying merge...")
    total_params = sum(p.numel() for p in merged_model.parameters())
    trainable_params = sum(p.numel() for p in merged_model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # Save merged model
    print("Saving merged model...")
    merged_model.save_pretrained(str(output_path), safe_serialization=True)
    processor.save_pretrained(str(output_path))

    # Create metadata
    metadata = {
        "adapter_name": adapter_name,
        "source_lora": str(lora_path),
        "base_model": BASE_MODEL,
        "merged_at": datetime.now().isoformat(),
        "safe_merge": safe_merge,
        "total_params": total_params,
    }

    # Read source metadata if available
    source_metadata_path = lora_path / "deploy_metadata.json"
    if source_metadata_path.exists():
        with open(source_metadata_path) as f:
            source_meta = json.load(f)
            metadata["source_eval_accuracy"] = source_meta.get("eval_accuracy")
            metadata["source_dataset"] = source_meta.get("dataset_name")

    with open(output_path / "merge_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Commit volume
    inference_volume.commit()

    print(f"Merge complete: {output_path}")
    return {"status": "success", "output_path": str(output_path), **metadata}


@app.local_entrypoint()
def main(
    adapter_name: str,
    checkpoint: str | None = None,
    safe_merge: bool = True,
):
    """CLI entrypoint for merging LoRA adapters."""
    result = merge_lora.remote(
        adapter_name=adapter_name,
        checkpoint_path=checkpoint,
        safe_merge=safe_merge,
    )
    print(f"\nResult: {json.dumps(result, indent=2)}")
```

### Step 2.2: Create Merge Script

**`scripts/merge.sh`**:
```bash
#!/bin/bash
# Merge a LoRA adapter into the base model
set -euo pipefail

usage() {
    echo "Usage: $0 --adapter-name <name> [--checkpoint <path>] [--no-safe-merge]"
    echo ""
    echo "Arguments:"
    echo "  --adapter-name    Name of adapter to merge (required)"
    echo "  --checkpoint      Path to specific checkpoint (default: deployed adapter)"
    echo "  --no-safe-merge   Skip NaN checking during merge"
    exit 1
}

ADAPTER_NAME=""
CHECKPOINT=""
SAFE_MERGE="--safe-merge"

while [[ $# -gt 0 ]]; do
    case $1 in
        --adapter-name)
            ADAPTER_NAME="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="--checkpoint $2"
            shift 2
            ;;
        --no-safe-merge)
            SAFE_MERGE="--no-safe-merge"
            shift
            ;;
        *)
            usage
            ;;
    esac
done

if [[ -z "$ADAPTER_NAME" ]]; then
    usage
fi

echo "Merging adapter: $ADAPTER_NAME"
modal run modal/merge.py --adapter-name "$ADAPTER_NAME" $CHECKPOINT $SAFE_MERGE
```

### Step 2.3: Update Volume Structure

**New Directory Structure**:
```
/inference/
├── routing/adapter/              # Router LoRA (unchanged)
├── loras/                        # Dynamic LoRA adapters (unchanged)
│   ├── calendar/adapter/
│   ├── claim-window/adapter/
│   └── ...
├── merged/                       # NEW: Merged models
│   ├── calendar/
│   │   ├── config.json
│   │   ├── model.safetensors     # ~16GB
│   │   ├── tokenizer files...
│   │   └── merge_metadata.json
│   └── claim-window/
│       └── ...
└── config/
    └── deployment.yaml           # NEW: Deployment configuration
```

---

## Phase 3: One-Server-Per-Adapter Architecture

### Step 3.1: Create Merged Inference Server Module

**`modal/merged_inference.py`**:
```python
#!/usr/bin/env python3
# Copyright (c) 2025 Tylt LLC. All rights reserved.
"""Merged model inference servers - one per adapter.

Each merged server loads a single pre-merged model (LoRA baked into base),
providing faster inference without adapter switching overhead.

Usage:
    modal deploy modal/merged_inference.py
"""

from pathlib import Path
from typing import Any

import modal

# Configuration
try:
    from sdk.modal_compat import get_volume_name, get_base_vlm
    INFERENCE_VOLUME_NAME = get_volume_name("inference")
    BASE_MODEL = get_base_vlm()
except ImportError:
    INFERENCE_VOLUME_NAME = "moe-inference"
    BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

app = modal.App("moe-merged-inference")

inference_volume = modal.Volume.from_name(
    INFERENCE_VOLUME_NAME, create_if_missing=False
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        extra_options="--index-url https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers>=4.57.0",
        "accelerate>=0.26.0",
        "qwen-vl-utils",
        "Pillow>=10.0.0",
        "fastapi",
    )
)

# System prompt (same as dynamic server)
COMPUTER_USE_SYSTEM_PROMPT = """Use a mouse and keyboard to interact with a computer...
[Full prompt from stacked_inference.py]
"""


def discover_merged_models() -> dict[str, str]:
    """Discover available merged models."""
    merged_path = Path("/inference/merged")
    if not merged_path.exists():
        return {}

    discovered = {}
    for adapter_dir in merged_path.iterdir():
        if not adapter_dir.is_dir():
            continue
        config_path = adapter_dir / "config.json"
        if config_path.exists():
            discovered[adapter_dir.name] = str(adapter_dir)

    return discovered


@app.cls(
    image=image,
    gpu="H100",
    timeout=600,
    scaledown_window=300,
    volumes={"/inference": inference_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class MergedCalendarServer:
    """Merged inference server for calendar adapter."""

    adapter_name = "calendar"

    @modal.enter()
    def load_model(self):
        """Load merged model on startup."""
        import torch
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        inference_volume.reload()

        model_path = Path(f"/inference/merged/{self.adapter_name}")
        if not model_path.exists():
            raise FileNotFoundError(
                f"Merged model not found: {model_path}. Run merge.py first."
            )

        print(f"Loading merged model: {self.adapter_name}")
        print(f"  Path: {model_path}")

        self.device = torch.device("cuda")
        self.processor = AutoProcessor.from_pretrained(
            str(model_path), trust_remote_code=True
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

        print(f"Merged {self.adapter_name} server ready!")

    @modal.method()
    def infer(self, image_b64: str, prompt: str) -> dict[str, Any]:
        """Run inference on merged model."""
        import torch
        import base64
        import tempfile
        from qwen_vl_utils import process_vision_info

        # Decode image
        if image_b64.startswith("data:"):
            b64_data = image_b64.split(",", 1)[1]
        else:
            b64_data = image_b64
        image_bytes = base64.b64decode(b64_data)
        temp_path = Path(tempfile.mktemp(suffix=".jpg"))
        temp_path.write_bytes(image_bytes)

        try:
            messages = [
                {"role": "system", "content": COMPUTER_USE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{temp_path}"},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            task_output = self.processor.decode(
                outputs[0][input_len:], skip_special_tokens=True
            ).strip()

            return {
                "task_output": task_output,
                "adapter": self.adapter_name,
                "mode": "merged",
            }

        finally:
            temp_path.unlink(missing_ok=True)


@app.cls(
    image=image,
    gpu="H100",
    timeout=600,
    scaledown_window=300,
    volumes={"/inference": inference_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class MergedClaimWindowServer:
    """Merged inference server for claim-window adapter."""

    adapter_name = "claim-window"

    @modal.enter()
    def load_model(self):
        # Same as MergedCalendarServer
        ...

    @modal.method()
    def infer(self, image_b64: str, prompt: str) -> dict[str, Any]:
        # Same as MergedCalendarServer
        ...


# HTTP endpoints for each merged server
@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def calendar_infer(data: dict) -> dict[str, Any]:
    """HTTP endpoint for calendar merged inference."""
    server = MergedCalendarServer()
    return server.infer.remote(
        image_b64=data.get("image_b64", ""),
        prompt=data.get("prompt", ""),
    )


@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def claim_window_infer(data: dict) -> dict[str, Any]:
    """HTTP endpoint for claim-window merged inference."""
    server = MergedClaimWindowServer()
    return server.infer.remote(
        image_b64=data.get("image_b64", ""),
        prompt=data.get("prompt", ""),
    )
```

### Step 3.2: Create API Gateway

**`modal/gateway.py`**:
```python
#!/usr/bin/env python3
# Copyright (c) 2025 Tylt LLC. All rights reserved.
"""API Gateway - routes requests to merged or dynamic servers.

Routes based on:
1. Explicit expert param → direct to appropriate server
2. Router classification → merged (if available) or dynamic
"""

from typing import Any
import modal

app = modal.App("moe-gateway")

# Import server references
from modal.inference import MoEInferenceServer
from modal.merged_inference import MergedCalendarServer, MergedClaimWindowServer

# Configuration: which adapters have merged models
MERGED_ADAPTERS = {"calendar", "claim-window"}


@app.function()
@modal.fastapi_endpoint(method="POST")
def infer(data: dict) -> dict[str, Any]:
    """Unified inference endpoint with intelligent routing.

    POST body:
        {
            "image_b64": "...",
            "prompt": "...",
            "expert": 0,      // optional: bypass router
            "prefer_merged": true  // optional: prefer merged if available
        }
    """
    expert = data.get("expert")
    prefer_merged = data.get("prefer_merged", True)

    # If explicit expert provided
    if expert is not None:
        adapter_name = LABEL_TO_ADAPTER.get(expert)
        if adapter_name and prefer_merged and adapter_name in MERGED_ADAPTERS:
            return route_to_merged(adapter_name, data)
        else:
            return route_to_dynamic(data)

    # Use router to classify, then route
    moe_server = MoEInferenceServer()
    result = moe_server.infer.remote(
        image_b64=data.get("image_b64", ""),
        prompt=data.get("prompt", ""),
    )

    routed_adapter = result.get("routed_adapter")

    # If routed to a merged adapter, re-run on merged server
    if prefer_merged and routed_adapter in MERGED_ADAPTERS:
        merged_result = route_to_merged(routed_adapter, data)
        merged_result["routed_via"] = "dynamic_router"
        return merged_result

    return result


def route_to_merged(adapter_name: str, data: dict) -> dict[str, Any]:
    """Route request to merged server."""
    if adapter_name == "calendar":
        server = MergedCalendarServer()
    elif adapter_name == "claim-window":
        server = MergedClaimWindowServer()
    else:
        raise ValueError(f"No merged server for: {adapter_name}")

    return server.infer.remote(
        image_b64=data.get("image_b64", ""),
        prompt=data.get("prompt", ""),
    )


def route_to_dynamic(data: dict) -> dict[str, Any]:
    """Route request to dynamic MoE server."""
    server = MoEInferenceServer()
    return server.infer.remote(
        image_b64=data.get("image_b64", ""),
        prompt=data.get("prompt", ""),
        expert=data.get("expert"),
    )
```

### Step 3.3: Deployment Configuration

**`config/deployment.yaml`**:
```yaml
# Deployment configuration for inference servers
# Controls which adapters use merged vs dynamic serving

version: "1.0"

# Merged model configuration
merged:
  enabled: true
  adapters:
    - name: calendar
      gpu: H100
      min_replicas: 1
      scaledown_window: 300
      enabled: true

    - name: claim-window
      gpu: H100
      min_replicas: 1
      scaledown_window: 300
      enabled: true

# Dynamic MoE server (fallback for non-merged adapters)
dynamic:
  enabled: true
  gpu: H100
  min_replicas: 1
  scaledown_window: 300
  adapters:
    - desktop
    - appointment
    - login-window
    - chart-screen
    - ocr

# Gateway configuration
gateway:
  prefer_merged: true  # Route to merged when available
  fallback_to_dynamic: true  # Fall back to dynamic if merged unavailable
```

---

## Phase 4: Testing & Validation

### Step 4.1: Unit Tests

**`tests/test_merge.py`**:
```python
"""Tests for LoRA merge functionality."""

def test_merge_creates_valid_model():
    """Verify merge produces loadable model."""
    ...

def test_merge_preserves_weights():
    """Verify merged model produces same output as LoRA."""
    ...

def test_safe_merge_detects_nan():
    """Verify safe_merge catches NaN values."""
    ...
```

### Step 4.2: Integration Tests

**`tests/test_inference.py`**:
```python
"""Integration tests for inference servers."""

def test_dynamic_server_inference():
    """Test dynamic MoE server."""
    ...

def test_merged_server_inference():
    """Test merged model server."""
    ...

def test_merged_matches_dynamic():
    """Verify merged and dynamic produce equivalent outputs."""
    ...
```

### Step 4.3: Performance Benchmarks

**`scripts/benchmark.sh`**:
```bash
#!/bin/bash
# Benchmark merged vs dynamic inference

echo "Benchmarking inference latency..."

# Dynamic server
echo "Dynamic server:"
for i in {1..10}; do
    time curl -X POST https://...--moe-inference-server.modal.run/infer_web \
        -d '{"image_b64": "...", "prompt": "Click calendar", "expert": 0}'
done

# Merged server
echo "Merged calendar server:"
for i in {1..10}; do
    time curl -X POST https://...--moe-merged-inference.modal.run/calendar_infer \
        -d '{"image_b64": "...", "prompt": "Click calendar"}'
done
```

---

## Phase 5: Documentation & Cleanup

### Step 5.1: Update Documentation

**Files to Create/Update**:
- `moe-inference-server/README.md` - Full usage documentation
- `moe-inference-server/CLAUDE.md` - AI assistant instructions
- `mole-trainer-server/README.md` - Remove inference docs, add pointer

### Step 5.2: Update CI/CD

**`.github/workflows/ci.yml`**:
```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install ruff mypy
      - run: ruff check modal/
      - run: mypy modal/

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: pytest tests/
```

### Step 5.3: Migration Checklist

- [ ] Create `moe-inference-server` repository
- [ ] Move inference code
- [ ] Update imports and paths
- [ ] Add as git submodule
- [ ] Remove code from `mole-trainer-server`
- [ ] Implement merge module
- [ ] Create merge scripts
- [ ] Implement merged servers
- [ ] Create gateway
- [ ] Write tests
- [ ] Update documentation
- [ ] Deploy and verify

---

## File Summary

### New Files in `moe-inference-server`

| File | Purpose | Lines (est.) |
|------|---------|--------------|
| `modal/inference.py` | Dynamic MoE server (from stacked_inference.py) | 750 |
| `modal/merge.py` | LoRA merge utility | 150 |
| `modal/merged_inference.py` | Per-adapter merged servers | 300 |
| `modal/gateway.py` | API routing gateway | 100 |
| `modal/deploy.py` | Adapter deployment | 200 |
| `scripts/deploy.sh` | Deploy script | 20 |
| `scripts/merge.sh` | Merge script | 40 |
| `scripts/test.sh` | Test script | 20 |
| `config/deployment.yaml` | Deployment config | 50 |
| `pyproject.toml` | Project config | 30 |
| `README.md` | Documentation | 200 |
| `CLAUDE.md` | AI instructions | 100 |

### Removed Files from `mole-trainer-server`

| File | Reason |
|------|--------|
| `modal/stacked_inference.py` | Moved to inference server |
| `modal/test.py` | Moved to inference server |

### Modified Files in `mole-trainer-server`

| File | Changes |
|------|---------|
| `README.md` | Remove inference docs, add pointer |
| `CLAUDE.md` | Remove inference instructions |

---

## Dependencies

### `moe-inference-server` Dependencies

**Runtime**:
- torch==2.4.0
- transformers>=4.57.0
- peft>=0.14.0
- accelerate>=0.26.0
- qwen-vl-utils
- chandra-ocr
- fastapi
- Pillow>=10.0.0
- pyyaml>=6.0

**Dev**:
- ruff>=0.1.0
- mypy>=1.0.0
- pytest>=7.0.0

### Shared (from parent repo)

- sdk/adapters.py
- sdk/modal_compat.py
- config/adapters.yaml

---

## Rollback Strategy

If issues arise:

1. **Inference server issues**: Redeploy `stacked_inference.py` from mole-trainer-server git history
2. **Merge issues**: Delete `/inference/merged/{adapter}`, fall back to dynamic
3. **Gateway issues**: Bypass gateway, call servers directly

---

## Success Criteria

1. **Separation Complete**: Inference server deploys independently
2. **Merge Working**: At least 2 adapters successfully merged
3. **Performance**: Merged servers show measurable latency improvement
4. **Parity**: Merged outputs match dynamic outputs within tolerance
5. **Documentation**: All new components documented
6. **Tests**: All tests passing

---

## Open Items for Discussion

1. **Repository hosting**: GitHub vs internal GitLab?
2. **Initial merged adapters**: calendar + claim-window confirmed?
3. **Gateway deployment**: Same Modal app or separate?
4. **Monitoring**: Datadog integration needed?
5. **Cost tracking**: Per-adapter billing labels?
