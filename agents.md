# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MOLE (Mixture of LoRA Experts) Training Server - distributed training infrastructure for fine-tuning Qwen3-VL vision-language models with LoRA adapters. Trains a "router" LoRA that classifies which specialized expert adapter to activate based on screen content.

## Commands

### Code Quality (run before committing)

```bash
./scripts/pre-commit.sh        # Check staged Python files (ruff, docstrings, mypy)
./scripts/pre-commit.sh --all  # Check all tracked Python files
```

### Training Pipeline (4 stages, auto-chains)

```bash
# Full pipeline: generate -> upload -> preprocess -> train
./scripts/generate.sh

# Individual stages (each auto-chains to next unless --dry):
./scripts/generate.sh [--dry]                              # Stage 1: Generate dataset locally
./scripts/upload.sh [dataset_dir] [--dry]                  # Stage 2: Upload to Modal volume
./scripts/preprocess.sh --dataset-name <NAME> [--dry]      # Stage 3: Preprocess on Modal (CPU)
./scripts/train.sh --run-name <NAME> --dataset-name <NAME> # Stage 4: Train on Modal (GPU)

# Quick iteration (20 steps)
./scripts/train.sh --run-name <NAME> --dataset-name <NAME> --fast

# Test mode (small dataset)
./scripts/train.sh --run-name <NAME> --dataset-name <NAME> --test
```

### Evaluation & Deployment

```bash
./scripts/eval.sh --run-name <NAME> --dataset-name <NAME>  # Evaluate on held-out data
./scripts/deploy-latest.sh                                  # Deploy latest checkpoint
./scripts/tensorboard.sh deploy                            # Deploy TensorBoard
```

### Direct Tool Invocation

```bash
uvx python -m ruff check <files>              # Lint
uvx python -m mypy --config-file mypy.ini <files>  # Type check
uvx python utils/check_docstrings.py <files>  # Docstring check
```

## Architecture

### Directory Structure

- `modal/` - Modal cloud deployment functions (training, preprocessing, eval, inference)
- `scripts/` - Bash wrapper scripts for the pipeline
- `config/` - YAML/JSON configuration (dataset.yaml, loras.json)
- `generator.py` - Local dataset generation (PEP 517 inline deps, standalone)
- `routing-generator/` - Expert routing data merging/balancing
- `utils/` - Shared utilities (docstring checker)

### Data Flow

1. **Generate** (`generator.py`): Reads from `config/dataset.yaml`, merges adapter datasets from `../generators/*/datasets/`, outputs `datasets/routing_YYYYMMDD_HHMMSS/`
2. **Upload** (`upload.sh`): Pushes `.jsonl` files to Modal volume `moe-lora-data`
3. **Preprocess** (`modal/preprocess.py`): Tokenizes via Qwen processor on Modal CPU instances
4. **Train** (`modal/training.py`): LoRA fine-tuning on Modal GPU with flash-attn
5. **Eval** (`modal/eval.py`): Evaluates on held-out `eval.jsonl`
6. **Deploy** (`modal/deploy.py`): Copies checkpoint to inference volume
7. **Inference** (`modal/stacked_inference.py`): Router selects expert adapter

### Configuration

- `config/dataset.yaml`: Adapter counts, train/val splits, system prompt
- `config/loras.json`: Registered LoRA checkpoint locations and datasets
- Modal files load config from `sdk.modal_compat` with hardcoded fallbacks

## Code Standards

- Python 3.11+, type hints mandatory, mypy strict mode
- Max cyclomatic complexity: 10, max function length: 50-60 lines, max nesting: 3
- Docstrings required for all modules, public classes, and functions
- Use `uv run` or `uvx` for Python execution
- See `CODE_QUALITY.md` for full standards

## Git Commits

**DO NOT CO-AUTHOR COMMITS** - only use the GitHub user's name when committing. Do not add co-author trailers or attribute commits to AI assistants.
