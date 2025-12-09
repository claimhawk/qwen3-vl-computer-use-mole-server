# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MOLE (Mixture of LoRA Experts) Training Server - distributed training infrastructure for fine-tuning Qwen3-VL vision-language models with LoRA adapters. Trains a "router" LoRA that classifies which specialized expert adapter to activate based on screen content.

## Subagent Execution Model (REQUIRED)

All AI assistants **must decompose complex tasks into explicit sub-tasks** and assign each sub-task to an isolated **subagent**. This is mandatory to:

* Prevent uncontrolled context growth
* Ensure deterministic, auditable reasoning
* Preserve repository-wide clarity and focus
* Enforce separation of concerns

### Subagent Requirements

Every non-trivial request (multi-step, multi-file, or multi-decision) must:

1. **Produce a task plan**

   * Break the task into atomic sub-tasks
   * Each sub-task must correspond to a subagent
   * Each subagent must have a clear contract: inputs, outputs, constraints

2. **Run subagents independently**

   * Subagents do not share context except the explicit inputs passed to them
   * Subagents must not add new unrelated context
   * Only the orchestrator (main agent) sees the entire plan

3. **Return a composed final output**

   * The orchestrator integrates the subagents' outputs
   * No subagent should assume global repository state
   * Subagent contamination of context is forbidden

### Subagent Execution Style

Subagents must:

* Operate statelessly
* Use only their given inputs
* Produce minimal, strictly-scoped outputs
* Never rewrite or infer beyond their assigned scope

The orchestrator must:

* Keep reasoning steps isolated
* Avoid long-context carryover
* Enforce strict task boundaries

**If a task does not use subagents for its sub-tasks, it is considered invalid and must be re-executed using the subagent protocol.**

## Three-Step Implementation Protocol (MANDATORY)

All coding tasks must follow a strict three-stage workflow to ensure traceability, clarity of thought, and separation of reasoning, planning, and execution.

### 1. Research Phase → `./.claude/research/<file>`

This file contains all initial thinking, exploration, reasoning, alternatives considered, risks, constraints, and relevant contextual evaluation.

* This stage is for raw cognitive work
* No code allowed
* Subagents may be used to analyze sub-problems
* Output must be structured and comprehensive

### 2. Planning Phase → `./.claude/plans/<file>`

This file contains the **implementation plan only**.

* No code allowed
* Must list steps, modules, functions, structures, data flows, edge cases, test strategies
* Subagents must be used to design and validate individual parts of the plan
* The plan must be deterministic and complete

### 3. Implementation Progress Log → `./.claude/implementation/progress.md`

This file is your "life update" journal for the maintainer.

* Every commit-sized action must be logged
* Summaries of what was done, blockers, decisions
* Subagent invocations must be recorded as separate, timestamped entries

**Coding may only begin after these three steps are complete.**

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
