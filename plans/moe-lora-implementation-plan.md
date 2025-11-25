# Mixture-of-LoRAs Implementation Plan (Qwen3-VL-8B-Instruct)

## Goals
- Route requests (prompt + screen) to the right specialist LoRA on top of base `Qwen3-VL-8B-Instruct`.
- Support soft routing (mixture) and hard routing (top-1/top-2) at inference.
- Keep base + LoRAs intact; train a small router with minimal code surface and clear evaluation.

## Architecture
- **Base & Experts**: Load base model once; attach K LoRA adapters (existing specialists).
- **Router Features**: Concatenate early text hidden state (e.g., token 0 after layer 2–4) with pooled vision embedding (from vision tower or first cross-attn fusion); LayerNorm + projection to router MLP.
- **Router Head**: Small 2-layer MLP → logits → softmax with temperature τ; optionally per-layer heads later, but start with one global router.
- **LoRA Mixing**: For each layer with LoRA, compute all LoRA outputs `u^(k) = A_k(B_k x)`; fuse `u = Σ g_k u^(k)`; apply `y = W x + α u`. Support hard routing (argmax or Gumbel-ST) and soft routing.
- **Module Wrapper**: `MultiLoRAWithRouter` wrapping Qwen3-VL; exposes APIs for `forward(input, images, routing_mode, top_k)` and returns gate probs for logging.

## Implementation Tasks
1) **Scaffold**
   - Add config for LoRA paths, adapter names, router dims, τ, entropy/aux weights.
   - Utility to load base + multiple LoRAs (PEFT or custom modules) into a single model instance.
   - Wire optional agent-provided expert IDs/confidence into the model API for fast-path routing and as labels.
2) **Router Module**
   - Implement feature extractor hook to grab text hidden states at layer N and pooled vision features.
   - Build router MLP + softmax; temperature scheduling; optional dropout.
   - Expose `routing_mode` (`soft`, `top1`, `top2` with renormalization) and optional Gumbel-ST during training; allow agent override or prior.
3) **LoRA Fusion**
   - Extend LoRA layers to accept gating vector and mix outputs; ensure batch-wise gating (per example).
   - Efficient batching: vectorize over experts where possible; fallback to small loop if K small.
   - Cache per-forward `g(x)` and attach to outputs for metrics.
4) **Training Pipeline**
   - **Stage 1 (Supervised router)**: freeze base + LoRAs; train router as task classifier on (prompt, screen, adapter_id); loss: CE on adapter_id.
   - **Stage 2 (End-to-end routing)**: still freeze base + LoRAs; use full model with routed LoRAs; loss: task loss (e.g., action/LM CE) + entropy penalty; optional KL to Stage-1 logits.
   - **Stage 3 (Optional joint)**: unfreeze LoRAs with low LR; router higher LR; keep base frozen.
   - Data loaders balanced by adapter/task; add load-balancing regularizer if gate marginal collapses.
   - Agent labels: use agent’s expert choice as supervision for Stage 1; at inference let agent provide top-1, with router as veto/top-2 fallback when confidence is low.
5) **Evaluation & Debug**
   - Metrics: gate accuracy vs labels (Stage 1), gate entropy, marginal gate distribution, task loss, per-adapter task metrics.
   - Ablations: soft vs hard routing; top-1 vs top-2; with/without KL; with/without entropy.
   - Logging: save gate histograms, confusion matrix, examples with gates.
6) **Inference Path**
   - Default to hard top-1 once gates are sharp; allow soft mixture and top-2 for quality runs.
   - Expose CLI/serving flag to choose routing mode and τ.
   - Fallback: manual adapter override for debugging.

## Data Format (calendar vs claim-window)
- Source: `sample-data/data.jsonl`, images under `sample-data/images/`.
- Record fields: `id`, `image` (relative path), `conversations` (system/human/gpt steps with tool calls), `metadata` (e.g., `task_type`, coordinates, dates).
- Adapter label: derive `adapter` for Stage-1 routing labels:
  - `calendar` when `metadata.task_type == "click_calendar_day"` or image filename contains `screen_`/calendar shots.
  - `claim-window` when IDs/images include `edit-claim-window` or `metadata.task_type` in claim UI tasks (`scroll-grid`, `provider-dropdown`, etc.).
- Training prep: materialize `adapter` into each JSONL row and store routing labels; keep original conversations for action/text supervision in Stage 2.

## LoRA Paths (Modal volumes)
- Modal volume: `claimhawk-checkpoints`.
- Calendar LoRA: `claimhawk-checkpoints:/calendar-tasks` (copied from `.../mike-day-clicks-graduated-8b/checkpoint-40`).
- Claim-window LoRA: set alongside calendar in `claimhawk-checkpoints` (configure env `LORA_CLAIM` to actual path, e.g., `/checkpoints/claim-window`).
 - Router defaults in `modal/router_train.py` expect `/checkpoints/calendar-tasks` and `/checkpoints/claim-window`.

## Milestones
- M1: Multi-LoRA loader working; forward with manual adapter selection.
- M2: Router feature hooks implemented; router MLP produces gates (no training).
- M3: Stage-1 supervised router reaches high adapter-ID accuracy on val set.
- M4: Stage-2 end-to-end with entropy regularization; gates sharpen; task metrics match manual routing.
- M5 (opt): Joint finetune with small LR on LoRAs; confirm no collapse.
- M6: Inference modes (soft/top-1/top-2) shipped with logs + flags.

## Risks & Mitigations
- **Gate collapse to one expert**: balance batches, entropy penalty, load-balancing term, keep experts frozen early.
- **Router overfits prompt text**: include vision features; apply dropout; augment screens where possible.
- **Throughput hit from mixing K LoRAs**: prefer hard routing at inference; micro-bench fusion; limit to top-2 mixture if needed.
- **Feature hook drift**: lock hook layer index; assert shapes; add unit tests on dummy inputs.

## Next Steps
- Decide hook layers (text layer index N, vision pool location) and router dim; add to config.
- Implement `MultiLoRAWithRouter` wrapper and extend LoRA layers for gated fusion.
- Stand up Stage-1 training script + metrics; run a small balanced experiment to sanity-check gate accuracy.

## Paper Alignment (MoLE, arXiv:2404.13628)
- Motivation match: naïve linear/normalized LoRA fusion degrades either base generation or LoRA traits; layer-wise specialization matters.
- Method gap: MoLE gates using per-layer LoRA outputs; our plan gates via prompt+vision features for cheaper inference—optionally add per-layer gating later.
- Training cues to keep: softmax with temperature annealing toward top-1; start with gates only, keep base/LoRAs frozen, optionally unfreeze LoRAs lightly once routing is stable.
- Flexibility: support masking/unmasking LoRAs at inference while re-normalizing gates; mirror MoLE’s ability to prune experts without retraining.
