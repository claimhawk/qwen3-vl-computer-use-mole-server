# MoLE Paper Notes (arXiv:2404.13628)

- Paper: “Mixture of LoRA Experts (MoLE)” — proposes a learned gate to compose multiple trained LoRAs per layer. Assets: see `references/mole-paper/` for PDFs/figures/LaTeX.
- Core motivation:
  - Observation 1: Direct or normalized linear composition of ≥3 LoRAs harms base generative ability; normalization preserves base but erodes individual LoRA traits.
  - Observation 2: Different LoRA layers encode different attributes; selectively activating layer ranges changes behavior (style, dataset-specific gains).
- Method sketch:
  - Treat each trained LoRA as an expert at every layer.
  - For a transformer block, compute outputs from each LoRA, concatenate, normalize, flatten, and score with a learnable parameter vector → logits → softmax with learnable temperature τ.
  - Fuse by weighted sum of LoRA outputs and add to frozen base block output.
  - Supports masking LoRAs at inference while re-normalizing gates.
- Practical cues relevant to our design:
  - Gating is per-layer; temperature annealing/sharpening yields near top-1 behavior while remaining differentiable.
  - Layer-level specialization matters; optional hierarchical gating (layer-wise vs matrix-wise) explored.
  - To avoid quality loss: keep base frozen; start with gates only, then optionally unfreeze LoRAs lightly.
- Visuals: `references/mole-paper/main_MoLE.pdf`, `references/mole-paper/worflow.pdf`, `references/mole-paper/main_motivation*.pdf`, gating imbalance plots in `references/mole-paper/gating_imbalance*.png`.
