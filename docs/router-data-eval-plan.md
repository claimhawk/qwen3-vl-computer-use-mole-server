# Router Data & Eval Plan

Goal: build routing train/eval splits from existing LoRA datasets on Modal (calendar + claim-window) and wire an evaluation path for router checkpoints.

## Data Sources (Modal volume: `claimhawk-training-data`)
- **Calendar**: `mike-im-day-clicks-system-prompt-8B_20251120_180854/`
  - Splits: `train.jsonl`, `val.jsonl`, `held_out.jsonl` (raw); preprocessed tensors under `preprocessed/.../train|val`.
- **Claim window**: `edit-claim-window-prod_20251122_234849/`
  - Splits: `train.jsonl`, `test.jsonl` (raw); preprocessed tensors under `preprocessed/.../train|val|test` for other runs.
- Other claim variants: `edit-claim-window-prod_20251123_*`, `select-provider-dropdown_*` as needed (similar schema).

## Routing Splits (JSONL)
- `routing-train.jsonl`:
  - Calendar: calendar `train.jsonl`.
  - Claim: claim `train.jsonl` (and any other claim train splits).
  - Label each row with `adapter` = `calendar` or `claim-window`.
- `routing-eval.jsonl` (held-out for accuracy):
  - Calendar: `val.jsonl` + `held_out.jsonl`.
  - Claim: `test.jsonl` (and other claim held-out/test splits).
  - Same `adapter` labels.
- Keep original fields intact (`id`, `image`, `conversations`, `metadata`), add:
  - `adapter`: routing label.
  - `source_dataset`: dataset name.

## Labeling Rules
- Calendar: dataset prefix `mike-im-day-clicks-system-prompt-8B...` OR `metadata.task_type` in calendar tasks (`click_calendar_button`, `click_calendar_day`).
- Claim-window: dataset prefix `edit-claim-window...` OR `metadata.task_type` in claim UI tasks (`scroll-grid`, `provider-dropdown`, etc.).

## Generator Updates
- Extend `routing-generator` to:
  - Accept multiple `--source` per split and flags for split type (`train` vs `eval`).
  - Write two outputs: `routing-train.jsonl`, `routing-eval.jsonl`.
  - Support reading directly from mounted Modal volume paths when run inside Modal.

## Modal Helper
- Add a small Modal function/CLI to:
  - Mount `claimhawk-training-data`.
  - Read the raw JSONLs for the specified splits.
  - Produce `routing-train.jsonl` and `routing-eval.jsonl` into `/data` (or another volume).
  - Optionally shuffle and report counts per adapter.

## Router Eval
- Add `mode=eval` to `modal/router_train.py`:
  - Load saved router head + tokenizer.
  - Run classification over `routing-eval.jsonl`.
  - Report accuracy, confusion matrix, gate entropy.
  - Optionally save metrics to `/output/router-eval.json`.

## Defaults / Paths
- Volume for data: `claimhawk-training-data`.
- Output routing files: `/data/routing-train.jsonl`, `/data/routing-eval.jsonl` (or configurable).
- LoRA checkpoints: `claimhawk-checkpoints:/calendar-tasks`, `claimhawk-checkpoints:/claim-window` (env overrides `LORA_CALENDAR`, `LORA_CLAIM`).

## Next Steps
1) Update `routing-generator` to emit train/eval outputs with adapter labels.
2) Add Modal data-extract function to generate routing train/eval from volume data.
3) Add router eval mode to `modal/router_train.py` to score held-out routing accuracy.
