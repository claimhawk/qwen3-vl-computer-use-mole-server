# Routing Data Generator

Builds a routing dataset by merging existing task datasets (e.g., calendar and claim-window) and stamping each row with an `adapter` label for Stage-1 router training.

## Usage

Run from the repo root:

```bash
python routing-generator/generate.py \
  --source label=calendar,jsonl=../generators/calendar/datasets/mike-im-day-clicks-system-prompt-8B_20251120_180854/data.jsonl,image_root=../generators/calendar/datasets \
  --source label=claim-window,jsonl=../generators/edit-claim-window/datasets/edit-claim-window-prod_20251122_234849/data.jsonl,image_root=../generators/edit-claim-window/datasets \
  --output routing-data.jsonl \
  --shuffle \
  --seed 42
```

`--source` fields (comma-separated key=val):
- `label` (required): adapter label to stamp (`calendar`, `claim-window`, etc.).
- `jsonl` (required): path to a source JSONL file.
- `image_root` (optional): base dir to resolve relative image paths. Defaults to the JSONL parent; if images are referenced with the dataset folder name, the script will also try the parent of the JSONL parent.
- `tasks` (optional): comma-separated `metadata.task_type` filter, e.g., `tasks=click_calendar_day,click_calendar_button`.
- `limit` (optional): cap number of rows taken from this source.

Flags:
- `--shuffle`: shuffle merged rows.
- `--seed N`: RNG seed for shuffling.

Output:
- JSONL where each row is the original record plus:
  - `adapter`: label provided for the source.
  - `source_dataset`: stem of the input JSONL (or `name` if provided).
  - `image_path`: resolved absolute image path (best-effort).

Notes:
- Use homogeneous sources per `label`, or filter with `tasks` when a source mixes task types.
- The script will warn if it cannot resolve an image; it still writes the row.
