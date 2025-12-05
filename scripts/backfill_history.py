#!/usr/bin/env python3
# Copyright (c) 2025 Tylt LLC. All rights reserved.
# Licensed for research use only. Commercial use requires a license from Tylt LLC.
# Contact: hello@claimhawk.app | See LICENSE for terms.

"""
Backfill Training History for MoE Router.

Retroactively save training records for runs that completed before
the history tracking system was implemented.

Usage:
    # Backfill a specific run
    python scripts/backfill_history.py --dataset routing_20251205_115325 --run routing-overnight-001

    # Backfill all runs
    python scripts/backfill_history.py --all

    # List available runs (dry run)
    python scripts/backfill_history.py --list

    # Dry run (show what would be saved)
    python scripts/backfill_history.py --dataset routing_20251205_115325 --run routing-overnight-001 --dry-run
"""

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

MOE_VOLUME = "moe-lora-data"
HISTORY_VOLUME = "claimhawk-training-history"


def run_modal_command(cmd: list[str], capture: bool = True) -> tuple[int, str, str]:
    """Run a modal command and return (returncode, stdout, stderr)."""
    result = subprocess.run(cmd, capture_output=capture, text=True, timeout=120)
    return result.returncode, result.stdout, result.stderr


def list_runs_for_dataset(dataset_name: str) -> list[str]:
    """List all run directories for a dataset."""
    cmd = ["uvx", "modal", "volume", "ls", MOE_VOLUME, f"checkpoints/{dataset_name}/"]
    code, stdout, stderr = run_modal_command(cmd)

    if code != 0:
        return []

    runs = []
    for line in stdout.strip().split("\n"):
        line = line.strip()
        if line and not line.startswith("Directory"):
            # Extract run name from path
            name = line.rstrip("/").split("/")[-1]
            if name and name != dataset_name:
                runs.append(name)

    return runs


def list_all_datasets() -> list[str]:
    """List all dataset directories in MoE volume."""
    cmd = ["uvx", "modal", "volume", "ls", MOE_VOLUME, "checkpoints/"]
    code, stdout, stderr = run_modal_command(cmd)

    if code != 0:
        print(f"Error listing datasets: {stderr}")
        return []

    datasets = []
    for line in stdout.strip().split("\n"):
        line = line.strip()
        if line and not line.startswith("Directory"):
            name = line.rstrip("/").split("/")[-1]
            if name:
                datasets.append(name)

    return datasets


def get_trainer_state(dataset_name: str, run_name: str, temp_dir: Path) -> dict | None:
    """Download and parse trainer_state.json for a run."""
    output_file = temp_dir / "trainer_state.json"

    # Try final/ first, then root
    paths_to_try = [
        f"checkpoints/{dataset_name}/{run_name}/final/trainer_state.json",
        f"checkpoints/{dataset_name}/{run_name}/trainer_state.json",
    ]

    for remote_path in paths_to_try:
        cmd = ["uvx", "modal", "volume", "get", MOE_VOLUME, remote_path, str(output_file)]
        code, stdout, stderr = run_modal_command(cmd)

        if code == 0 and output_file.exists():
            try:
                with open(output_file) as f:
                    return json.load(f)
            except json.JSONDecodeError:
                continue
            finally:
                output_file.unlink(missing_ok=True)

    return None


def get_eval_results(dataset_name: str, run_name: str, temp_dir: Path) -> dict | None:
    """Download and parse eval-report.json for a run."""
    output_file = temp_dir / "eval_report.json"

    # MoE router stores eval results in dataset folder
    base_dataset = dataset_name.replace("datasets/", "")
    paths_to_try = [
        f"datasets/{base_dataset}/eval-report.json",
        f"data/{dataset_name}/eval-report.json",
        f"checkpoints/{dataset_name}/{run_name}/eval-report.json",
    ]

    for remote_path in paths_to_try:
        cmd = ["uvx", "modal", "volume", "get", MOE_VOLUME, remote_path, str(output_file)]
        code, stdout, stderr = run_modal_command(cmd)

        if code == 0 and output_file.exists():
            try:
                with open(output_file) as f:
                    return json.load(f)
            except json.JSONDecodeError:
                continue
            finally:
                output_file.unlink(missing_ok=True)

    return None


def get_train_results(dataset_name: str, run_name: str, temp_dir: Path) -> dict | None:
    """Download and parse train-results.json for a run."""
    output_file = temp_dir / "train_results.json"

    paths_to_try = [
        f"checkpoints/{dataset_name}/{run_name}/train-results.json",
        f"checkpoints/{dataset_name}/{run_name}/final/train-results.json",
    ]

    for remote_path in paths_to_try:
        cmd = ["uvx", "modal", "volume", "get", MOE_VOLUME, remote_path, str(output_file)]
        code, stdout, stderr = run_modal_command(cmd)

        if code == 0 and output_file.exists():
            try:
                with open(output_file) as f:
                    return json.load(f)
            except json.JSONDecodeError:
                continue
            finally:
                output_file.unlink(missing_ok=True)

    return None


def get_cost_report(dataset_name: str, run_name: str, temp_dir: Path) -> dict | None:
    """Download and parse cost_report.json for a run."""
    output_file = temp_dir / "cost_report.json"

    paths_to_try = [
        f"checkpoints/{dataset_name}/{run_name}/cost_report.json",
    ]

    for remote_path in paths_to_try:
        cmd = ["uvx", "modal", "volume", "get", MOE_VOLUME, remote_path, str(output_file)]
        code, stdout, stderr = run_modal_command(cmd)

        if code == 0 and output_file.exists():
            try:
                with open(output_file) as f:
                    return json.load(f)
            except json.JSONDecodeError:
                continue
            finally:
                output_file.unlink(missing_ok=True)

    return None


def check_history_exists(dataset_name: str, run_name: str) -> bool:
    """Check if a history record already exists for this run."""
    cmd = ["uvx", "modal", "volume", "ls", HISTORY_VOLUME, "router/"]
    code, stdout, stderr = run_modal_command(cmd)

    if code != 0:
        return False

    # Check if any file matches this dataset/run combo
    search_pattern = f"{dataset_name}__{run_name}__"
    return search_pattern in stdout


def save_history_record(dataset_name: str, run_name: str,
                        trainer_state: dict | None, eval_results: dict | None,
                        train_results: dict | None, cost_report: dict | None,
                        dry_run: bool = False) -> bool:
    """Save a training history record to the history volume."""

    # Extract metrics from trainer_state
    val_loss = None
    total_steps = None

    if trainer_state:
        # Get final validation loss from log history
        log_history = trainer_state.get("log_history", [])
        for entry in reversed(log_history):
            if "eval_loss" in entry:
                val_loss = entry["eval_loss"]
                break

        total_steps = trainer_state.get("global_step", 0)

    # Extract eval accuracy from eval results
    eval_accuracy = None
    per_class_accuracy = {}
    if eval_results:
        accuracy_data = eval_results.get("accuracy", {})
        eval_accuracy = accuracy_data.get("overall")
        per_class_accuracy = accuracy_data.get("per_class", {})

    # Extract cost from cost_report
    cost_usd = None
    training_time_hours = None
    if cost_report:
        cost_data = cost_report.get("cost", {})
        cost_usd = cost_data.get("total_usd")

        time_data = cost_report.get("time", {})
        training_time_hours = time_data.get("total_hours")

        # Also get total_steps from cost_report if not in trainer_state
        if total_steps is None:
            run_metadata = cost_report.get("run_metadata", {})
            total_steps = run_metadata.get("total_steps")

    # Extract hyperparams and dataset size from train results
    hyperparams = {}
    dataset_size = {}

    if train_results:
        training_data = train_results.get("training", {})
        hyperparams = {
            "lora_rank": training_data.get("lora_rank"),
            "lora_alpha": training_data.get("lora_alpha"),
            "learning_rate": training_data.get("learning_rate"),
            "batch_size": training_data.get("batch_size"),
            "gradient_accumulation_steps": training_data.get("gradient_accumulation_steps"),
        }

        dataset_data = train_results.get("dataset", {})
        dataset_size = {
            "train": dataset_data.get("train_samples"),
            "val": dataset_data.get("val_samples"),
            "eval": 100,  # Standard eval set size
        }

    # Build record
    timestamp = datetime.now().isoformat()
    run_id = f"{dataset_name}__{run_name}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    record = {
        "run_id": run_id,
        "type": "router",
        "dataset_name": dataset_name,
        "run_name": run_name,
        "timestamp": timestamp,
        "backfilled": True,  # Mark as retroactively added
        "metrics": {
            "val_loss": val_loss,
            "eval_accuracy": eval_accuracy,
            "per_class_accuracy": per_class_accuracy,
            "cost_usd": cost_usd,
            "training_time_hours": training_time_hours,
            "total_steps": total_steps,
            "early_stopped": False,
        },
        "dataset_size": dataset_size,
        "hyperparams": hyperparams,
    }

    if dry_run:
        print(f"\n[DRY RUN] Would save record:")
        print(json.dumps(record, indent=2))
        return True

    # Save to temporary file and upload
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(record, f, indent=2)
        temp_json = f.name

    try:
        # Upload individual record
        remote_record_path = f"router/{run_id}.json"
        cmd = ["uvx", "modal", "volume", "put", HISTORY_VOLUME, temp_json, remote_record_path]
        code, stdout, stderr = run_modal_command(cmd)

        if code != 0:
            print(f"  Error uploading record: {stderr}")
            return False

        # Append to JSONL log
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(record) + "\n")
            temp_jsonl = f.name

        # Download existing log, append, and re-upload
        existing_log = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        existing_log.close()

        cmd = ["uvx", "modal", "volume", "get", HISTORY_VOLUME, "router/runs.jsonl", existing_log.name]
        code, _, _ = run_modal_command(cmd)

        # Append new record
        with open(existing_log.name, "a") as f:
            f.write(json.dumps(record) + "\n")

        # Upload updated log
        cmd = ["uvx", "modal", "volume", "put", HISTORY_VOLUME, existing_log.name, "router/runs.jsonl"]
        code, stdout, stderr = run_modal_command(cmd)

        Path(existing_log.name).unlink(missing_ok=True)
        Path(temp_jsonl).unlink(missing_ok=True)

        if code != 0:
            print(f"  Error updating log: {stderr}")
            return False

        return True

    finally:
        Path(temp_json).unlink(missing_ok=True)


def backfill_run(dataset_name: str, run_name: str, dry_run: bool = False) -> bool:
    """Backfill history for a single run."""
    print(f"\nProcessing: {dataset_name} / {run_name}")

    # Check if already exists
    if check_history_exists(dataset_name, run_name):
        print(f"  Skipping: History record already exists")
        return True

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Fetch available data
        print(f"  Fetching trainer_state.json...", end=" ", flush=True)
        trainer_state = get_trainer_state(dataset_name, run_name, temp_path)
        print("found" if trainer_state else "not found")

        print(f"  Fetching eval-report.json...", end=" ", flush=True)
        eval_results = get_eval_results(dataset_name, run_name, temp_path)
        print("found" if eval_results else "not found")

        print(f"  Fetching train-results.json...", end=" ", flush=True)
        train_results = get_train_results(dataset_name, run_name, temp_path)
        print("found" if train_results else "not found")

        print(f"  Fetching cost_report.json...", end=" ", flush=True)
        cost_report = get_cost_report(dataset_name, run_name, temp_path)
        print("found" if cost_report else "not found")

        if not any([trainer_state, eval_results, train_results]):
            print(f"  Skipping: No training artifacts found (run may still be in progress)")
            return False

        # Save record
        print(f"  Saving history record...", end=" ", flush=True)
        success = save_history_record(
            dataset_name, run_name,
            trainer_state, eval_results, train_results, cost_report,
            dry_run=dry_run
        )

        if success:
            print("done" if not dry_run else "")
        else:
            print("failed")

        return success


def main():
    parser = argparse.ArgumentParser(description="Backfill router training history")
    parser.add_argument("--dataset", "-d", help="Dataset name")
    parser.add_argument("--run", "-r", help="Run name")
    parser.add_argument("--all", "-a", action="store_true", help="Backfill all runs")
    parser.add_argument("--list", "-l", action="store_true", help="List available runs (dry run)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be saved without saving")
    args = parser.parse_args()

    if args.list:
        # List mode
        print("Scanning MoE volume for datasets...")
        datasets = list_all_datasets()

        for ds in sorted(datasets):
            runs = list_runs_for_dataset(ds)
            if runs:
                print(f"\n{ds}:")
                for run in runs:
                    print(f"  - {run}")
        return

    if args.dataset and args.run:
        # Single run mode
        success = backfill_run(args.dataset, args.run, dry_run=args.dry_run)
        sys.exit(0 if success else 1)

    if args.all:
        # Backfill all runs
        print("Scanning MoE volume for datasets...")
        datasets = list_all_datasets()

        total = 0
        success = 0

        for ds in sorted(datasets):
            runs = list_runs_for_dataset(ds)
            for run in runs:
                total += 1
                if backfill_run(ds, run, dry_run=args.dry_run):
                    success += 1

        print(f"\n{'='*60}")
        print(f"Backfill complete: {success}/{total} runs processed")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
