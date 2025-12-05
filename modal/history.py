# Copyright (c) 2025 Tylt LLC. All rights reserved.
# Licensed for research use only. Commercial use requires a license from Tylt LLC.
# Contact: hello@claimhawk.app | See LICENSE for terms.

"""
Training History Tracking Module for MoE Router.

Persists training metrics to a dedicated volume so they survive dataset cleanups.
Each training run is recorded with:
- Validation loss (final)
- Eval accuracy (from held-out eval set)
- Training cost (USD)
- Dataset size (train/val/eval splits)
- Hyperparameters used
- Timestamps
- Per-class accuracy breakdown

Usage:
    # After training completes
    from history import save_training_record
    save_training_record(
        dataset_name="routing_20251205_115325",
        run_name="routing-overnight-001",
        metrics={...}
    )

    # Query history
    uvx modal run modal/history.py --plot
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import modal

# Volume for persistent training history (shared with lora-trainer)
HISTORY_VOLUME_NAME = "claimhawk-training-history"

app = modal.App("router-training-history")
history_volume = modal.Volume.from_name(HISTORY_VOLUME_NAME, create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "pandas>=2.0.0",
    "matplotlib>=3.7.0",
)


def generate_run_id(dataset_name: str, run_name: str) -> str:
    """Generate unique run ID from dataset and run name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{dataset_name}__{run_name}__{timestamp}"


@app.function(
    image=image,
    volumes={"/history": history_volume},
    timeout=300,
)
def save_training_record(
    dataset_name: str,
    run_name: str,
    metrics: dict[str, Any],
) -> str:
    """
    Save a router training record to the history volume.

    Args:
        dataset_name: Full dataset name (e.g., routing_20251205_115325)
        run_name: Training run name
        metrics: Dict containing:
            - val_loss: Final validation loss
            - eval_accuracy: Eval set accuracy (0-100)
            - per_class_accuracy: Dict of {adapter: accuracy}
            - cost_usd: Total training cost
            - dataset_size: {train: N, val: N, eval: N}
            - hyperparams: {lora_rank, learning_rate, ...}
            - training_time_hours: Total training time
            - total_steps: Steps completed
            - early_stopped: Whether training stopped early

    Returns:
        Run ID of saved record
    """
    run_id = generate_run_id(dataset_name, run_name)

    # Build complete record
    record = {
        "run_id": run_id,
        "type": "router",  # Distinguish from lora expert training
        "dataset_name": dataset_name,
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "val_loss": metrics.get("val_loss"),
            "eval_accuracy": metrics.get("eval_accuracy"),
            "per_class_accuracy": metrics.get("per_class_accuracy", {}),
            "cost_usd": metrics.get("cost_usd"),
            "training_time_hours": metrics.get("training_time_hours"),
            "total_steps": metrics.get("total_steps"),
            "early_stopped": metrics.get("early_stopped", False),
        },
        "dataset_size": metrics.get("dataset_size", {}),
        "hyperparams": metrics.get("hyperparams", {}),
    }

    # Router history goes in a dedicated "router" directory
    router_dir = Path("/history/router")
    router_dir.mkdir(parents=True, exist_ok=True)

    # Save individual record as JSON
    record_path = router_dir / f"{run_id}.json"
    with open(record_path, "w") as f:
        json.dump(record, f, indent=2)

    # Append to JSONL log for easy querying
    log_path = router_dir / "runs.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")

    # Commit volume
    history_volume.commit()

    print(f"📊 Router training record saved: {run_id}")
    print(f"   Dataset: {dataset_name}")
    print(f"   Val Loss: {record['metrics']['val_loss']}")
    print(f"   Eval Accuracy: {record['metrics']['eval_accuracy']:.1f}%" if record['metrics']['eval_accuracy'] else "   Eval Accuracy: N/A")
    if record['metrics']['cost_usd']:
        print(f"   Cost: ${record['metrics']['cost_usd']:.2f}")

    return run_id


@app.function(
    image=image,
    volumes={"/history": history_volume},
    timeout=300,
)
def get_router_history() -> list[dict]:
    """Get all router training records."""
    log_path = Path("/history/router/runs.jsonl")

    if not log_path.exists():
        print("No router training history found")
        return []

    records = []
    with open(log_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Sort by timestamp
    records.sort(key=lambda r: r.get("timestamp", ""))

    return records


@app.function(
    image=image,
    volumes={"/history": history_volume},
    timeout=600,
)
def plot_router_history(output_path: str = "/history/plots") -> str:
    """Generate plots for router training history."""
    import matplotlib.pyplot as plt
    import pandas as pd

    # Get history
    log_path = Path("/history/router/runs.jsonl")
    if not log_path.exists():
        return "No router training history found"

    records = []
    with open(log_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        return "No records found"

    # Convert to DataFrame
    df = pd.DataFrame([
        {
            "timestamp": r.get("timestamp"),
            "run_name": r.get("run_name"),
            "val_loss": r.get("metrics", {}).get("val_loss"),
            "eval_accuracy": r.get("metrics", {}).get("eval_accuracy"),
            "cost_usd": r.get("metrics", {}).get("cost_usd"),
            "train_size": r.get("dataset_size", {}).get("train"),
            "training_hours": r.get("metrics", {}).get("training_time_hours"),
        }
        for r in records
    ])

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("MoE Router Training History", fontsize=14, fontweight="bold")

    # Plot 1: Eval Accuracy over time
    ax1 = axes[0, 0]
    if df["eval_accuracy"].notna().any():
        ax1.plot(df["timestamp"], df["eval_accuracy"], "b-o", markersize=8)
        ax1.set_ylabel("Eval Accuracy (%)")
        ax1.set_xlabel("Date")
        ax1.set_title("Router Accuracy Over Time")
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
    else:
        ax1.text(0.5, 0.5, "No accuracy data", ha="center", va="center")

    # Plot 2: Validation Loss over time
    ax2 = axes[0, 1]
    if df["val_loss"].notna().any():
        ax2.plot(df["timestamp"], df["val_loss"], "r-o", markersize=8)
        ax2.set_ylabel("Validation Loss")
        ax2.set_xlabel("Date")
        ax2.set_title("Validation Loss Over Time")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No loss data", ha="center", va="center")

    # Plot 3: Training Cost over time
    ax3 = axes[1, 0]
    if df["cost_usd"].notna().any():
        ax3.bar(range(len(df)), df["cost_usd"], color="green", alpha=0.7)
        ax3.set_xticks(range(len(df)))
        ax3.set_xticklabels([r[:15] for r in df["run_name"]], rotation=45, ha="right")
        ax3.set_ylabel("Cost (USD)")
        ax3.set_title("Training Cost Per Run")
        ax3.grid(True, alpha=0.3, axis="y")
    else:
        ax3.text(0.5, 0.5, "No cost data", ha="center", va="center")

    # Plot 4: Per-class accuracy heatmap for latest run
    ax4 = axes[1, 1]
    latest_record = records[-1] if records else None
    if latest_record:
        per_class = latest_record.get("metrics", {}).get("per_class_accuracy", {})
        if per_class:
            adapters = list(per_class.keys())
            accuracies = [per_class[a] if per_class[a] is not None else 0 for a in adapters]
            colors = ['green' if a >= 80 else 'orange' if a >= 60 else 'red' for a in accuracies]
            ax4.barh(adapters, accuracies, color=colors, alpha=0.7)
            ax4.set_xlabel("Accuracy (%)")
            ax4.set_title(f"Per-Class Accuracy (Latest: {latest_record.get('run_name', 'unknown')[:20]})")
            ax4.set_xlim(0, 100)
            ax4.grid(True, alpha=0.3, axis="x")
        else:
            ax4.text(0.5, 0.5, "No per-class data", ha="center", va="center")
    else:
        ax4.text(0.5, 0.5, "No records", ha="center", va="center")

    plt.tight_layout()

    # Save plot
    plot_dir = Path(output_path)
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_file = plot_dir / "router_history.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close()

    # Commit volume with plot
    history_volume.commit()

    return str(plot_file)


@app.function(
    image=image,
    volumes={"/history": history_volume},
    timeout=300,
)
def print_summary():
    """Print summary of router training history."""
    log_path = Path("/history/router/runs.jsonl")

    if not log_path.exists():
        print("No router training history found.")
        return

    records = []
    with open(log_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        print("No records found.")
        return

    print(f"\n{'='*70}")
    print("MoE ROUTER TRAINING HISTORY")
    print(f"{'='*70}")
    print(f"Total runs: {len(records)}")

    # Get metrics
    accuracies = [r.get("metrics", {}).get("eval_accuracy") for r in records
                  if r.get("metrics", {}).get("eval_accuracy") is not None]
    costs = [r.get("metrics", {}).get("cost_usd") for r in records
             if r.get("metrics", {}).get("cost_usd") is not None]

    if accuracies:
        print(f"Best accuracy: {max(accuracies):.1f}%")
        print(f"Latest accuracy: {accuracies[-1]:.1f}%")

    if costs:
        print(f"Total cost: ${sum(costs):.2f}")
        print(f"Avg cost/run: ${sum(costs)/len(costs):.2f}")

    # Show recent runs
    print(f"\nRecent runs:")
    print(f"{'Date':<12} {'Run Name':<35} {'Accuracy':<10} {'Cost':<10}")
    print("-" * 70)
    for r in records[-10:]:
        ts = r.get("timestamp", "")[:10]
        acc = r.get("metrics", {}).get("eval_accuracy")
        acc_str = f"{acc:.1f}%" if acc else "N/A"
        cost = r.get("metrics", {}).get("cost_usd")
        cost_str = f"${cost:.2f}" if cost else "N/A"
        run_name = r.get('run_name', 'unknown')[:35]
        print(f"{ts:<12} {run_name:<35} {acc_str:<10} {cost_str:<10}")

    # Show per-class breakdown for latest run
    latest = records[-1]
    per_class = latest.get("metrics", {}).get("per_class_accuracy", {})
    if per_class:
        print(f"\nLatest run per-class accuracy:")
        for adapter, acc in sorted(per_class.items()):
            acc_str = f"{acc:.1f}%" if acc is not None else "N/A"
            print(f"  {adapter:<20}: {acc_str}")


@app.local_entrypoint()
def main(
    plot: bool = False,
):
    """
    Query and visualize router training history.

    Examples:
        # Show summary
        uvx modal run modal/history.py

        # Generate plots
        uvx modal run modal/history.py --plot
    """
    if plot:
        plot_file = plot_router_history.remote()
        print(f"Plot saved to: {plot_file}")
        return

    print_summary.remote()
