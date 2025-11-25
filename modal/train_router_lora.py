"""
Train router as LoRA adapter using existing calendar training infrastructure.

This is a thin wrapper that calls the calendar LoRA training Modal app
with router-specific dataset and output paths.
"""

import modal
import sys
from pathlib import Path

# Import the calendar training Modal app
sys.path.append("/Users/michaeloneal/development/claimhawk/trainers/calendar/build")
from modal_bundle import app as calendar_app, train as calendar_train

# Create our own app that uses the same infrastructure
app = modal.App("router-lora-training")

# Re-export the calendar training function under our app
@app.function()
def train_router(
    dataset_name: str,
    epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
):
    """Train router LoRA adapter."""
    return calendar_train.remote(
        datasets=dataset_name,
        data_path=f"/data/datasets/{dataset_name}/train.jsonl",
        eval_data_path=f"/data/datasets/{dataset_name}/eval.jsonl",
        output_dir="/output/router-classifier",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=learning_rate,
        save_strategy="epoch",
        evaluation_strategy="epoch",
    )


@app.local_entrypoint()
def main(dataset_name: str):
    """Entry point for Modal CLI."""
    print(f"Starting router LoRA training for dataset: {dataset_name}")
    train_router.remote(dataset_name)
    print("Training complete!")
