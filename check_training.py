"""
Simple script to check training progress
"""

import json
from pathlib import Path
import time

results_path = Path('models/multihorizon/training_results.json')
model_path = Path('models/multihorizon/best_model.pth')

print("Checking training progress...")
print("=" * 60)

if model_path.exists():
    print("[COMPLETE] Training finished!")
    print(f"Model saved: {model_path}")

    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)

        print(f"\nBest Validation Loss: {results['best_val_loss']:.4f}")
        print(f"Best Epoch: {results['best_epoch'] + 1}")
        print(f"Total Epochs: {len(results['history']['train_loss'])}")

        print("\nYou can now make predictions:")
        print("  python predict_stock.py AAPL")
        print("  python batch_predict.py")

elif results_path.exists():
    print("[IN PROGRESS] Training is running...")

    with open(results_path, 'r') as f:
        results = json.load(f)

    epochs_completed = len(results['history']['train_loss'])
    print(f"Epochs completed: {epochs_completed}")
    print(f"Current train loss: {results['history']['train_loss'][-1]:.4f}")
    print(f"Current val loss: {results['history']['val_loss'][-1]:.4f}")
    print(f"Best val loss so far: {results['best_val_loss']:.4f} (epoch {results['best_epoch'] + 1})")

else:
    print("[STARTING] Training hasn't saved results yet...")
    print("This usually takes 1-2 minutes for the first epoch.")
    print("\nRun this script again in a few minutes to check progress.")

print("=" * 60)
