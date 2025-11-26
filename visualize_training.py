"""
Visualize training results

Shows training/validation loss curves and learning rate schedule
"""

from pathlib import Path
import sys

results_path = Path('models/multihorizon/training_results.json')

if not results_path.exists():
    print("ERROR: Training results not found!")
    print("\nPlease train the model first:")
    print("  python train_model.py")
    sys.exit(1)

from stockbot.ml.visualization import plot_training_history

print("Generating training history visualizations...")

plot_training_history(
    str(results_path),
    save_path='models/multihorizon/training_history.png'
)

print("\n✓ Training history plot saved to: models/multihorizon/training_history.png")
