"""
Simple training script for Windows

This makes it easy to train without complex command-line arguments
"""

import sys
from pathlib import Path

# Check if data exists
data_path = Path('data/training_data_multihorizon.csv')
if not data_path.exists():
    print("ERROR: Training data not found!")
    print("\nPlease run data collection first:")
    print("  python collect_training_data.py")
    sys.exit(1)

# Import training module
from stockbot.ml.train_multihorizon import train_model, walk_forward_split
from stockbot.ml.data_collection import load_dataset

print("="*80)
print("MULTI-HORIZON STOCK PREDICTION - MODEL TRAINING")
print("="*80)

# Load dataset
print("\nLoading dataset...")
dataset = load_dataset(str(data_path))
print(f"Total samples: {len(dataset):,}")
print(f"Date range: {dataset['date'].min()} to {dataset['date'].max()}")

# Configuration
model_config = {
    'encoder_type': 'transformer',  # or 'lstm'
    'd_model': 128,
    'sequence_length': 60,
    'horizons': ['1week', '1month', '3month']
}

training_config = {
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'early_stopping_patience': 15,
    'classification_weight': 1.0,
    'regression_weight': 1.0
}

save_dir = 'models/multihorizon'

print("\n" + "="*80)
print("CONFIGURATION")
print("="*80)
print(f"Encoder: {model_config['encoder_type']}")
print(f"Model dimension: {model_config['d_model']}")
print(f"Sequence length: {model_config['sequence_length']} days")
print(f"Horizons: {', '.join(model_config['horizons'])}")
print(f"\nMax epochs: {training_config['epochs']}")
print(f"Batch size: {training_config['batch_size']}")
print(f"Learning rate: {training_config['learning_rate']}")
print(f"Early stopping patience: {training_config['early_stopping_patience']}")
print(f"\nSave directory: {save_dir}")
print("="*80)

# Create train/val/test splits
print("\nCreating walk-forward validation splits...")
splits = walk_forward_split(dataset, n_splits=1, test_size=0.2)
train_df, val_df, test_df = splits[0]

# Train model
print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80)
print("\nThis may take 15-45 minutes depending on your hardware...")
print("(GPU will be much faster if available)")
print()

results = train_model(
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    model_config=model_config,
    training_config=training_config,
    save_dir=save_dir
)

# Print final results
print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"\nModel saved to: {save_dir}/best_model.pth")
print(f"Best validation loss: {results['best_val_loss']:.4f}")
print(f"Test loss: {results['test_losses']['total']:.4f}")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("\n1. Make predictions:")
print("   python predict_stock.py AAPL")
print("\n2. Batch predictions:")
print("   python batch_predict.py")
print("\n3. View training history:")
print("   python visualize_training.py")
print("="*80)
