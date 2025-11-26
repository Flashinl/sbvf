"""
Multi-Horizon Training Pipeline with Walk-Forward Validation

Features:
- Walk-forward validation (expanding window) to prevent lookahead bias
- Class weighting to handle imbalanced BUY/SELL/HOLD distribution
- Proper temporal train/val/test splits
- Early stopping based on validation loss
- Model checkpointing
- Comprehensive logging and metrics tracking

Usage:
    python -m stockbot.ml.train_multihorizon --data-path data/training_data.csv --epochs 100
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
from sklearn.utils.class_weight import compute_class_weight

from .models.transformer_multihorizon import (
    MultiHorizonStockModel,
    MultiHorizonLoss,
    create_multihorizon_model
)
from .data_collection import load_dataset, HORIZONS


class StockSequenceDataset(Dataset):
    """Dataset for multi-horizon stock prediction"""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        sequence_length: int = 60,
        horizons: List[str] = ['1week', '1month', '3month'],
        feature_columns: Optional[List[str]] = None
    ):
        """
        Args:
            dataframe: DataFrame with features and labels
            sequence_length: Number of timesteps in each sequence
            horizons: List of prediction horizons
            feature_columns: List of feature column names (auto-detect if None)
        """
        self.df = dataframe.sort_values('date').reset_index(drop=True)
        self.sequence_length = sequence_length
        self.horizons = horizons

        # Auto-detect feature columns if not provided
        if feature_columns is None:
            exclude_cols = ['ticker', 'date'] + [
                col for col in self.df.columns
                if col.startswith('target_')
            ]
            self.feature_columns = [
                col for col in self.df.columns
                if col not in exclude_cols
            ]
        else:
            self.feature_columns = feature_columns

        self.num_features = len(self.feature_columns)

        # Compute valid indices (need sequence_length history)
        # Process each ticker separately to ensure consecutive sequences
        self.valid_indices = []
        self.ticker_dataframes = {}  # Cache ticker dataframes

        for ticker in self.df['ticker'].unique():
            # Get all rows for this ticker, sorted by date
            ticker_df = self.df[self.df['ticker'] == ticker].sort_values('date').copy()
            ticker_df = ticker_df.reset_index(drop=True)

            # Cache the ticker dataframe
            self.ticker_dataframes[ticker] = ticker_df

            # For each valid sequence position in this ticker's data
            if len(ticker_df) >= sequence_length:
                for i in range(sequence_length, len(ticker_df)):
                    # Store: (ticker, position in ticker's dataframe)
                    self.valid_indices.append((ticker, i))

        # Normalize features (fit on training data)
        self.feature_mean = None
        self.feature_std = None

    def fit_normalization(self):
        """Compute normalization parameters"""
        feature_data = self.df[self.feature_columns].values
        self.feature_mean = np.nanmean(feature_data, axis=0)
        self.feature_std = np.nanstd(feature_data, axis=0) + 1e-8

    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using fitted parameters"""
        if self.feature_mean is None:
            return features
        return (features - self.feature_mean) / self.feature_std

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Get ticker and position
        ticker, ticker_df_idx = self.valid_indices[idx]
        ticker_df = self.ticker_dataframes[ticker]

        # Get sequence of features from ticker's dataframe
        seq_start = ticker_df_idx - self.sequence_length
        seq_end = ticker_df_idx
        sequence_df = ticker_df.iloc[seq_start:seq_end]

        # Extract features
        features = sequence_df[self.feature_columns].values.astype(np.float32)
        features = np.nan_to_num(features, nan=0.0)
        features = self.normalize_features(features)

        # Extract labels for current position
        current_row = ticker_df.iloc[ticker_df_idx]

        labels_class = {}
        labels_regression = {}

        for horizon in self.horizons:
            class_label = int(current_row[f'target_class_{horizon}'])
            regression_label = float(current_row[f'target_return_{horizon}'])

            labels_class[horizon] = class_label
            labels_regression[horizon] = np.float32(regression_label)  # Ensure float32

        return {
            'features': torch.FloatTensor(features),
            'labels_class': labels_class,
            'labels_regression': labels_regression
        }


def compute_class_weights(dataframe: pd.DataFrame, horizons: List[str]) -> Dict[str, torch.Tensor]:
    """
    Compute class weights for each horizon to handle imbalanced data

    Args:
        dataframe: Training dataframe
        horizons: List of horizons

    Returns:
        Dict mapping horizon -> class weights tensor
    """
    class_weights = {}

    for horizon in horizons:
        class_col = f'target_class_{horizon}'
        classes = dataframe[class_col].values

        # Compute weights (0=SELL, 1=HOLD, 2=BUY)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=np.array([0, 1, 2]),
            y=classes
        )

        class_weights[horizon] = torch.FloatTensor(weights)

        # Print distribution
        unique, counts = np.unique(classes, return_counts=True)
        print(f"  {horizon} class distribution:")
        for cls, count in zip(unique, counts):
            cls_name = ['SELL', 'HOLD', 'BUY'][cls]
            print(f"    {cls_name}: {count} ({count/len(classes)*100:.1f}%) - weight: {weights[cls]:.3f}")

    return class_weights


def walk_forward_split(
    dataframe: pd.DataFrame,
    n_splits: int = 5,
    test_size: float = 0.2
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Create walk-forward (expanding window) splits

    Args:
        dataframe: Full dataset sorted by date
        n_splits: Number of validation splits
        test_size: Fraction of data reserved for final test

    Returns:
        List of (train_df, val_df, test_df) tuples
    """
    df_sorted = dataframe.sort_values('date').reset_index(drop=True)

    # Reserve test set (most recent data)
    test_idx = int(len(df_sorted) * (1 - test_size))
    test_df = df_sorted.iloc[test_idx:]
    trainval_df = df_sorted.iloc[:test_idx]

    # Create expanding window splits
    splits = []
    split_size = len(trainval_df) // (n_splits + 1)

    for i in range(n_splits):
        # Expanding window: train on all data up to split point
        train_end = split_size * (i + 2)
        val_start = split_size * (i + 1)
        val_end = split_size * (i + 2)

        train_df = trainval_df.iloc[:train_end]
        val_df = trainval_df.iloc[val_start:val_end]

        splits.append((train_df, val_df, test_df))

        print(f"Split {i+1}: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        print(f"  Train dates: {train_df['date'].min()} to {train_df['date'].max()}")
        print(f"  Val dates: {val_df['date'].min()} to {val_df['date'].max()}")

    return splits


def train_epoch(
    model: MultiHorizonStockModel,
    dataloader: DataLoader,
    criterion: MultiHorizonLoss,
    optimizer: torch.optim.Optimizer,
    device: str,
    horizons: List[str]
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    epoch_losses = {f'{h}_{t}': [] for h in horizons for t in ['classification', 'regression', 'total']}
    epoch_losses['total'] = []

    for batch in dataloader:
        features = batch['features'].to(device)
        labels_class = {h: batch['labels_class'][h].to(device) for h in horizons}
        labels_regression = {h: batch['labels_regression'][h].to(device) for h in horizons}

        # Forward pass
        optimizer.zero_grad()
        predictions = model(features, return_probabilities=False)

        # Prepare targets
        targets = {
            h: (labels_class[h], labels_regression[h])
            for h in horizons
        }

        # Compute loss
        loss, loss_dict = criterion(predictions, targets)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Record losses
        for key, value in loss_dict.items():
            if key in epoch_losses:
                epoch_losses[key].append(value)

    # Average losses
    avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
    return avg_losses


def evaluate(
    model: MultiHorizonStockModel,
    dataloader: DataLoader,
    criterion: MultiHorizonLoss,
    device: str,
    horizons: List[str]
) -> Tuple[Dict[str, float], Dict[str, Dict[str, List]]]:
    """
    Evaluate model

    Returns:
        avg_losses: Dictionary of average losses
        predictions_dict: Dictionary of predictions for metrics computation
    """
    model.eval()
    epoch_losses = {f'{h}_{t}': [] for h in horizons for t in ['classification', 'regression', 'total']}
    epoch_losses['total'] = []

    # Store predictions for metrics
    predictions_dict = {h: {'probs': [], 'preds': [], 'labels': [], 'regression_preds': [], 'regression_labels': []}
                        for h in horizons}

    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            labels_class = {h: batch['labels_class'][h].to(device) for h in horizons}
            labels_regression = {h: batch['labels_regression'][h].to(device) for h in horizons}

            # Forward pass
            predictions = model(features, return_probabilities=False)

            # Prepare targets
            targets = {
                h: (labels_class[h], labels_regression[h])
                for h in horizons
            }

            # Compute loss
            loss, loss_dict = criterion(predictions, targets)

            # Record losses
            for key, value in loss_dict.items():
                if key in epoch_losses:
                    epoch_losses[key].append(value)

            # Store predictions
            for h in horizons:
                class_logits, regression_pred = predictions[h]
                probs = torch.softmax(class_logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

                predictions_dict[h]['probs'].extend(probs.cpu().numpy())
                predictions_dict[h]['preds'].extend(preds.cpu().numpy())
                predictions_dict[h]['labels'].extend(labels_class[h].cpu().numpy())
                predictions_dict[h]['regression_preds'].extend(regression_pred.cpu().numpy().flatten())
                predictions_dict[h]['regression_labels'].extend(labels_regression[h].cpu().numpy())

    # Average losses
    avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}

    return avg_losses, predictions_dict


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_config: Dict,
    training_config: Dict,
    save_dir: str
) -> Dict:
    """
    Train multi-horizon model with early stopping

    Returns:
        Training history and metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    horizons = model_config['horizons']

    # Create datasets
    print("\nPreparing datasets...")
    train_dataset = StockSequenceDataset(
        train_df,
        sequence_length=model_config['sequence_length'],
        horizons=horizons
    )
    train_dataset.fit_normalization()

    val_dataset = StockSequenceDataset(
        val_df,
        sequence_length=model_config['sequence_length'],
        horizons=horizons,
        feature_columns=train_dataset.feature_columns
    )
    val_dataset.feature_mean = train_dataset.feature_mean
    val_dataset.feature_std = train_dataset.feature_std

    test_dataset = StockSequenceDataset(
        test_df,
        sequence_length=model_config['sequence_length'],
        horizons=horizons,
        feature_columns=train_dataset.feature_columns
    )
    test_dataset.feature_mean = train_dataset.feature_mean
    test_dataset.feature_std = train_dataset.feature_std

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=0
    )

    # Compute class weights
    print("\nComputing class weights...")
    class_weights_dict = compute_class_weights(train_df, horizons)

    # Create model
    print(f"\nCreating {model_config['encoder_type']} model...")
    model = create_multihorizon_model(
        input_dim=train_dataset.num_features,
        encoder_type=model_config['encoder_type'],
        d_model=model_config['d_model'],
        device=device
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create loss function with class weights
    # Average class weights across horizons
    avg_class_weights = torch.stack(list(class_weights_dict.values())).mean(dim=0).to(device)

    criterion = MultiHorizonLoss(
        horizons=horizons,
        class_weights=avg_class_weights,
        classification_weight=training_config['classification_weight'],
        regression_weight=training_config['regression_weight']
    )

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay']
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    # Training loop
    print(f"\nStarting training for {training_config['epochs']} epochs...")
    print("="*80)

    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(training_config['epochs']):
        # Train
        train_losses = train_epoch(model, train_loader, criterion, optimizer, device, horizons)

        # Validate
        val_losses, val_predictions = evaluate(model, val_loader, criterion, device, horizons)

        # Update learning rate
        scheduler.step(val_losses['total'])

        # Record history
        history['train_loss'].append(train_losses['total'])
        history['val_loss'].append(val_losses['total'])
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{training_config['epochs']}:")
            print(f"  Train Loss: {train_losses['total']:.4f}")
            print(f"  Val Loss: {val_losses['total']:.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            save_path = Path(save_dir) / 'best_model.pth'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'model_config': model_config,
                'feature_mean': train_dataset.feature_mean,
                'feature_std': train_dataset.feature_std,
                'feature_columns': train_dataset.feature_columns
            }, save_path)

            print(f"  [BEST] New best model saved (val_loss: {best_val_loss:.4f})")

        else:
            patience_counter += 1

        if patience_counter >= training_config['early_stopping_patience']:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
            break

    # Load best model for final evaluation
    print("\nLoading best model for test evaluation...")
    checkpoint = torch.load(Path(save_dir) / 'best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Final test evaluation
    test_losses, test_predictions = evaluate(model, test_loader, criterion, device, horizons)

    print("\n" + "="*80)
    print("Training Complete!")
    print(f"Best Validation Loss: {best_val_loss:.4f} (epoch {best_epoch+1})")
    print(f"Test Loss: {test_losses['total']:.4f}")
    print("="*80)

    return {
        'history': history,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'test_losses': test_losses,
        'test_predictions': test_predictions
    }


def main():
    parser = argparse.ArgumentParser(description='Train multi-horizon stock prediction model')

    # Data
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to training data CSV/parquet')
    parser.add_argument('--save-dir', type=str, default='models/multihorizon',
                       help='Directory to save models')

    # Model
    parser.add_argument('--encoder-type', choices=['transformer', 'lstm'], default='transformer',
                       help='Encoder architecture')
    parser.add_argument('--d-model', type=int, default=128,
                       help='Model dimension')
    parser.add_argument('--sequence-length', type=int, default=60,
                       help='Input sequence length')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Maximum number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='L2 regularization')
    parser.add_argument('--early-stopping-patience', type=int, default=15,
                       help='Early stopping patience')

    # Loss weights
    parser.add_argument('--classification-weight', type=float, default=1.0,
                       help='Weight for classification loss')
    parser.add_argument('--regression-weight', type=float, default=1.0,
                       help='Weight for regression loss')

    # Validation
    parser.add_argument('--n-splits', type=int, default=1,
                       help='Number of walk-forward validation splits')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Fraction of data for test set')

    args = parser.parse_args()

    print("="*80)
    print("Multi-Horizon Stock Prediction Training")
    print("="*80)

    # Load dataset
    print(f"\nLoading dataset from {args.data_path}...")
    dataset = load_dataset(args.data_path)
    print(f"Total samples: {len(dataset)}")
    print(f"Date range: {dataset['date'].min()} to {dataset['date'].max()}")

    # Create walk-forward splits
    print(f"\nCreating {args.n_splits} walk-forward splits...")
    splits = walk_forward_split(dataset, n_splits=args.n_splits, test_size=args.test_size)

    # Model config
    model_config = {
        'encoder_type': args.encoder_type,
        'd_model': args.d_model,
        'sequence_length': args.sequence_length,
        'horizons': ['1week', '1month', '3month']
    }

    # Training config
    training_config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'early_stopping_patience': args.early_stopping_patience,
        'classification_weight': args.classification_weight,
        'regression_weight': args.regression_weight
    }

    # Train on each split (or just the last one)
    for split_idx, (train_df, val_df, test_df) in enumerate(splits):
        print(f"\n{'='*80}")
        print(f"Training Split {split_idx + 1}/{len(splits)}")
        print(f"{'='*80}")

        save_dir = f"{args.save_dir}/split_{split_idx+1}"

        results = train_model(
            train_df, val_df, test_df,
            model_config, training_config,
            save_dir
        )

        # Save results
        results_path = Path(save_dir) / 'training_results.json'
        with open(results_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            serializable_results = {
                'history': results['history'],
                'best_epoch': int(results['best_epoch']),
                'best_val_loss': float(results['best_val_loss']),
                'test_losses': {k: float(v) for k, v in results['test_losses'].items()}
            }
            json.dump(serializable_results, f, indent=2)

        print(f"\nResults saved to: {results_path}")

    print("\n" + "="*80)
    print("All training complete!")
    print(f"Models saved to: {args.save_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
