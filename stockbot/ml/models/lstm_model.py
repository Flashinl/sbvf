"""
LSTM Model for Time-Series Stock Prediction

Architecture:
- Input: 60 days of OHLCV data + technical indicators
- 2 LSTM layers with dropout for regularization
- Dense output layer for price prediction
- Trained to predict 30-day forward return
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
import pickle
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed. Install with: pip install torch")


class StockSequenceDataset(Dataset):
    """Dataset for LSTM training"""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class LSTMPredictor(nn.Module):
    """
    LSTM model for stock price prediction

    Input shape: (batch_size, sequence_length, num_features)
    Output: (batch_size, 1) - predicted 30-day return %
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1
    ):
        super(LSTMPredictor, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

        # Activation
        self.relu = nn.ReLU()

    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)

        # Take output from last time step
        last_output = lstm_out[:, -1, :]

        # Fully connected layers
        out = self.dropout(last_output)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class LSTMStockModel:
    """
    Wrapper class for LSTM model training and inference
    """

    def __init__(
        self,
        sequence_length: int = 60,
        input_features: int = 10,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        device: Optional[str] = None
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")

        self.sequence_length = sequence_length
        self.input_features = input_features
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model
        self.model = LSTMPredictor(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            output_size=1
        ).to(self.device)

        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Scaler for normalization (fit during training)
        self.scaler_params = None

    def prepare_sequences(
        self,
        data: pd.DataFrame,
        target_col: str = 'return_30d',
        feature_cols: Optional[list] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare time-series sequences from DataFrame

        Args:
            data: DataFrame with columns [date, close, volume, feature1, feature2, ...]
            target_col: Column name for target variable (30-day return)
            feature_cols: List of feature column names

        Returns:
            sequences: (num_samples, sequence_length, num_features)
            targets: (num_samples,)
        """
        if feature_cols is None:
            feature_cols = ['close', 'volume', 'rsi_14', 'macd', 'sma_20',
                           'sma_50', 'bb_width', 'atr', 'obv', 'volume_ratio']

        # Normalize features
        feature_data = data[feature_cols].values
        if self.scaler_params is None:
            # Fit scaler (only on training data)
            self.scaler_params = {
                'mean': np.mean(feature_data, axis=0),
                'std': np.std(feature_data, axis=0) + 1e-8
            }

        normalized_data = (feature_data - self.scaler_params['mean']) / self.scaler_params['std']

        # Create sequences
        sequences = []
        targets = []

        for i in range(len(normalized_data) - self.sequence_length):
            seq = normalized_data[i:i + self.sequence_length]
            target = data[target_col].iloc[i + self.sequence_length]

            if not np.isnan(target):  # Skip if target is missing
                sequences.append(seq)
                targets.append(target)

        return np.array(sequences), np.array(targets)

    def train(
        self,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> Dict:
        """
        Train the LSTM model

        Returns:
            training_history: Dict with loss curves
        """
        # Prepare data
        X_train, y_train = self.prepare_sequences(train_data)
        train_dataset = StockSequenceDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if val_data is not None:
            X_val, y_val = self.prepare_sequences(val_data)
            val_dataset = StockSequenceDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'best_val_loss': float('inf'),
            'best_epoch': 0
        }

        # Early stopping
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_losses = []

            for sequences, targets in train_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device).unsqueeze(1)

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = self.criterion(outputs, targets)

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)

            # Validation phase
            if val_data is not None:
                self.model.eval()
                val_losses = []

                with torch.no_grad():
                    for sequences, targets in val_loader:
                        sequences = sequences.to(self.device)
                        targets = targets.to(self.device).unsqueeze(1)

                        outputs = self.model(sequences)
                        loss = self.criterion(outputs, targets)
                        val_losses.append(loss.item())

                avg_val_loss = np.mean(val_losses)
                history['val_loss'].append(avg_val_loss)

                # Early stopping check
                if avg_val_loss < history['best_val_loss']:
                    history['best_val_loss'] = avg_val_loss
                    history['best_epoch'] = epoch
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1

                if verbose and (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

                # Early stopping
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if verbose and (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            if verbose:
                print(f"Restored best model from epoch {history['best_epoch']+1}")

        return history

    def predict(
        self,
        data: pd.DataFrame,
        return_confidence: bool = False
    ) -> Tuple[float, Optional[float]]:
        """
        Make prediction for a single stock

        Args:
            data: DataFrame with recent history (at least sequence_length rows)
            return_confidence: If True, return confidence score

        Returns:
            prediction: Predicted 30-day return %
            confidence: Model confidence (0-1) based on prediction uncertainty
        """
        # Prepare sequence
        X, _ = self.prepare_sequences(data)

        if len(X) == 0:
            return 0.0, 0.5  # Return neutral prediction with low confidence

        # Take most recent sequence
        sequence = torch.FloatTensor(X[-1:]).to(self.device)

        # Predict
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(sequence).cpu().numpy()[0, 0]

        if return_confidence:
            # Confidence based on prediction magnitude (simple heuristic)
            # In production, use dropout at inference for uncertainty estimation
            confidence = min(0.5 + abs(prediction) * 0.1, 0.95)
            return float(prediction), float(confidence)

        return float(prediction), None

    def predict_with_uncertainty(
        self,
        data: pd.DataFrame,
        n_samples: int = 100
    ) -> Tuple[float, float, float]:
        """
        Predict with uncertainty using Monte Carlo Dropout

        Returns:
            mean_prediction: Mean of predictions
            lower_bound: 10th percentile
            upper_bound: 90th percentile
        """
        X, _ = self.prepare_sequences(data)

        if len(X) == 0:
            return 0.0, -5.0, 5.0

        sequence = torch.FloatTensor(X[-1:]).to(self.device)

        # Enable dropout at inference for uncertainty estimation
        self.model.train()
        predictions = []

        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.model(sequence).cpu().numpy()[0, 0]
                predictions.append(pred)

        self.model.eval()

        predictions = np.array(predictions)
        mean_pred = np.mean(predictions)
        lower_bound = np.percentile(predictions, 10)
        upper_bound = np.percentile(predictions, 90)

        return float(mean_pred), float(lower_bound), float(upper_bound)

    def save(self, path: str):
        """Save model to disk"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'scaler_params': self.scaler_params,
            'config': {
                'sequence_length': self.sequence_length,
                'input_features': self.input_features,
                'hidden_size': self.model.hidden_size,
                'num_layers': self.model.num_layers
            }
        }
        torch.save(save_dict, path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from disk"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler_params = checkpoint['scaler_params']
        self.model.eval()
        print(f"Model loaded from {path}")


def create_lstm_model(sequence_length: int = 60) -> LSTMStockModel:
    """Factory function to create LSTM model"""
    return LSTMStockModel(
        sequence_length=sequence_length,
        input_features=10,  # Will be adjusted based on actual features
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        learning_rate=0.001
    )
