"""
Multi-Horizon Transformer Model for Stock Prediction

Architecture:
- Input: Sequential OHLCV + technical indicators (sequence_length x num_features)
- Shared Feature Extractor: Transformer Encoder or LSTM
- Multi-Horizon Prediction Heads:
  * 1-week head: 3-class logits (BUY/HOLD/SELL) + regression (expected return)
  * 1-month head: 3-class logits + regression
  * 3-month head: 3-class logits + regression

Outputs per horizon:
- Classification logits: (batch_size, 3) -> BUY/HOLD/SELL probabilities after softmax
- Regression value: (batch_size, 1) -> expected return percentage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class PredictionHead(nn.Module):
    """
    Prediction head for a single horizon

    Outputs:
    - Classification logits: (batch_size, 3) for BUY/HOLD/SELL
    - Regression value: (batch_size, 1) for expected return
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2
    ):
        super().__init__()

        # Shared processing
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Classification head (BUY/HOLD/SELL)
        self.classification = nn.Linear(hidden_dim // 2, 3)

        # Regression head (expected return %)
        self.regression = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features (batch_size, input_dim)

        Returns:
            class_logits: (batch_size, 3) - logits for BUY/HOLD/SELL
            regression_value: (batch_size, 1) - predicted return %
        """
        shared_features = self.shared(x)
        class_logits = self.classification(shared_features)
        regression_value = self.regression(shared_features)
        return class_logits, regression_value


class TransformerEncoder(nn.Module):
    """Transformer encoder for sequential feature extraction"""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False  # (seq_len, batch, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.d_model = d_model

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input sequences (batch_size, seq_len, input_dim)
            src_mask: Optional attention mask

        Returns:
            Encoded features from last timestep (batch_size, d_model)
        """
        # Project input
        x = self.input_projection(x) * math.sqrt(self.d_model)  # (batch, seq, d_model)

        # Transpose for transformer: (seq, batch, d_model)
        x = x.transpose(0, 1)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Apply transformer encoder
        x = self.transformer_encoder(x, src_mask)  # (seq, batch, d_model)

        # Take output from last timestep
        x = x[-1, :, :]  # (batch, d_model)

        return x


class LSTMEncoder(nn.Module):
    """LSTM encoder for sequential feature extraction (alternative to Transformer)"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input sequences (batch_size, seq_len, input_dim)

        Returns:
            Encoded features from last timestep (batch_size, hidden_dim)
        """
        lstm_out, (hidden, cell) = self.lstm(x)
        # Take output from last timestep
        last_output = lstm_out[:, -1, :]
        return self.dropout(last_output)


class MultiHorizonStockModel(nn.Module):
    """
    Multi-horizon stock prediction model

    Shared encoder + separate prediction heads for each horizon
    """

    def __init__(
        self,
        input_dim: int,
        encoder_type: str = 'transformer',  # 'transformer' or 'lstm'
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        dropout: float = 0.2,
        horizons: List[str] = ['1week', '1month', '3month']
    ):
        """
        Args:
            input_dim: Number of input features per timestep
            encoder_type: 'transformer' or 'lstm'
            d_model: Dimension of encoder output
            nhead: Number of attention heads (transformer only)
            num_encoder_layers: Number of encoder layers
            dropout: Dropout rate
            horizons: List of horizon names
        """
        super().__init__()

        self.input_dim = input_dim
        self.encoder_type = encoder_type
        self.horizons = horizons

        # Shared feature encoder
        if encoder_type == 'transformer':
            self.encoder = TransformerEncoder(
                input_dim=input_dim,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_encoder_layers,
                dropout=dropout
            )
        elif encoder_type == 'lstm':
            self.encoder = LSTMEncoder(
                input_dim=input_dim,
                hidden_dim=d_model,
                num_layers=num_encoder_layers,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        # Prediction heads for each horizon
        self.heads = nn.ModuleDict({
            horizon: PredictionHead(
                input_dim=d_model,
                hidden_dim=128,
                dropout=dropout
            )
            for horizon in horizons
        })

    def forward(
        self,
        x: torch.Tensor,
        return_probabilities: bool = False
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass

        Args:
            x: Input sequences (batch_size, seq_len, input_dim)
            return_probabilities: If True, apply softmax to classification logits

        Returns:
            Dictionary mapping horizon -> (class_output, regression_output)
            - class_output: (batch_size, 3) - logits or probabilities
            - regression_output: (batch_size, 1) - expected return %
        """
        # Encode sequence
        encoded = self.encoder(x)  # (batch_size, d_model)

        # Generate predictions for each horizon
        outputs = {}
        for horizon in self.horizons:
            class_logits, regression_value = self.heads[horizon](encoded)

            if return_probabilities:
                class_output = F.softmax(class_logits, dim=-1)
            else:
                class_output = class_logits

            outputs[horizon] = (class_output, regression_value)

        return outputs

    def predict(
        self,
        x: torch.Tensor,
        temperature: float = 1.0
    ) -> Dict[str, Dict[str, any]]:
        """
        Make predictions with calibrated probabilities

        Args:
            x: Input sequences (batch_size, seq_len, input_dim)
            temperature: Temperature for probability calibration

        Returns:
            Dictionary mapping horizon -> prediction dict with:
            - signal: 'BUY', 'HOLD', or 'SELL'
            - probabilities: dict mapping class -> probability
            - expected_return: predicted return percentage
            - confidence: max probability
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, return_probabilities=False)

            results = {}
            class_names = ['SELL', 'HOLD', 'BUY']

            for horizon, (class_logits, regression_value) in outputs.items():
                # Apply temperature scaling
                scaled_logits = class_logits / temperature
                probs = F.softmax(scaled_logits, dim=-1)

                # Get predicted class
                pred_class_idx = torch.argmax(probs, dim=-1).item()
                signal = class_names[pred_class_idx]

                # Get probabilities for each class
                probabilities = {
                    class_name: float(probs[0, idx])
                    for idx, class_name in enumerate(class_names)
                }

                # Get expected return
                expected_return = float(regression_value[0, 0])

                # Confidence is max probability
                confidence = float(probs.max())

                results[horizon] = {
                    'signal': signal,
                    'probabilities': probabilities,
                    'expected_return': expected_return,
                    'confidence': confidence
                }

            return results


class MultiHorizonLoss(nn.Module):
    """
    Combined loss for multi-horizon prediction

    Combines:
    - Classification loss (CrossEntropy) with class weights
    - Regression loss (MSE or Huber)
    - Multi-task weighting
    """

    def __init__(
        self,
        horizons: List[str],
        class_weights: Optional[torch.Tensor] = None,
        classification_weight: float = 1.0,
        regression_weight: float = 1.0,
        horizon_weights: Optional[Dict[str, float]] = None,
        use_huber: bool = False
    ):
        super().__init__()

        self.horizons = horizons
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight

        # Classification loss with class weights
        self.classification_loss = nn.CrossEntropyLoss(weight=class_weights)

        # Regression loss
        if use_huber:
            self.regression_loss = nn.HuberLoss(delta=2.0)
        else:
            self.regression_loss = nn.MSELoss()

        # Horizon weights (default: equal weighting)
        if horizon_weights is None:
            self.horizon_weights = {h: 1.0 for h in horizons}
        else:
            self.horizon_weights = horizon_weights

    def forward(
        self,
        predictions: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        targets: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss

        Args:
            predictions: Dict mapping horizon -> (class_logits, regression_pred)
            targets: Dict mapping horizon -> (class_labels, regression_targets)

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        total_loss = 0.0
        loss_dict = {}

        for horizon in self.horizons:
            class_logits, regression_pred = predictions[horizon]
            class_labels, regression_target = targets[horizon]

            # Classification loss
            cls_loss = self.classification_loss(class_logits, class_labels)

            # Regression loss
            reg_loss = self.regression_loss(regression_pred.squeeze(), regression_target)

            # Combined loss for this horizon
            horizon_loss = (
                self.classification_weight * cls_loss +
                self.regression_weight * reg_loss
            ) * self.horizon_weights[horizon]

            total_loss += horizon_loss

            # Store individual losses
            loss_dict[f'{horizon}_classification'] = cls_loss.item()
            loss_dict[f'{horizon}_regression'] = reg_loss.item()
            loss_dict[f'{horizon}_total'] = horizon_loss.item()

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


def create_multihorizon_model(
    input_dim: int,
    encoder_type: str = 'transformer',
    d_model: int = 128,
    device: str = 'cpu'
) -> MultiHorizonStockModel:
    """
    Factory function to create multi-horizon model

    Args:
        input_dim: Number of input features
        encoder_type: 'transformer' or 'lstm'
        d_model: Encoder output dimension
        device: 'cpu' or 'cuda'

    Returns:
        MultiHorizonStockModel instance
    """
    model = MultiHorizonStockModel(
        input_dim=input_dim,
        encoder_type=encoder_type,
        d_model=d_model,
        nhead=8,
        num_encoder_layers=3,
        dropout=0.2,
        horizons=['1week', '1month', '3month']
    )

    return model.to(device)


if __name__ == '__main__':
    # Test the model
    print("Testing Multi-Horizon Stock Model")
    print("="*60)

    # Create model
    input_dim = 20  # Number of features per timestep
    seq_len = 60    # 60 days of history
    batch_size = 4

    model = create_multihorizon_model(
        input_dim=input_dim,
        encoder_type='transformer',
        d_model=128
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Test forward pass
    x = torch.randn(batch_size, seq_len, input_dim)
    outputs = model.forward(x, return_probabilities=True)

    print("\nForward pass output:")
    for horizon, (class_probs, regression) in outputs.items():
        print(f"  {horizon}:")
        print(f"    Classification: {class_probs.shape} (probabilities sum: {class_probs[0].sum():.3f})")
        print(f"    Regression: {regression.shape}")

    # Test prediction
    predictions = model.predict(x[:1])  # Single sample
    print("\nPrediction output:")
    for horizon, pred_dict in predictions.items():
        print(f"  {horizon}:")
        print(f"    Signal: {pred_dict['signal']}")
        print(f"    Probabilities: {pred_dict['probabilities']}")
        print(f"    Expected Return: {pred_dict['expected_return']:.2f}%")
        print(f"    Confidence: {pred_dict['confidence']:.3f}")

    print("\n" + "="*60)
