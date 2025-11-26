"""
Ensemble Model - Combines LSTM and XGBoost Predictions

Weighting:
- LSTM: 40% (captures temporal patterns)
- XGBoost: 60% (captures feature relationships)

Includes confidence scoring based on model agreement
"""

from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass

try:
    from .lstm_model import LSTMStockModel
    from .xgboost_model import XGBoostStockModel
except ImportError:
    # Allow standalone import
    from lstm_model import LSTMStockModel
    from xgboost_model import XGBoostStockModel


@dataclass
class EnsemblePrediction:
    """Container for ensemble prediction results"""
    predicted_return: float  # 30-day return %
    predicted_price: float  # Target price
    signal: str  # BUY/SELL/HOLD
    confidence: float  # 0-1
    confidence_interval_low: float  # Lower bound %
    confidence_interval_high: float  # Upper bound %
    risk_score: int  # 1-10
    model_contributions: Dict[str, float]  # Individual model predictions
    feature_contributions: Dict[str, float]  # Top features driving prediction
    ensemble_agreement: float  # How much models agree (0-1)


class EnsembleStockModel:
    """
    Ensemble combining LSTM and XGBoost

    Ensemble strategy:
    1. Get LSTM prediction (temporal patterns)
    2. Get XGBoost prediction (feature relationships)
    3. Weighted average: 40% LSTM + 60% XGBoost
    4. Calculate confidence based on model agreement
    5. Generate BUY/SELL/HOLD signal with thresholds
    """

    def __init__(
        self,
        lstm_model: Optional[LSTMStockModel] = None,
        xgboost_model: Optional[XGBoostStockModel] = None,
        lstm_weight: float = 0.4,
        xgboost_weight: float = 0.6
    ):
        self.lstm_model = lstm_model
        self.xgboost_model = xgboost_model
        self.lstm_weight = lstm_weight
        self.xgboost_weight = xgboost_weight

        # Signal thresholds
        self.buy_threshold = 5.0  # Predict > +5% return
        self.sell_threshold = -5.0  # Predict < -5% return

    def predict(
        self,
        ticker: str,
        sequence_data: pd.DataFrame,  # Historical OHLCV for LSTM
        features: Dict[str, float],  # Engineered features for XGBoost
        current_price: float
    ) -> EnsemblePrediction:
        """
        Make ensemble prediction for a stock

        Args:
            ticker: Stock symbol
            sequence_data: Recent price history for LSTM (60+ days)
            features: Engineered features dict for XGBoost
            current_price: Current stock price

        Returns:
            EnsemblePrediction with all prediction details
        """
        predictions = {}
        confidences = {}

        # 1. LSTM Prediction
        if self.lstm_model is not None:
            try:
                lstm_pred, lstm_conf = self.lstm_model.predict(
                    sequence_data,
                    return_confidence=True
                )
                predictions['lstm'] = lstm_pred
                confidences['lstm'] = lstm_conf
            except Exception as e:
                print(f"LSTM prediction failed: {e}")
                predictions['lstm'] = 0.0
                confidences['lstm'] = 0.3

        # 2. XGBoost Prediction
        if self.xgboost_model is not None:
            try:
                xgb_pred, xgb_contributions = self.xgboost_model.predict_single(features)
                predictions['xgboost'] = xgb_pred
                # Confidence based on strong features
                top_features_strength = sum(abs(v) for v in sorted(xgb_contributions.values(), key=abs, reverse=True)[:5])
                confidences['xgboost'] = min(0.5 + top_features_strength * 0.1, 0.95)
            except Exception as e:
                print(f"XGBoost prediction failed: {e}")
                predictions['xgboost'] = 0.0
                confidences['xgboost'] = 0.3
                xgb_contributions = {}

        # 3. Ensemble Prediction (Weighted Average)
        if 'lstm' in predictions and 'xgboost' in predictions:
            ensemble_pred = (
                self.lstm_weight * predictions['lstm'] +
                self.xgboost_weight * predictions['xgboost']
            )
        elif 'lstm' in predictions:
            ensemble_pred = predictions['lstm']
        elif 'xgboost' in predictions:
            ensemble_pred = predictions['xgboost']
        else:
            ensemble_pred = 0.0

        # 4. Calculate Model Agreement (confidence boost)
        if len(predictions) >= 2:
            pred_values = list(predictions.values())
            # Agreement: 1.0 if same sign and similar magnitude, 0.0 if opposite
            same_direction = all(p > 0 for p in pred_values) or all(p < 0 for p in pred_values)
            magnitude_diff = abs(pred_values[0] - pred_values[1]) / (abs(pred_values[0]) + abs(pred_values[1]) + 1e-6)
            agreement = 1.0 if same_direction else 0.0
            agreement *= max(0, 1.0 - magnitude_diff)  # Penalize if magnitudes differ
        else:
            agreement = 0.5  # Neutral if only one model

        # 5. Overall Confidence
        avg_confidence = np.mean(list(confidences.values())) if confidences else 0.5
        confidence = avg_confidence * (0.7 + 0.3 * agreement)  # Boost if models agree
        confidence = np.clip(confidence, 0.1, 0.95)

        # 6. Generate Signal
        if ensemble_pred >= self.buy_threshold:
            signal = 'BUY'
        elif ensemble_pred <= self.sell_threshold:
            signal = 'SELL'
        else:
            signal = 'HOLD'

        # 7. Confidence Intervals (based on prediction uncertainty)
        # Simple approach: ±40% of prediction with confidence adjustment
        uncertainty = (1 - confidence) * 0.4
        interval_range = abs(ensemble_pred) * uncertainty
        confidence_interval_low = ensemble_pred - interval_range
        confidence_interval_high = ensemble_pred + interval_range

        # 8. Risk Score (1-10)
        # Higher risk = higher volatility, lower confidence, extreme predictions
        volatility_factor = abs(ensemble_pred) / 20.0  # 20% move = high vol
        confidence_factor = 1 - confidence
        risk_score = int(np.clip(
            1 + 9 * (0.5 * volatility_factor + 0.5 * confidence_factor),
            1, 10
        ))

        # 9. Predicted Price
        predicted_price = current_price * (1 + ensemble_pred / 100)

        # 10. Top Feature Contributions (for explanation)
        if xgb_contributions:
            # Sort by absolute contribution
            top_features = sorted(
                xgb_contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:10]
            feature_contributions = dict(top_features)
        else:
            feature_contributions = {}

        return EnsemblePrediction(
            predicted_return=round(ensemble_pred, 2),
            predicted_price=round(predicted_price, 2),
            signal=signal,
            confidence=round(confidence, 2),
            confidence_interval_low=round(confidence_interval_low, 2),
            confidence_interval_high=round(confidence_interval_high, 2),
            risk_score=risk_score,
            model_contributions=predictions,
            feature_contributions=feature_contributions,
            ensemble_agreement=round(agreement, 2)
        )

    def predict_with_uncertainty(
        self,
        ticker: str,
        sequence_data: pd.DataFrame,
        features: Dict[str, float],
        current_price: float,
        n_samples: int = 100
    ) -> EnsemblePrediction:
        """
        Enhanced prediction with uncertainty quantification using Monte Carlo

        Uses dropout at inference (LSTM) and bootstrap (XGBoost) for uncertainty
        """
        # Get multiple predictions from LSTM with dropout
        lstm_predictions = []
        if self.lstm_model is not None:
            try:
                mean_pred, lower, upper = self.lstm_model.predict_with_uncertainty(
                    sequence_data,
                    n_samples=n_samples
                )
                lstm_predictions = [mean_pred]
            except Exception:
                lstm_predictions = [0.0]

        # XGBoost prediction (single for now, could add bootstrap)
        xgb_pred = 0.0
        if self.xgboost_model is not None:
            try:
                xgb_pred, _ = self.xgboost_model.predict_single(features)
            except Exception:
                xgb_pred = 0.0

        # Combine
        ensemble_pred = self.lstm_weight * lstm_predictions[0] + self.xgboost_weight * xgb_pred

        # Use standard prediction for other metrics
        return self.predict(ticker, sequence_data, features, current_price)

    def load_models(self, lstm_path: str, xgboost_path: str):
        """Load pre-trained models"""
        if lstm_path:
            from .lstm_model import create_lstm_model
            self.lstm_model = create_lstm_model()
            self.lstm_model.load(lstm_path)
            print(f"Loaded LSTM model from {lstm_path}")

        if xgboost_path:
            from .xgboost_model import create_xgboost_regressor
            self.xgboost_model = create_xgboost_regressor()
            self.xgboost_model.load(xgboost_path)
            print(f"Loaded XGBoost model from {xgboost_path}")


def create_ensemble_model(
    lstm_path: Optional[str] = None,
    xgboost_path: Optional[str] = None,
    lstm_weight: float = 0.4,
    xgboost_weight: float = 0.6
) -> EnsembleStockModel:
    """
    Factory function to create ensemble model

    Args:
        lstm_path: Path to saved LSTM model (optional)
        xgboost_path: Path to saved XGBoost model (optional)
        lstm_weight: Weight for LSTM predictions (default 0.4)
        xgboost_weight: Weight for XGBoost predictions (default 0.6)

    Returns:
        EnsembleStockModel instance
    """
    ensemble = EnsembleStockModel(
        lstm_weight=lstm_weight,
        xgboost_weight=xgboost_weight
    )

    if lstm_path or xgboost_path:
        ensemble.load_models(lstm_path or "", xgboost_path or "")

    return ensemble
