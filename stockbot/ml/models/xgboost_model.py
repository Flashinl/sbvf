"""
XGBoost Model for Tabular Stock Prediction

Uses all 50+ engineered features:
- Technical indicators
- Fundamental metrics
- Sentiment scores
- Market context

Predicts 30-day forward return with high interpretability via SHAP
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import pickle
from pathlib import Path

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed. Install with: pip install xgboost")


class XGBoostStockModel:
    """
    XGBoost model for stock prediction using engineered features
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        objective: str = 'reg:squarederror',
        random_state: int = 42
    ):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is required. Install with: pip install xgboost")

        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'objective': objective,
            'random_state': random_state,
            'tree_method': 'hist',
            'eval_metric': 'rmse'
        }

        self.model = xgb.XGBRegressor(**self.params)
        self.feature_names = None
        self.feature_importance = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        early_stopping_rounds: int = 50,
        verbose: bool = True
    ) -> Dict:
        """
        Train XGBoost model

        Args:
            X_train: Training features
            y_train: Training targets (30-day return %)
            X_val: Validation features
            y_val: Validation targets
            early_stopping_rounds: Stop if no improvement for N rounds
            verbose: Print training progress

        Returns:
            training_history: Dict with metrics
        """
        self.feature_names = list(X_train.columns)

        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        # Train model
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds if X_val is not None else None,
            verbose=verbose
        )

        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Training metrics
        history = {
            'best_iteration': self.model.best_iteration if X_val is not None else len(self.model.evals_result()['validation_0']['rmse']),
            'train_rmse': self.model.evals_result()['validation_0']['rmse'][-1],
            'val_rmse': self.model.evals_result()['validation_1']['rmse'][-1] if X_val is not None else None,
            'feature_importance': self.feature_importance.to_dict('records')
        }

        if verbose:
            print(f"\nTraining complete!")
            print(f"Best iteration: {history['best_iteration']}")
            print(f"Train RMSE: {history['train_rmse']:.4f}")
            if history['val_rmse']:
                print(f"Validation RMSE: {history['val_rmse']:.4f}")
            print("\nTop 10 Most Important Features:")
            print(self.feature_importance.head(10).to_string(index=False))

        return history

    def predict(
        self,
        X: pd.DataFrame,
        return_feature_contributions: bool = False
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Make predictions

        Args:
            X: Features DataFrame
            return_feature_contributions: If True, return SHAP-like contributions

        Returns:
            predictions: Predicted 30-day returns
            contributions: Dict of feature contributions (if requested)
        """
        predictions = self.model.predict(X)

        if return_feature_contributions:
            # Get feature contributions (approximate SHAP values)
            contributions = {}
            for i, feature in enumerate(self.feature_names):
                # Feature importance weighted by feature value
                importance = self.model.feature_importances_[i]
                contributions[feature] = importance * X[feature].iloc[0] if len(X) == 1 else importance

            return predictions, contributions

        return predictions, None

    def predict_single(
        self,
        features: Dict[str, float]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Predict for a single stock with feature contributions

        Args:
            features: Dict of feature_name -> value

        Returns:
            prediction: Predicted 30-day return %
            contributions: Dict of feature contributions
        """
        # Create DataFrame with correct column order
        X = pd.DataFrame([features])[self.feature_names]

        # Fill missing features with 0
        X = X.fillna(0)

        prediction = self.model.predict(X)[0]

        # Calculate feature contributions
        contributions = {}
        for i, feature in enumerate(self.feature_names):
            importance = self.model.feature_importances_[i]
            value = X[feature].iloc[0]
            # Simple contribution: importance * normalized_value
            contributions[feature] = importance * (value / (abs(value) + 1))

        return float(prediction), contributions

    def get_top_features(self, n: int = 10) -> pd.DataFrame:
        """Get top N most important features"""
        if self.feature_importance is None:
            raise ValueError("Model not trained yet")
        return self.feature_importance.head(n)

    def save(self, path: str):
        """Save model to disk"""
        save_dict = {
            'model': self.model,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'params': self.params
        }
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from disk"""
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)

        self.model = save_dict['model']
        self.feature_names = save_dict['feature_names']
        self.feature_importance = save_dict['feature_importance']
        self.params = save_dict['params']
        print(f"Model loaded from {path}")


class XGBoostClassifier(XGBoostStockModel):
    """
    XGBoost classifier for BUY/SELL/HOLD signals

    Maps 30-day return to discrete signals:
    - BUY: return > +5%
    - SELL: return < -5%
    - HOLD: -5% <= return <= +5%
    """

    def __init__(self, **kwargs):
        # Override objective for classification
        kwargs['objective'] = 'multi:softmax'
        kwargs['num_class'] = 3
        super().__init__(**kwargs)

    def _return_to_signal(self, returns: np.ndarray) -> np.ndarray:
        """Convert continuous returns to discrete signals (0=SELL, 1=HOLD, 2=BUY)"""
        signals = np.zeros(len(returns), dtype=int)
        signals[returns > 5.0] = 2  # BUY
        signals[returns < -5.0] = 0  # SELL
        signals[(returns >= -5.0) & (returns <= 5.0)] = 1  # HOLD
        return signals

    def _signal_to_label(self, signal: int) -> str:
        """Convert numeric signal to string label"""
        return ['SELL', 'HOLD', 'BUY'][signal]

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,  # Continuous returns
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs
    ) -> Dict:
        """Train with continuous returns converted to signals"""
        # Convert returns to signals
        y_train_signals = self._return_to_signal(y_train.values)
        y_val_signals = self._return_to_signal(y_val.values) if y_val is not None else None

        # Update eval_metric for classification
        self.model.set_params(eval_metric='mlogloss')

        return super().train(
            X_train,
            pd.Series(y_train_signals, index=y_train.index),
            X_val,
            pd.Series(y_val_signals, index=y_val.index) if y_val_signals is not None else None,
            **kwargs
        )

    def predict_signal(
        self,
        X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict signals with probabilities

        Returns:
            signals: Array of 'BUY', 'SELL', 'HOLD'
            probabilities: Array of shape (n_samples, 3) with class probabilities
        """
        signal_nums = self.model.predict(X)
        signals = np.array([self._signal_to_label(int(s)) for s in signal_nums])

        # Get probabilities
        probabilities = self.model.predict_proba(X)

        return signals, probabilities


def create_xgboost_regressor() -> XGBoostStockModel:
    """Factory function to create XGBoost regressor"""
    return XGBoostStockModel(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )


def create_xgboost_classifier() -> XGBoostClassifier:
    """Factory function to create XGBoost classifier"""
    return XGBoostClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
