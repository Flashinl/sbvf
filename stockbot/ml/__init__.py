"""
Machine Learning Module for Stock Prediction

Components:
- feature_engineering: Extract 50+ features from stock data
- models/lstm_model: LSTM for time-series prediction
- models/xgboost_model: XGBoost for tabular features
- models/ensemble: Ensemble combining LSTM + XGBoost
- shap_explainer: SHAP for feature importance
- explainer: Specific, actionable explanations
- predictor: Main prediction engine
- train: Model training script
"""

from .predictor import StockPredictor, create_predictor
from .feature_engineering import extract_all_features
from .explainer import SpecificExplainer

__all__ = [
    'StockPredictor',
    'create_predictor',
    'extract_all_features',
    'SpecificExplainer'
]
