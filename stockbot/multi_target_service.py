"""
Multi-Target RF Model Service

Provides predictions for:
1. Volatility (HIGH/LOW)
2. Relative Performance (OUTPERFORM/NEUTRAL/UNDERPERFORM market)
3. Breakout/Consolidation (TRENDING/CONSOLIDATION)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List
import joblib
import yfinance as yf
from datetime import datetime, timedelta


class MultiTargetService:
    """Service for multi-target RF predictions"""

    def __init__(self, model_dir: str = 'models/multi_target'):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.scalers = {}
        self.sequence_length = 30
        self._available = False

        # Try to load models
        self._load_models()

    def _load_models(self):
        """Load all three trained models"""
        targets = [
            ('target_volatility_1month', 'volatility'),
            ('target_relative_1month', 'relative'),
            ('target_breakout', 'breakout')
        ]

        loaded = 0
        for target_col, name in targets:
            model_path = self.model_dir / f'{target_col}.pkl'
            scaler_path = self.model_dir / f'{target_col}_scaler.pkl'

            if model_path.exists() and scaler_path.exists():
                try:
                    self.models[name] = joblib.load(model_path)
                    self.scalers[name] = joblib.load(scaler_path)
                    loaded += 1
                except Exception as e:
                    print(f"[WARNING] Failed to load {name} model: {e}")

        self._available = loaded == 3
        if self._available:
            print(f"[INFO] Multi-target service loaded {loaded}/3 models successfully")
        else:
            print(f"[WARNING] Multi-target service only loaded {loaded}/3 models")

    def is_available(self) -> bool:
        """Check if all models are loaded"""
        return self._available

    def _fetch_features(self, ticker: str, days: int = 100) -> Optional[np.ndarray]:
        """
        Fetch and calculate features for prediction
        Returns flattened sequence ready for model input
        """
        try:
            # Download data
            stock = yf.Ticker(ticker)
            hist = stock.history(period='6mo')

            if len(hist) < self.sequence_length + 20:
                return None

            # Calculate features
            df = pd.DataFrame()
            df['close'] = hist['Close']
            df['high'] = hist['High']
            df['low'] = hist['Low']
            df['volume'] = hist['Volume']

            # Returns
            df['return_1d'] = df['close'].pct_change() * 100
            df['return_5d'] = df['close'].pct_change(5) * 100
            df['return_20d'] = df['close'].pct_change(20) * 100
            df['return_60d'] = df['close'].pct_change(60) * 100

            # Volatility
            df['volatility'] = df['return_1d'].rolling(20).std() * np.sqrt(252)
            df['volatility_60d'] = df['return_1d'].rolling(60).std() * np.sqrt(252)

            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-10)
            df['rsi_14'] = 100 - (100 / (1 + rs))

            # Moving averages
            df['sma_10'] = df['close'].rolling(10).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()

            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']

            # Volume
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1)
            df['volume_change'] = df['volume'].pct_change()

            # ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['atr'] = true_range.rolling(14).mean()
            df['atr_pct'] = (df['atr'] / df['close']) * 100

            # Bollinger Bands
            bb_sma = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['bb_upper'] = bb_sma + (bb_std * 2)
            df['bb_lower'] = bb_sma - (bb_std * 2)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (bb_sma + 1e-10)

            # Get fundamental data
            info = stock.info
            fundamentals = {
                'fund_pe_ratio': info.get('trailingPE', np.nan),
                'fund_forward_pe': info.get('forwardPE', np.nan),
                'fund_eps': info.get('trailingEps', np.nan),
                'fund_profit_margin': info.get('profitMargins', np.nan),
                'fund_roe': info.get('returnOnEquity', np.nan),
                'fund_debt_to_equity': info.get('debtToEquity', np.nan),
                'fund_beta': info.get('beta', np.nan),
            }

            # Add fundamentals as constants
            for key, value in fundamentals.items():
                df[key] = value

            # Drop NaN rows
            df = df.dropna()

            if len(df) < self.sequence_length:
                return None

            # Get last sequence
            feature_cols = [c for c in df.columns if c not in ['close', 'high', 'low', 'volume']]
            sequence = df[feature_cols].iloc[-self.sequence_length:].values.flatten()

            return sequence

        except Exception as e:
            print(f"[WARNING] Feature extraction failed for {ticker}: {e}")
            return None

    async def predict(self, ticker: str) -> Optional[Dict]:
        """
        Make multi-target predictions for a stock

        Returns:
            dict with keys:
                - volatility: {prediction: str, confidence: float}
                - relative_performance: {prediction: str, confidence: float}
                - breakout: {prediction: str, confidence: float}
                - trading_signal: str (BULLISH/BEARISH/NEUTRAL)
                - risk_level: str (HIGH/MODERATE/LOW)
        """
        if not self.is_available():
            return None

        # Fetch features
        sequence = self._fetch_features(ticker)
        if sequence is None:
            return None

        results = {}

        # Predict each target
        for model_name in ['volatility', 'relative', 'breakout']:
            model = self.models[model_name]
            scaler = self.scalers[model_name]

            # Prepare input
            X = sequence.reshape(1, -1)

            # Handle size mismatch - pad with zeros if needed
            expected_features = scaler.n_features_in_
            current_features = X.shape[1]

            if current_features < expected_features:
                padding = np.zeros((1, expected_features - current_features))
                X = np.hstack([X, padding])
            elif current_features > expected_features:
                X = X[:, :expected_features]

            X = np.nan_to_num(X, nan=0)
            X_scaled = scaler.transform(X)

            # Predict
            pred = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0]

            results[model_name] = {
                'prediction_code': int(pred),
                'probabilities': proba.tolist(),
                'confidence': float(proba.max())
            }

        # Interpret results
        volatility_pred = "HIGH" if results['volatility']['prediction_code'] == 1 else "LOW"

        rel_code = results['relative']['prediction_code']
        relative_pred = ["UNDERPERFORM", "NEUTRAL", "OUTPERFORM"][rel_code]

        breakout_pred = "TRENDING" if results['breakout']['prediction_code'] == 1 else "CONSOLIDATION"

        # Determine trading signal
        if relative_pred == "OUTPERFORM" and results['relative']['confidence'] > 0.6:
            trading_signal = "BULLISH"
        elif relative_pred == "UNDERPERFORM" and results['relative']['confidence'] > 0.6:
            trading_signal = "BEARISH"
        else:
            trading_signal = "NEUTRAL"

        # Determine risk level
        if volatility_pred == "HIGH":
            risk_level = "HIGH"
        elif volatility_pred == "LOW":
            risk_level = "LOW"
        else:
            risk_level = "MODERATE"

        return {
            'ticker': ticker,
            'volatility': {
                'prediction': volatility_pred,
                'confidence': results['volatility']['confidence']
            },
            'relative_performance': {
                'prediction': relative_pred,
                'confidence': results['relative']['confidence']
            },
            'breakout': {
                'prediction': breakout_pred,
                'confidence': results['breakout']['confidence']
            },
            'trading_signal': trading_signal,
            'risk_level': risk_level,
            'interpretation': self._get_interpretation(
                volatility_pred, relative_pred, breakout_pred, trading_signal, risk_level
            )
        }

    def _get_interpretation(self, volatility, relative, breakout, signal, risk):
        """Generate human-readable interpretation"""
        messages = []

        # Trading signal
        if signal == "BULLISH":
            messages.append("Expected to outperform the market")
        elif signal == "BEARISH":
            messages.append("Expected to underperform the market")
        else:
            messages.append("Expected to move in line with the market")

        # Volatility warning
        if volatility == "HIGH":
            messages.append("High volatility expected - use wider stop losses and smaller position sizes")
        else:
            messages.append("Low volatility expected - calmer price movements")

        # Pattern info
        if breakout == "CONSOLIDATION":
            messages.append("Currently in consolidation - potential breakout setup")
        else:
            messages.append("Actively trending")

        return " | ".join(messages)


# Global instance
_multi_target_service: Optional[MultiTargetService] = None


def init_multi_target_service(model_dir: str = 'models/multi_target'):
    """Initialize the multi-target service"""
    global _multi_target_service
    _multi_target_service = MultiTargetService(model_dir)


def get_multi_target_service() -> MultiTargetService:
    """Get the multi-target service instance"""
    global _multi_target_service
    if _multi_target_service is None:
        init_multi_target_service()
    return _multi_target_service
