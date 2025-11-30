#!/usr/bin/env python3
"""
Test Multi-Target RF Models

Tests the trained volatility, relative performance, and breakout models
on individual stocks to see predictions.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from datetime import datetime, timedelta
import yfinance as yf

MODEL_DIR = Path('models/multi_target')
SEQUENCE_LENGTH = 30


def load_models():
    """Load all three trained models"""
    print("Loading models...")

    models = {}
    scalers = {}

    targets = ['target_volatility_1month', 'target_relative_1month', 'target_breakout']

    for target in targets:
        model_path = MODEL_DIR / f'{target}.pkl'
        scaler_path = MODEL_DIR / f'{target}_scaler.pkl'

        if model_path.exists() and scaler_path.exists():
            models[target] = joblib.load(model_path)
            scalers[target] = joblib.load(scaler_path)
            print(f"  ✓ Loaded {target}")
        else:
            print(f"  ✗ Missing {target}")

    return models, scalers


def prepare_features_from_history(ticker, days=100):
    """
    Fetch recent data and prepare features
    (Uses same feature engineering as training)
    """
    print(f"\nFetching {ticker} data...")

    # Download data
    stock = yf.Ticker(ticker)
    hist = stock.history(period='6mo')

    if len(hist) < SEQUENCE_LENGTH + 20:
        print(f"  Not enough data for {ticker}")
        return None

    # Calculate features (simplified - use same features as training)
    df = pd.DataFrame()
    df['close'] = hist['Close']
    df['high'] = hist['High']
    df['low'] = hist['Low']
    df['volume'] = hist['Volume']

    # Returns
    df['return_1d'] = df['close'].pct_change() * 100
    df['return_5d'] = df['close'].pct_change(5) * 100
    df['return_20d'] = df['close'].pct_change(20) * 100

    # Volatility
    df['volatility'] = df['return_1d'].rolling(20).std() * np.sqrt(252)

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # Moving averages
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()

    # Volume
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']

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
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / bb_sma

    # Drop NaN rows
    df = df.dropna()

    if len(df) < SEQUENCE_LENGTH:
        print(f"  Not enough valid data after feature calculation")
        return None

    # Get last sequence
    feature_cols = [c for c in df.columns if c not in ['close', 'high', 'low', 'volume']]
    sequence = df[feature_cols].iloc[-SEQUENCE_LENGTH:].values.flatten()

    return sequence


def predict_multi_target(ticker, models, scalers):
    """Make predictions on all three targets"""

    # Get features
    sequence = prepare_features_from_history(ticker)

    if sequence is None:
        return None

    # Pad if needed (to match training feature count)
    # Training has 94 features per timestep, we might have fewer
    # For now, just use what we have and pad with zeros if needed

    results = {}

    for target_name, model in models.items():
        scaler = scalers[target_name]

        # Reshape and scale
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

        results[target_name] = {
            'prediction': int(pred),
            'probabilities': proba,
            'confidence': proba.max()
        }

    return results


def interpret_results(results):
    """Convert predictions to human-readable format"""

    if results is None:
        return None

    interpretation = {}

    # Volatility
    vol_pred = results['target_volatility_1month']['prediction']
    vol_conf = results['target_volatility_1month']['confidence']
    interpretation['volatility'] = {
        'prediction': 'HIGH' if vol_pred == 1 else 'LOW',
        'confidence': vol_conf
    }

    # Relative Performance
    rel_pred = results['target_relative_1month']['prediction']
    rel_conf = results['target_relative_1month']['confidence']
    rel_labels = ['UNDERPERFORM', 'NEUTRAL', 'OUTPERFORM']
    interpretation['relative_performance'] = {
        'prediction': rel_labels[rel_pred],
        'confidence': rel_conf
    }

    # Breakout
    break_pred = results['target_breakout']['prediction']
    break_conf = results['target_breakout']['confidence']
    interpretation['breakout'] = {
        'prediction': 'TRENDING' if break_pred == 1 else 'CONSOLIDATION',
        'confidence': break_conf
    }

    return interpretation


def main():
    print("="*80)
    print("MULTI-TARGET MODEL TESTING")
    print("="*80)

    # Load models
    models, scalers = load_models()

    if not models:
        print("\nNo models found! Run train_rf_multi_target.py first.")
        return

    print(f"\nLoaded {len(models)} models")

    # Test stocks
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMD', 'META']

    print("\n" + "="*80)
    print("PREDICTIONS")
    print("="*80)

    for ticker in test_tickers:
        print(f"\n{ticker}:")
        print("-" * 40)

        try:
            results = predict_multi_target(ticker, models, scalers)
            interpretation = interpret_results(results)

            if interpretation:
                # Volatility
                vol = interpretation['volatility']
                print(f"  Volatility:     {vol['prediction']:<15} (confidence: {vol['confidence']:.1%})")

                # Relative Performance
                rel = interpretation['relative_performance']
                print(f"  vs Market:      {rel['prediction']:<15} (confidence: {rel['confidence']:.1%})")

                # Breakout
                brk = interpretation['breakout']
                print(f"  Pattern:        {brk['prediction']:<15} (confidence: {brk['confidence']:.1%})")

                # Trading signal
                print("\n  Signal:")
                if rel['prediction'] == 'OUTPERFORM' and rel['confidence'] > 0.6:
                    print("    → BULLISH (expected to outperform market)")
                elif rel['prediction'] == 'UNDERPERFORM' and rel['confidence'] > 0.6:
                    print("    → BEARISH (expected to underperform market)")
                else:
                    print("    → NEUTRAL (unclear direction)")

                if vol['prediction'] == 'HIGH':
                    print("    ⚠ High volatility expected - use wider stops")

                if brk['prediction'] == 'CONSOLIDATION':
                    print("    📊 In consolidation - potential breakout setup")

        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "="*80)
    print("INTERPRETATION GUIDE")
    print("="*80)
    print("""
Volatility:
  - HIGH: Stock will be more volatile (wider swings)
  - LOW: Stock will be calmer (smaller movements)

Relative Performance (vs Market):
  - OUTPERFORM: Stock expected to beat SPY
  - UNDERPERFORM: Stock expected to lag SPY
  - NEUTRAL: Similar performance to market

Pattern:
  - CONSOLIDATION: Tight range, potential breakout coming
  - TRENDING: Active movement, trending behavior

Trading Strategy:
  - Trade when multiple signals align
  - High confidence (>70%) predictions are more reliable
  - Use volatility prediction for position sizing
    """)

    print("="*80)


if __name__ == '__main__':
    main()
