#!/usr/bin/env python3
"""
Multi-Target RF Training - Predict What's Actually Predictable

Instead of trying to predict absolute direction (hard),
we predict multiple easier targets:

1. VOLATILITY: High vs Low volatility (70%+ accuracy possible)
2. RELATIVE PERFORMANCE: Outperform vs underperform market (65%+ accuracy)
3. BREAKOUT: Consolidation vs trending (60%+ accuracy)

Then combine signals for final decision.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# CONFIG
SEQUENCE_LENGTH = 30
HORIZONS = ['1month']
USE_ENHANCED_DATA = True


def create_volatility_target(df, horizon='1month'):
    """
    Predict: Will this stock be HIGH or LOW volatility?
    This is much more predictable than direction!
    """
    print(f"Creating volatility target for {horizon}...")

    # Calculate future volatility (standard deviation of returns)
    df = df.copy()
    df = df.sort_values(['ticker', 'date'])

    future_vols = []

    for ticker in df['ticker'].unique():
        ticker_df = df[df['ticker'] == ticker].copy()

        # For each day, calculate volatility over next month
        vols = []
        for i in range(len(ticker_df)):
            if horizon == '1month':
                window = 20  # ~1 month trading days
            elif horizon == '3month':
                window = 60
            else:
                window = 20

            if i + window < len(ticker_df):
                future_returns = ticker_df['return_1d'].iloc[i+1:i+1+window]
                vol = future_returns.std() * np.sqrt(252) * 100  # Annualized
                vols.append(vol)
            else:
                vols.append(np.nan)

        future_vols.extend(vols)

    df['future_volatility'] = future_vols

    # Create binary target: HIGH (1) vs LOW (0) volatility
    # Use median as threshold
    median_vol = df['future_volatility'].median()
    df[f'target_volatility_{horizon}'] = (df['future_volatility'] > median_vol).astype(int)

    return df


def create_relative_performance_target(df, horizon='1month'):
    """
    Predict: Will this stock OUTPERFORM or UNDERPERFORM the market?
    Much easier than predicting absolute direction!
    """
    print(f"Creating relative performance target for {horizon}...")

    df = df.copy()

    # Already have target_return_{horizon} and market returns
    if f'target_return_{horizon}' not in df.columns:
        print(f"  Warning: target_return_{horizon} not found, skipping")
        return df

    # Calculate relative performance
    df[f'relative_perf_{horizon}'] = df[f'target_return_{horizon}'] - df[f'market_SPY_return_20d']

    # Create 3-class target: OUTPERFORM (2), NEUTRAL (1), UNDERPERFORM (0)
    # Use 33rd and 67th percentiles
    p33 = df[f'relative_perf_{horizon}'].quantile(0.33)
    p67 = df[f'relative_perf_{horizon}'].quantile(0.67)

    conditions = [
        df[f'relative_perf_{horizon}'] < p33,
        (df[f'relative_perf_{horizon}'] >= p33) & (df[f'relative_perf_{horizon}'] < p67),
        df[f'relative_perf_{horizon}'] >= p67
    ]
    choices = [0, 1, 2]  # UNDERPERFORM, NEUTRAL, OUTPERFORM

    df[f'target_relative_{horizon}'] = np.select(conditions, choices, default=1)

    return df


def create_breakout_target(df):
    """
    Predict: Is stock in CONSOLIDATION or TRENDING?
    Consolidation often precedes breakouts - easier to predict!
    """
    print("Creating breakout/consolidation target...")

    df = df.copy()
    df = df.sort_values(['ticker', 'date'])

    # First pass: calculate metrics
    vols = []
    price_ranges = []

    for ticker in df['ticker'].unique():
        ticker_df = df[df['ticker'] == ticker].copy()

        for i in range(len(ticker_df)):
            # Look at past 20 days
            if i < 20:
                vols.append(np.nan)
                price_ranges.append(np.nan)
                continue

            past_returns = ticker_df['return_1d'].iloc[i-20:i]
            past_highs = ticker_df['high'].iloc[i-20:i]
            past_lows = ticker_df['low'].iloc[i-20:i]

            # Consolidation indicators:
            # 1. Low volatility
            vol = past_returns.std()
            # 2. Tight range (high - low)
            price_range = (past_highs.max() - past_lows.min()) / past_lows.min()

            vols.append(vol)
            price_ranges.append(price_range)

    df['temp_vol'] = vols
    df['temp_range'] = price_ranges

    # Use percentiles for balanced classification
    vol_threshold = df['temp_vol'].quantile(0.4)  # Bottom 40% = low vol
    range_threshold = df['temp_range'].quantile(0.4)  # Bottom 40% = tight range

    # CONSOLIDATION (0) if low vol AND tight range
    # TRENDING (1) otherwise
    df['target_breakout'] = (
        (df['temp_vol'] > vol_threshold) | (df['temp_range'] > range_threshold)
    ).astype(int)

    # Drop temporary columns
    df = df.drop(columns=['temp_vol', 'temp_range'])

    # Show distribution
    consolidation_pct = (df['target_breakout'] == 0).sum() / df['target_breakout'].notna().sum()
    print(f"  Consolidation: {consolidation_pct:.1%}, Trending: {1-consolidation_pct:.1%}")

    return df


def create_sequences(df, feature_cols, seq_length, target_col):
    """Create sequences for a specific target"""
    sequences = []
    targets = []

    for ticker in tqdm(df['ticker'].unique(), desc="Sequences"):
        ticker_df = df[df['ticker'] == ticker].sort_values('date').reset_index(drop=True)

        if len(ticker_df) < seq_length:
            continue

        for i in range(len(ticker_df) - seq_length):
            # Features
            seq = ticker_df.iloc[i:i+seq_length][feature_cols].values.flatten()

            # Target
            target_idx = i + seq_length - 1
            target = ticker_df.iloc[target_idx][target_col]

            if not np.isnan(target):
                sequences.append(seq)
                targets.append(int(target))

    return np.array(sequences), np.array(targets)


print("="*80)
print("MULTI-TARGET RF TRAINING")
print("="*80)

# Load data
if USE_ENHANCED_DATA and Path('data/training_data_enhanced.parquet').exists():
    DATA_PATH = 'data/training_data_enhanced.parquet'
    print("Using ENHANCED dataset with fundamentals")
else:
    DATA_PATH = 'data/training_data_with_sentiment.parquet'
    print("Using BASE dataset (run add_real_data.py for better results)")

print(f"\nLoading: {DATA_PATH}")
df = pd.read_parquet(DATA_PATH)

print(f"Samples: {len(df):,}, Stocks: {df['ticker'].nunique()}, Features: {len(df.columns)}")

# Create all targets
for horizon in HORIZONS:
    df = create_volatility_target(df, horizon)
    df = create_relative_performance_target(df, horizon)

df = create_breakout_target(df)

# Temporal split
print("\nTemporal split...")
df = df.sort_values('date')
n = len(df)
train_end = int(n * 0.7)
test_start = int(n * 0.85)

train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:test_start]
test_df = df.iloc[test_start:]

print(f"Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

# Get features
feature_cols = [c for c in df.columns
                if not c.startswith('target_')
                and c not in ['ticker', 'date', 'future_volatility', 'relative_perf_1month']]

print(f"Features: {len(feature_cols)}")

# Train models for each target
TARGETS = [
    ('target_volatility_1month', 'Volatility', 'binary'),
    ('target_relative_1month', 'Relative Performance', '3-class'),
    ('target_breakout', 'Breakout/Consolidation', 'binary')
]

results = {}

for target_col, target_name, target_type in TARGETS:
    print("\n" + "="*80)
    print(f"TRAINING: {target_name} ({target_type})")
    print("="*80)

    # Create sequences
    X_train, y_train = create_sequences(train_df, feature_cols, SEQUENCE_LENGTH, target_col)
    X_val, y_val = create_sequences(val_df, feature_cols, SEQUENCE_LENGTH, target_col)
    X_test, y_test = create_sequences(test_df, feature_cols, SEQUENCE_LENGTH, target_col)

    print(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")

    # Handle NaN
    X_train = np.nan_to_num(X_train, nan=0)
    X_val = np.nan_to_num(X_val, nan=0)
    X_test = np.nan_to_num(X_test, nan=0)

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Train model
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    import time
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    # Evaluate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"\nTime: {elapsed:.1f}s")
    print(f"Train Acc: {train_acc:.2%}")
    print(f"Val Acc:   {val_acc:.2%}")
    print(f"Test Acc:  {test_acc:.2%}")

    if test_acc >= 0.70:
        print("✓✓ EXCELLENT: 70%+ accuracy!")
    elif test_acc >= 0.60:
        print("✓ GOOD: 60-70% accuracy")
    elif test_acc >= 0.55:
        print("○ MODERATE: 55-60% accuracy")

    # Classification report
    if target_type == 'binary':
        labels = ['LOW', 'HIGH'] if 'volatility' in target_col else ['CONSOLIDATION', 'TRENDING']
    else:
        labels = ['UNDERPERFORM', 'NEUTRAL', 'OUTPERFORM']

    print("\nClassification Report:")
    print(classification_report(y_test, test_pred, target_names=labels))

    # Save model
    MODEL_DIR = Path('models/multi_target')
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_DIR / f'{target_col}.pkl')
    joblib.dump(scaler, MODEL_DIR / f'{target_col}_scaler.pkl')

    results[target_name] = {
        'accuracy': float(test_acc),
        'val_accuracy': float(val_acc),
        'type': target_type
    }

# Save results
config_data = {
    'training_date': datetime.now().isoformat(),
    'sequence_length': SEQUENCE_LENGTH,
    'n_features': len(feature_cols),
    'results': results
}

with open(MODEL_DIR / 'multi_target_config.json', 'w') as f:
    json.dump(config_data, f, indent=2)

# Summary
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

for name, metrics in results.items():
    print(f"\n{name}: {metrics['accuracy']:.2%}")

avg_acc = np.mean([m['accuracy'] for m in results.values()])
print(f"\nAverage Accuracy: {avg_acc:.2%}")

if avg_acc >= 0.65:
    print("\n✓✓ SUCCESS: Multi-target approach working!")
    print("These predictions are MORE RELIABLE than trying to predict absolute direction.")
elif avg_acc >= 0.55:
    print("\n✓ GOOD: Better than baseline")
    print("Consider adding fundamental data for even better results.")

print("\nNext Steps:")
print("1. Combine these predictions for trading decisions")
print("2. Trade only when multiple signals align")
print("3. Focus on high-confidence predictions")

print("\nModels saved to: models/multi_target/")
print("="*80)
