#!/usr/bin/env python3
"""
Comprehensive Multi-Target Model Testing

Tests models on proper test set with ALL features (including fundamentals)
Shows detailed accuracy, confusion matrices, and per-stock performance.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json

MODEL_DIR = Path('models/multi_target')
SEQUENCE_LENGTH = 30


def create_sequences(df, feature_cols, seq_length, target_col):
    """Create sequences for testing (same as training)"""
    sequences = []
    targets = []
    tickers = []
    dates = []

    for ticker in df['ticker'].unique():
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
                tickers.append(ticker)
                dates.append(ticker_df.iloc[target_idx]['date'])

    return np.array(sequences), np.array(targets), tickers, dates


def evaluate_model(model, scaler, X_test, y_test, target_name, target_type):
    """Evaluate a single model"""
    print(f"\n{'='*80}")
    print(f"EVALUATING: {target_name}")
    print(f"{'='*80}")

    # Preprocess
    X_test_clean = np.nan_to_num(X_test, nan=0)
    X_test_scaled = scaler.transform(X_test_clean)

    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.2%}")

    if accuracy >= 0.75:
        print("✓✓✓ EXCELLENT: 75%+ accuracy!")
    elif accuracy >= 0.70:
        print("✓✓ VERY GOOD: 70-75% accuracy")
    elif accuracy >= 0.60:
        print("✓ GOOD: 60-70% accuracy")
    elif accuracy >= 0.55:
        print("○ MODERATE: 55-60% accuracy")
    else:
        print("~ Below 55%")

    # Classification report
    if target_type == 'binary':
        if 'volatility' in target_name:
            labels = ['LOW', 'HIGH']
        else:
            labels = ['CONSOLIDATION', 'TRENDING']
    else:
        labels = ['UNDERPERFORM', 'NEUTRAL', 'OUTPERFORM']

    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=labels, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    if target_type == 'binary':
        print(f"              Predicted")
        print(f"           {labels[0]:<12} {labels[1]:<12}")
        print(f"Actual {labels[0]:<6}: {cm[0][0]:5}        {cm[0][1]:5}")
        if len(cm) > 1:
            print(f"Actual {labels[1]:<6}: {cm[1][0]:5}        {cm[1][1]:5}")
    else:
        print(f"                  Predicted")
        print(f"           {labels[0]:<12} {labels[1]:<12} {labels[2]:<12}")
        for i, label in enumerate(labels):
            if i < len(cm):
                row = cm[i]
                print(f"Actual {label:<10}: {row[0]:5}        {row[1]:5}        {row[2]:5}")

    # Confidence analysis
    print(f"\n{'='*80}")
    print("CONFIDENCE ANALYSIS")
    print(f"{'='*80}")

    max_proba = y_proba.max(axis=1)

    print(f"\n{'Threshold':<15} {'Accuracy':<15} {'Coverage':<15} {'Count':<10}")
    print("-"*60)

    for threshold in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]:
        mask = max_proba >= threshold
        if mask.sum() > 0:
            conf_acc = accuracy_score(y_test[mask], y_pred[mask])
            coverage = mask.mean()
            count = mask.sum()

            indicator = ""
            if conf_acc >= 0.75:
                indicator = "✓✓ TARGET"
            elif conf_acc >= 0.70:
                indicator = "✓ CLOSE"

            print(f"{threshold:.0%}             {conf_acc:.2%}            {coverage:.1%}             {count:<10,} {indicator}")

    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_proba,
        'confusion_matrix': cm.tolist()
    }


def per_stock_analysis(tickers, y_test, y_pred, target_name):
    """Analyze performance per stock"""
    print(f"\n{'='*80}")
    print(f"PER-STOCK ACCURACY - {target_name}")
    print(f"{'='*80}")

    ticker_results = {}
    unique_tickers = sorted(set(tickers))

    for ticker in unique_tickers:
        mask = [t == ticker for t in tickers]
        if sum(mask) > 0:
            ticker_y_test = y_test[mask]
            ticker_y_pred = y_pred[mask]
            acc = accuracy_score(ticker_y_test, ticker_y_pred)
            ticker_results[ticker] = {
                'accuracy': acc,
                'samples': sum(mask)
            }

    # Sort by accuracy
    sorted_tickers = sorted(ticker_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

    print(f"\n{'Stock':<10} {'Accuracy':<12} {'Samples':<10}")
    print("-"*35)

    # Top 10
    print("\nTop 10 Best:")
    for ticker, metrics in sorted_tickers[:10]:
        indicator = "✓✓" if metrics['accuracy'] >= 0.75 else "✓" if metrics['accuracy'] >= 0.70 else ""
        print(f"{ticker:<10} {metrics['accuracy']:.2%}        {metrics['samples']:<10,} {indicator}")

    # Bottom 10
    print("\nBottom 10 Worst:")
    for ticker, metrics in sorted_tickers[-10:]:
        print(f"{ticker:<10} {metrics['accuracy']:.2%}        {metrics['samples']:<10}")

    # Overall stats
    accuracies = [m['accuracy'] for m in ticker_results.values()]
    print(f"\nOverall Statistics:")
    print(f"  Mean accuracy: {np.mean(accuracies):.2%}")
    print(f"  Median accuracy: {np.median(accuracies):.2%}")
    print(f"  Best: {max(accuracies):.2%}")
    print(f"  Worst: {min(accuracies):.2%}")
    print(f"  Stocks >75%: {sum(1 for a in accuracies if a >= 0.75)}/{len(accuracies)}")
    print(f"  Stocks >70%: {sum(1 for a in accuracies if a >= 0.70)}/{len(accuracies)}")


def main():
    print("="*80)
    print("COMPREHENSIVE MULTI-TARGET MODEL TESTING")
    print("="*80)

    # Load enhanced dataset
    DATA_PATH = Path('data/training_data_enhanced.parquet')

    if not DATA_PATH.exists():
        print(f"\nERROR: {DATA_PATH} not found!")
        print("Run 'python add_real_data.py' first to create enhanced dataset.")
        return

    print(f"\nLoading dataset: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    print(f"Total samples: {len(df):,}")
    print(f"Stocks: {df['ticker'].nunique()}")
    print(f"Features: {len(df.columns)}")

    # Create targets (same as training)
    print("\nCreating targets...")

    # Import target creation functions
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from train_rf_multi_target import create_volatility_target, create_relative_performance_target, create_breakout_target

    df = create_volatility_target(df, '1month')
    df = create_relative_performance_target(df, '1month')
    df = create_breakout_target(df)

    # Temporal split (same as training)
    print("\nTemporal split...")
    df = df.sort_values('date')
    n = len(df)
    train_end = int(n * 0.7)
    test_start = int(n * 0.85)

    test_df = df.iloc[test_start:]
    print(f"Test set: {len(test_df):,} samples")
    print(f"Date range: {test_df['date'].min()} to {test_df['date'].max()}")

    # Get features
    feature_cols = [c for c in df.columns
                    if not c.startswith('target_')
                    and c not in ['ticker', 'date', 'future_volatility', 'relative_perf_1month']]

    print(f"Features: {len(feature_cols)}")

    # Test each model
    TARGETS = [
        ('target_volatility_1month', 'Volatility', 'binary'),
        ('target_relative_1month', 'Relative Performance', '3-class'),
        ('target_breakout', 'Breakout/Consolidation', 'binary')
    ]

    results = {}

    for target_col, target_name, target_type in TARGETS:
        print(f"\n\n{'#'*80}")
        print(f"# {target_name.upper()}")
        print(f"{'#'*80}")

        # Load model
        model_path = MODEL_DIR / f'{target_col}.pkl'
        scaler_path = MODEL_DIR / f'{target_col}_scaler.pkl'

        if not model_path.exists() or not scaler_path.exists():
            print(f"Model not found: {model_path}")
            continue

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        print(f"✓ Loaded model and scaler")

        # Create test sequences
        print(f"\nCreating test sequences...")
        X_test, y_test, test_tickers, test_dates = create_sequences(
            test_df, feature_cols, SEQUENCE_LENGTH, target_col
        )

        print(f"Test sequences: {len(X_test):,}")

        # Evaluate
        result = evaluate_model(model, scaler, X_test, y_test, target_name, target_type)
        results[target_name] = result

        # Per-stock analysis
        per_stock_analysis(test_tickers, y_test, result['predictions'], target_name)

    # Overall summary
    print("\n\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    for target_name, result in results.items():
        print(f"\n{target_name}: {result['accuracy']:.2%}")

    avg_acc = np.mean([r['accuracy'] for r in results.values()])
    print(f"\nAverage Accuracy: {avg_acc:.2%}")

    if avg_acc >= 0.70:
        print("\n✓✓ SUCCESS: Models performing well!")
    elif avg_acc >= 0.60:
        print("\n✓ GOOD: Models showing useful signal")
    else:
        print("\n○ MODERATE: Room for improvement")

    # Save detailed results
    output_file = MODEL_DIR / 'test_results.json'
    test_results = {
        'test_date': pd.Timestamp.now().isoformat(),
        'test_samples': len(test_df),
        'results': {
            name: {'accuracy': float(res['accuracy'])}
            for name, res in results.items()
        }
    }

    with open(output_file, 'w') as f:
        json.dump(test_results, f, indent=2)

    print(f"\n✓ Detailed results saved to: {output_file}")
    print("="*80)


if __name__ == '__main__':
    main()
