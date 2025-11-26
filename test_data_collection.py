"""
Quick test to verify data collection works before training

Run this to ensure everything is ready:
python test_data_collection.py
"""

import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

print("\n" + "="*60)
print("Testing Data Collection for Model Training")
print("="*60 + "\n")

# Test 1: Download historical data
print("Test 1: Downloading historical data from Yahoo Finance...")
test_tickers = ['AAPL', 'MSFT', 'GOOGL']
start_date = '2023-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')

success_count = 0
for ticker in test_tickers:
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)

        if len(hist) > 0:
            print(f"  OK: {ticker} - {len(hist)} days ({hist.index[0].strftime('%Y-%m-%d')} to {hist.index[-1].strftime('%Y-%m-%d')})")
            success_count += 1
        else:
            print(f"  FAIL: {ticker} - No data downloaded")
    except Exception as e:
        print(f"  ERROR: {ticker} - {e}")

print(f"\nResult: {success_count}/{len(test_tickers)} tickers successful")

# Test 2: Extract features
print("\n" + "-"*60)
print("Test 2: Testing feature extraction...")

try:
    from stockbot.ml.feature_engineering import extract_all_features

    features = extract_all_features(ticker='AAPL', news_items=[], lookback_days=252)

    if features:
        print(f"  OK: Extracted {len(features)} features")
        print(f"  Sample features: {list(features.keys())[:10]}")

        # Check critical features
        critical = ['price_current', 'rsi_14', 'pe_ratio', 'market_cap']
        missing = [f for f in critical if f not in features]

        if missing:
            print(f"  WARNING: Missing critical features: {missing}")
        else:
            print(f"  OK: All critical features present")
    else:
        print(f"  FAIL: No features extracted")

except Exception as e:
    print(f"  ERROR: {e}")

# Test 3: Check ML dependencies
print("\n" + "-"*60)
print("Test 3: Checking ML dependencies...")

dependencies = {
    'xgboost': 'XGBoost for gradient boosting',
    'torch': 'PyTorch for LSTM',
    'sklearn': 'Scikit-learn for utilities',
    'numpy': 'NumPy for numerical operations',
    'pandas': 'Pandas for data handling'
}

missing_deps = []
for dep, description in dependencies.items():
    try:
        module = __import__(dep)
        version = getattr(module, '__version__', 'unknown')
        print(f"  OK: {dep} {version} - {description}")
    except ImportError:
        print(f"  MISSING: {dep} - {description}")
        missing_deps.append(dep)

# Optional dependencies
optional_deps = {'shap': 'SHAP for explainability'}
for dep, description in optional_deps.items():
    try:
        module = __import__(dep)
        version = getattr(module, '__version__', 'unknown')
        print(f"  OK: {dep} {version} - {description} (optional)")
    except ImportError:
        print(f"  MISSING: {dep} - {description} (optional, but recommended)")

# Summary
print("\n" + "="*60)
print("Summary")
print("="*60)

if success_count == len(test_tickers) and not missing_deps:
    print("\nSTATUS: READY TO TRAIN!")
    print("\nNext steps:")
    print("1. Run training script:")
    print("   python -m stockbot.ml.train --tickers AAPL MSFT GOOGL AMZN TSLA --start-date 2020-01-01")
    print("\n2. Training will:")
    print("   - Download historical data from Yahoo Finance")
    print("   - Extract 50+ features per date")
    print("   - Calculate 30-day forward returns")
    print("   - Train XGBoost model")
    print("   - Save to models/xgboost.pkl")
    print("\n3. Expected time: 10-30 minutes depending on tickers and date range")
else:
    print("\nSTATUS: NOT READY")
    if missing_deps:
        print(f"\nMissing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install " + " ".join(missing_deps))
    if success_count < len(test_tickers):
        print("\nData download failed - check internet connection")

print("="*60 + "\n")
