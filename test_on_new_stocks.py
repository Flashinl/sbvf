#!/usr/bin/env python3
"""
Test on Completely New Stocks

Tests models on stocks they've NEVER seen during training
to verify true generalization capability.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import yfinance as yf
from datetime import datetime, timedelta
from tqdm import tqdm
from sklearn.metrics import accuracy_score

MODEL_DIR = Path('models/multi_target')
SEQUENCE_LENGTH = 30

# Stocks NOT in training data
NEW_STOCKS = [
    'SHOP', 'SQ', 'COIN', 'RBLX', 'ABNB',  # Tech/Growth
    'BA', 'CAT', 'DE', 'GE', 'HON',  # Industrials
    'JPM', 'BAC', 'WFC', 'GS', 'MS',  # Financials
    'PFE', 'JNJ', 'ABBV', 'MRK', 'LLY',  # Healthcare
    'XOM', 'CVX', 'COP', 'SLB', 'EOG'  # Energy
]


def fetch_stock_data(ticker, start_date, end_date):
    """Fetch and prepare features for a stock"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)

        if len(hist) < 100:
            return None

        # Get fundamental data
        info = stock.info

        fundamentals = {
            'fund_pe_ratio': info.get('trailingPE', np.nan),
            'fund_forward_pe': info.get('forwardPE', np.nan),
            'fund_peg_ratio': info.get('pegRatio', np.nan),
            'fund_eps': info.get('trailingEps', np.nan),
            'fund_profit_margin': info.get('profitMargins', np.nan),
            'fund_operating_margin': info.get('operatingMargins', np.nan),
            'fund_roe': info.get('returnOnEquity', np.nan),
            'fund_roa': info.get('returnOnAssets', np.nan),
            'fund_debt_to_equity': info.get('debtToEquity', np.nan),
            'fund_current_ratio': info.get('currentRatio', np.nan),
            'fund_quick_ratio': info.get('quickRatio', np.nan),
            'fund_book_value': info.get('bookValue', np.nan),
            'fund_price_to_book': info.get('priceToBook', np.nan),
            'fund_revenue_growth': info.get('revenueGrowth', np.nan),
            'fund_earnings_growth': info.get('earningsGrowth', np.nan),
            'fund_beta': info.get('beta', np.nan),
        }

        # Prepare DataFrame
        df = pd.DataFrame()
        df['ticker'] = ticker
        df['date'] = hist.index
        df['close'] = hist['Close'].values
        df['high'] = hist['High'].values
        df['low'] = hist['Low'].values
        df['volume'] = hist['Volume'].values

        # Add fundamentals
        for key, value in fundamentals.items():
            df[key] = value

        # Calculate technical features
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

        # Calculate targets (for ground truth)
        df['target_return_1month'] = df['close'].pct_change(20).shift(-20) * 100

        # Volatility target
        df['future_vol'] = df['return_1d'].shift(-1).rolling(20).std() * np.sqrt(252)
        df['target_volatility_1month'] = (df['future_vol'] > df['future_vol'].median()).astype(int)

        # Relative performance target (assume SPY return = 0.5% monthly for now)
        df['target_relative_1month'] = pd.cut(
            df['target_return_1month'],
            bins=[-np.inf, -2, 2, np.inf],
            labels=[0, 1, 2]
        ).astype(float)

        # Breakout target
        vol_20 = df['return_1d'].rolling(20).std()
        price_range = (df['high'].rolling(20).max() - df['low'].rolling(20).min()) / df['low'].rolling(20).min()
        vol_threshold = vol_20.quantile(0.4)
        range_threshold = price_range.quantile(0.4)
        df['target_breakout'] = ((vol_20 > vol_threshold) | (price_range > range_threshold)).astype(int)

        df = df.dropna()

        return df

    except Exception as e:
        print(f"  Error fetching {ticker}: {e}")
        return None


def create_sequences(df, feature_cols, seq_length, target_col):
    """Create sequences for prediction"""
    sequences = []
    targets = []

    if len(df) < seq_length + 20:
        return None, None

    # Get available feature columns (intersection)
    available_features = [col for col in feature_cols if col in df.columns]

    for i in range(len(df) - seq_length - 20):
        # Extract available features
        seq_data = df.iloc[i:i+seq_length][available_features].values.flatten()

        # Pad to match expected feature count
        expected_size = len(feature_cols) * seq_length
        if len(seq_data) < expected_size:
            seq_data = np.pad(seq_data, (0, expected_size - len(seq_data)), 'constant', constant_values=0)
        elif len(seq_data) > expected_size:
            seq_data = seq_data[:expected_size]

        # Get target
        target = df.iloc[i+seq_length-1][target_col]

        if not np.isnan(target) and target >= 0:
            sequences.append(seq_data)
            targets.append(int(target))

    if len(sequences) == 0:
        return None, None

    return np.array(sequences), np.array(targets)


def main():
    print("="*80)
    print("TESTING ON COMPLETELY NEW STOCKS")
    print("="*80)
    print(f"\nTesting on {len(NEW_STOCKS)} stocks NOT seen during training")

    # Load models
    print("\nLoading models...")
    models = {}
    scalers = {}

    targets = [
        ('target_volatility_1month', 'Volatility'),
        ('target_relative_1month', 'Relative Performance'),
        ('target_breakout', 'Breakout/Consolidation')
    ]

    for target_col, target_name in targets:
        model_path = MODEL_DIR / f'{target_col}.pkl'
        scaler_path = MODEL_DIR / f'{target_col}_scaler.pkl'

        if model_path.exists() and scaler_path.exists():
            models[target_col] = joblib.load(model_path)
            scalers[target_col] = joblib.load(scaler_path)
            print(f"  OK {target_name}")

    if not models:
        print("No models found! Train first.")
        return

    # Fetch data for new stocks
    print("\nFetching stock data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # 2 years

    stock_data = {}
    for ticker in tqdm(NEW_STOCKS, desc="Downloading"):
        df = fetch_stock_data(ticker, start_date, end_date)
        if df is not None:
            stock_data[ticker] = df

    print(f"\n* Successfully loaded {len(stock_data)} stocks")

    # Get training features
    training_data = pd.read_parquet('data/training_data_enhanced.parquet')
    feature_cols = [c for c in training_data.columns
                    if not c.startswith('target_')
                    and c not in ['ticker', 'date', 'future_volatility', 'relative_perf_1month']]

    print(f"Using {len(feature_cols)} features")

    # Test each model
    results = {}

    for target_col, target_name in targets:
        print(f"\n{'='*80}")
        print(f"TESTING: {target_name}")
        print(f"{'='*80}")

        model = models[target_col]
        scaler = scalers[target_col]

        all_predictions = []
        all_targets = []
        stock_accuracies = {}

        for ticker, df in stock_data.items():
            # Create sequences
            X, y = create_sequences(df, feature_cols, SEQUENCE_LENGTH, target_col)

            if X is None or len(X) < 10:
                print(f"  {ticker}: Skipped (insufficient data)")
                continue

            print(f"  {ticker}: {len(X)} sequences")

            # Preprocess
            # Match feature count
            if X.shape[1] < scaler.n_features_in_:
                padding = np.zeros((X.shape[0], scaler.n_features_in_ - X.shape[1]))
                X = np.hstack([X, padding])
            elif X.shape[1] > scaler.n_features_in_:
                X = X[:, :scaler.n_features_in_]

            X = np.nan_to_num(X, nan=0)
            X_scaled = scaler.transform(X)

            # Predict
            y_pred = model.predict(X_scaled)

            # Accuracy
            acc = accuracy_score(y, y_pred)
            stock_accuracies[ticker] = acc

            all_predictions.extend(y_pred)
            all_targets.extend(y)

        # Overall accuracy
        if len(all_predictions) > 0:
            overall_acc = accuracy_score(all_targets, all_predictions)

            print(f"\nOverall Accuracy: {overall_acc:.2%}")
            print(f"Test samples: {len(all_predictions):,}")

            if overall_acc >= 0.70:
                print("** EXCELLENT: Model generalizes well!")
            elif overall_acc >= 0.60:
                print("* GOOD: Decent generalization")
            elif overall_acc >= 0.50:
                print("○ MODERATE: Some signal")
            else:
                print("~ WEAK: Poor generalization")

            # Per-stock results
            print(f"\nPer-Stock Accuracy:")
            print(f"{'Stock':<10} {'Accuracy':<12} {'Samples':<10}")
            print("-"*35)

            for ticker in sorted(stock_accuracies.keys(), key=lambda t: stock_accuracies[t], reverse=True):
                acc = stock_accuracies[ticker]
                samples = sum(1 for i, t in enumerate(stock_data[ticker]['ticker']) if i < len(stock_data[ticker]) - SEQUENCE_LENGTH - 20)
                indicator = "**" if acc >= 0.75 else "*" if acc >= 0.70 else ""
                print(f"{ticker:<10} {acc:.2%}        {samples:<10,} {indicator}")

            results[target_name] = {
                'accuracy': overall_acc,
                'stocks_tested': len(stock_accuracies),
                'total_samples': len(all_predictions)
            }

    # Summary
    print("\n\n" + "="*80)
    print("GENERALIZATION SUMMARY")
    print("="*80)

    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Accuracy: {result['accuracy']:.2%}")
        print(f"  Stocks tested: {result['stocks_tested']}")
        print(f"  Total samples: {result['total_samples']:,}")

    avg_acc = np.mean([r['accuracy'] for r in results.values()])
    print(f"\nAverage Generalization Accuracy: {avg_acc:.2%}")

    if avg_acc >= 0.65:
        print("\n*** SUCCESS: Models generalize to new stocks!")
    elif avg_acc >= 0.55:
        print("\n* GOOD: Models show useful generalization")
    else:
        print("\n○ MODERATE: Limited generalization")

    print("="*80)


if __name__ == '__main__':
    main()
