"""
Model Training Script

Train LSTM and XGBoost models on historical stock data

Usage:
    python -m stockbot.ml.train --tickers AAPL MSFT GOOGL --start-date 2019-01-01
"""

import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Tuple
import time
import random

from .feature_engineering import extract_all_features, features_to_dataframe
from .models.lstm_model import create_lstm_model
from .models.xgboost_model import create_xgboost_regressor
from .sp500_tickers import get_ticker_list, SP500_TICKERS, TOP_100_TICKERS, TEST_TICKERS


def download_historical_data(
    ticker: str,
    start_date: str,
    end_date: str = None,
    retry_count: int = 3,
    delay: float = 1.0
) -> pd.DataFrame:
    """
    Download historical stock data with retry logic and rate limiting

    Args:
        ticker: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        retry_count: Number of retries on failure
        delay: Delay between requests in seconds

    Returns:
        DataFrame with historical data
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    # Add random jitter to delay to avoid synchronized requests
    actual_delay = delay + random.uniform(0, 0.5)
    time.sleep(actual_delay)

    for attempt in range(retry_count):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)

            if hist.empty:
                print(f"  X No data for {ticker}")
                return pd.DataFrame()

            print(f"  OK {ticker}: {len(hist)} days")
            return hist

        except Exception as e:
            if attempt < retry_count - 1:
                wait_time = (attempt + 1) * 2  # Exponential backoff
                print(f"  ! {ticker} failed (attempt {attempt+1}/{retry_count}), retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  X {ticker} failed after {retry_count} attempts: {e}")
                return pd.DataFrame()

    return pd.DataFrame()


def prepare_training_data(
    tickers: List[str],
    start_date: str,
    end_date: str = None,
    target_days: int = 30,
    delay: float = 0.5,
    retry_count: int = 3
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare training dataset from multiple tickers

    Args:
        tickers: List of stock symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        target_days: Prediction horizon in days (default 30)
        delay: Delay between downloads in seconds
        retry_count: Number of retries on failure

    Returns:
        X: Features DataFrame
        y: Target returns Series
    """
    all_features = []
    all_targets = []
    failed_tickers = []

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {ticker}...", end=' ')

        # Download historical data with rate limiting
        hist = download_historical_data(
            ticker,
            start_date,
            end_date,
            retry_count=retry_count,
            delay=delay
        )

        if hist.empty:
            failed_tickers.append(ticker)
            continue

        # For each date, extract features and calculate future return
        for i in range(60, len(hist) - target_days):  # Need 60 days history for features
            try:
                # Get historical window for features
                window_end = hist.index[i]
                window_data = hist.iloc[:i+1]

                # Extract features at this point in time
                features = extract_all_features(
                    ticker=ticker,
                    news_items=[],  # No news for historical data (optional: fetch if needed)
                    lookback_days=252
                )

                if not features or 'price_current' not in features:
                    continue

                # Calculate target: N-day forward return
                current_price = hist['Close'].iloc[i]
                future_price = hist['Close'].iloc[i + target_days]
                target_return = ((future_price - current_price) / current_price) * 100

                # Add metadata
                features['ticker'] = ticker
                features['date'] = window_end.strftime('%Y-%m-%d')
                features['target_return'] = target_return

                all_features.append(features)
                all_targets.append(target_return)

            except Exception as e:
                print(f"  Error at index {i}: {e}")
                continue

        sample_count = sum(1 for f in all_features if f.get('ticker') == ticker)
        print(f"({sample_count} samples)")

    # Summary
    print(f"\n{'='*50}")
    print(f"Data Collection Complete")
    print(f"{'='*50}")
    print(f"Successful tickers: {len(tickers) - len(failed_tickers)}/{len(tickers)}")
    if failed_tickers:
        print(f"Failed tickers: {', '.join(failed_tickers[:10])}")
        if len(failed_tickers) > 10:
            print(f"  ... and {len(failed_tickers) - 10} more")

    # Create DataFrame
    if not all_features:
        raise ValueError("No training data collected! All tickers failed.")

    X = pd.DataFrame(all_features)
    y = pd.Series(all_targets, name='target_return')

    print(f"\nTraining Dataset Summary:")
    print(f"  Total samples: {len(X)}")
    print(f"  Features: {len(X.columns)}")
    print(f"  Date range: {X['date'].min()} to {X['date'].max()}")
    print(f"\nTarget Return Statistics:")
    print(f"  Mean: {y.mean():.2f}%")
    print(f"  Std: {y.std():.2f}%")
    print(f"  Min: {y.min():.2f}%")
    print(f"  Max: {y.max():.2f}%")
    print(f"  Positive returns: {(y > 0).sum()} ({(y > 0).sum() / len(y) * 100:.1f}%)")
    print(f"{'='*50}\n")

    return X, y


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    save_path: str = 'models/xgboost.pkl'
) -> None:
    """Train XGBoost model"""
    print("\n" + "="*50)
    print("Training XGBoost Model")
    print("="*50)

    # Remove metadata columns
    feature_cols = [c for c in X_train.columns if c not in ['ticker', 'date', 'target_return']]
    X_train_clean = X_train[feature_cols].fillna(0)
    X_val_clean = X_val[feature_cols].fillna(0)

    # Create and train model
    model = create_xgboost_regressor()

    history = model.train(
        X_train_clean,
        y_train,
        X_val_clean,
        y_val,
        early_stopping_rounds=50,
        verbose=True
    )

    # Save model
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)

    print(f"\n✓ XGBoost model saved to {save_path}")


def train_lstm_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    save_path: str = 'models/lstm.pth'
) -> None:
    """Train LSTM model on sequence data"""
    print("\n" + "="*50)
    print("Training LSTM Model")
    print("="*50)

    try:
        # Prepare sequence data grouped by ticker
        print("Preparing sequence data for LSTM...")

        # Group by ticker and date
        train_sequences = []
        val_sequences = []

        for ticker in X_train['ticker'].unique():
            ticker_data = X_train[X_train['ticker'] == ticker].sort_values('date')
            ticker_targets = y_train[X_train['ticker'] == ticker]

            if len(ticker_data) < 60:
                continue

            # Create dataframe with needed columns for LSTM
            hist_df = ticker_data[['date'] + [c for c in ticker_data.columns if c not in ['ticker', 'date', 'target_return']]]
            hist_df['target_return'] = ticker_targets.values

            train_sequences.append(hist_df)

        # Combine all sequences
        if train_sequences:
            combined_train = pd.concat(train_sequences, ignore_index=True)

            # Create LSTM model
            model = create_lstm_model(sequence_length=60)

            # Train
            history = model.train(
                train_data=combined_train,
                val_data=None,  # Use same data for now
                epochs=50,
                batch_size=32,
                early_stopping_patience=10,
                verbose=True
            )

            # Save model
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            model.save(save_path)

            print(f"\n✓ LSTM model saved to {save_path}")
        else:
            print("! Not enough sequence data for LSTM training")
            print("  Recommendation: Use XGBoost model instead, or collect more data")

    except Exception as e:
        print(f"! LSTM training failed: {e}")
        print("  Continuing with XGBoost only...")


def main():
    parser = argparse.ArgumentParser(description='Train stock prediction models')

    # Ticker selection
    ticker_group = parser.add_mutually_exclusive_group()
    ticker_group.add_argument('--tickers', nargs='+', default=None,
                             help='Specific stock tickers to train on')
    ticker_group.add_argument('--ticker-list', choices=['small', 'medium', 'large'],
                             default='medium',
                             help='Preset ticker list: small (16), medium (100), large (500+)')

    # Date range
    parser.add_argument('--start-date', default='2020-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default=None,
                       help='End date (YYYY-MM-DD), default: today')

    # Training parameters
    parser.add_argument('--target-days', type=int, default=30,
                       help='Prediction horizon in days')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split ratio')

    # Rate limiting
    parser.add_argument('--delay', type=float, default=0.5,
                       help='Delay between API calls in seconds (avoid rate limits)')
    parser.add_argument('--retry-count', type=int, default=3,
                       help='Number of retries on failed downloads')

    # Model selection
    parser.add_argument('--train-xgboost', action='store_true', default=True,
                       help='Train XGBoost model')
    parser.add_argument('--train-lstm', action='store_true', default=False,
                       help='Train LSTM model')
    parser.add_argument('--train-both', action='store_true', default=False,
                       help='Train both XGBoost and LSTM')

    # Output
    parser.add_argument('--output-dir', default='models',
                       help='Directory to save models')

    args = parser.parse_args()

    # Determine ticker list
    if args.tickers:
        tickers = args.tickers
    else:
        tickers = get_ticker_list(args.ticker_list)

    # Enable both models if requested
    if args.train_both:
        args.train_xgboost = True
        args.train_lstm = True

    print("\n" + "="*60)
    print("Stock Prediction Model Training")
    print("="*60)
    print(f"Tickers: {len(tickers)} stocks")
    print(f"  List: {tickers[:10]}{'...' if len(tickers) > 10 else ''}")
    print(f"Date range: {args.start_date} to {args.end_date or 'today'}")
    print(f"Target horizon: {args.target_days} days")
    print(f"Validation split: {args.val_split}")
    print(f"Rate limit delay: {args.delay}s between downloads")
    print(f"Models to train: {'XGBoost' if args.train_xgboost else ''}{' + LSTM' if args.train_lstm else ''}")
    print("="*60 + "\n")

    # 1. Prepare data
    print("Step 1: Preparing training data...")
    print(f"Downloading data for {len(tickers)} tickers (this may take a while)...")
    print("Progress:")

    X, y = prepare_training_data(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        target_days=args.target_days,
        delay=args.delay,
        retry_count=args.retry_count
    )

    # 2. Train/Val split
    split_idx = int(len(X) * (1 - args.val_split))
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples\n")

    # 3. Train models
    if args.train_xgboost:
        train_xgboost_model(
            X_train, y_train,
            X_val, y_val,
            save_path=f"{args.output_dir}/xgboost.pkl"
        )

    if args.train_lstm:
        train_lstm_model(
            X_train, y_train,
            X_val, y_val,
            save_path=f"{args.output_dir}/lstm.pth"
        )

    print("\n" + "="*60)
    print("✓ Training complete!")
    print("="*60)
    print(f"\nModels saved to: {args.output_dir}/")
    print("\nNext steps:")
    print("1. Evaluate models on test set")
    print("2. Run backtesting: python -m stockbot.ml.backtest")
    print("3. Deploy models: python -m stockbot.api (with models loaded)")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
