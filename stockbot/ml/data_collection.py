"""
Enhanced Data Collection for Multi-Horizon Stock Prediction

Collects historical OHLCV data and computes labels for multiple prediction horizons:
- 1 week (5 trading days)
- 1 month (21 trading days)
- 3 months (63 trading days)

Each horizon has:
- Classification label: BUY/HOLD/SELL based on forward return thresholds
- Regression target: actual forward return percentage
- Probability distribution for calibration

Thresholds:
- BUY: forward return > +3%
- HOLD: forward return between -3% and +3%
- SELL: forward return < -3%
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from pathlib import Path
import time
import random
from dataclasses import dataclass


@dataclass
class HorizonConfig:
    """Configuration for a prediction horizon"""
    name: str
    days: int
    buy_threshold: float = 3.0  # %
    sell_threshold: float = -3.0  # %


# Define prediction horizons
HORIZONS = [
    HorizonConfig(name="1week", days=5, buy_threshold=3.0, sell_threshold=-3.0),
    HorizonConfig(name="1month", days=21, buy_threshold=3.0, sell_threshold=-3.0),
    HorizonConfig(name="3month", days=63, buy_threshold=3.0, sell_threshold=-3.0),
]


def compute_forward_return(prices: pd.Series, current_idx: int, horizon_days: int) -> Optional[float]:
    """
    Compute forward return from current index

    Args:
        prices: Price series
        current_idx: Current index position
        horizon_days: Number of days to look forward

    Returns:
        Forward return percentage, or None if not enough data
    """
    if current_idx + horizon_days >= len(prices):
        return None

    current_price = prices.iloc[current_idx]
    future_price = prices.iloc[current_idx + horizon_days]

    if current_price == 0 or np.isnan(current_price) or np.isnan(future_price):
        return None

    return ((future_price - current_price) / current_price) * 100


def classify_return(return_pct: float, horizon: HorizonConfig) -> str:
    """
    Classify return into BUY/HOLD/SELL signal

    Args:
        return_pct: Forward return percentage
        horizon: Horizon configuration with thresholds

    Returns:
        'BUY', 'HOLD', or 'SELL'
    """
    if return_pct > horizon.buy_threshold:
        return 'BUY'
    elif return_pct < horizon.sell_threshold:
        return 'SELL'
    else:
        return 'HOLD'


def download_stock_data(
    ticker: str,
    start_date: str,
    end_date: Optional[str] = None,
    retry_count: int = 3,
    delay: float = 0.5
) -> pd.DataFrame:
    """
    Download historical OHLCV data with retry logic

    Args:
        ticker: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), defaults to today
        retry_count: Number of retries on failure
        delay: Base delay between requests (seconds)

    Returns:
        DataFrame with OHLCV data
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    # Add jitter to avoid rate limits
    actual_delay = delay + random.uniform(0, 0.3)
    time.sleep(actual_delay)

    for attempt in range(retry_count):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date, auto_adjust=False)

            if hist.empty:
                return pd.DataFrame()

            # Rename columns to standard format
            hist = hist.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # Keep only OHLCV
            hist = hist[['open', 'high', 'low', 'close', 'volume']].copy()
            hist['date'] = hist.index
            hist.reset_index(drop=True, inplace=True)

            return hist

        except Exception as e:
            if attempt < retry_count - 1:
                wait_time = (attempt + 1) * 2
                time.sleep(wait_time)
            else:
                print(f"  Failed to download {ticker}: {e}")
                return pd.DataFrame()

    return pd.DataFrame()


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators from OHLCV data

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        DataFrame with additional technical indicator columns
    """
    df = df.copy()

    # RSI (14-day)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Bollinger Bands (20-day, 2 std)
    df['bb_mid'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_mid'] + (2 * bb_std)
    df['bb_lower'] = df['bb_mid'] - (2 * bb_std)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

    # ATR (14-day)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(14).mean()

    # SMAs
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()

    # Volume indicators
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']

    # Returns
    df['return_1d'] = df['close'].pct_change() * 100
    df['return_5d'] = df['close'].pct_change(5) * 100
    df['return_20d'] = df['close'].pct_change(20) * 100

    return df


def fetch_vix() -> float:
    """Fetch current VIX (market volatility index)"""
    try:
        vix = yf.Ticker("^VIX")
        vix_hist = vix.history(period="5d")
        if not vix_hist.empty:
            return float(vix_hist['Close'].iloc[-1])
    except:
        pass
    return 15.0  # Default value


def fetch_sector_etf_correlations(ticker: str, sector_etfs: Dict[str, str]) -> Dict[str, float]:
    """
    Fetch correlations with sector ETFs

    Args:
        ticker: Stock symbol
        sector_etfs: Dict mapping sector name to ETF symbol

    Returns:
        Dict mapping sector to correlation coefficient
    """
    correlations = {}

    try:
        stock = yf.Ticker(ticker)
        stock_hist = stock.history(period="3mo")['Close']

        for sector_name, etf_symbol in sector_etfs.items():
            try:
                etf = yf.Ticker(etf_symbol)
                etf_hist = etf.history(period="3mo")['Close']

                # Align dates and compute correlation
                combined = pd.DataFrame({
                    'stock': stock_hist,
                    'etf': etf_hist
                }).dropna()

                if len(combined) > 20:
                    corr = combined['stock'].corr(combined['etf'])
                    correlations[f'corr_{sector_name}'] = float(corr)
                else:
                    correlations[f'corr_{sector_name}'] = 0.0
            except:
                correlations[f'corr_{sector_name}'] = 0.0
    except:
        for sector_name in sector_etfs.keys():
            correlations[f'corr_{sector_name}'] = 0.0

    return correlations


def prepare_dataset(
    tickers: List[str],
    start_date: str,
    end_date: Optional[str] = None,
    min_history_days: int = 250,
    include_sector_correlation: bool = False,
    delay: float = 0.5,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Prepare complete training dataset with multi-horizon labels

    Args:
        tickers: List of stock symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        min_history_days: Minimum days of history required
        include_sector_correlation: Whether to include sector ETF correlations
        delay: Delay between API calls
        verbose: Print progress

    Returns:
        DataFrame with features and multi-horizon labels
    """
    all_samples = []
    failed_tickers = []

    # Sector ETFs for correlation
    sector_etfs = {
        'tech': 'XLK',
        'health': 'XLV',
        'finance': 'XLF',
        'energy': 'XLE',
        'consumer': 'XLY'
    }

    # Fetch VIX once (market-wide volatility)
    vix_value = fetch_vix()

    for idx, ticker in enumerate(tickers, 1):
        if verbose:
            print(f"[{idx}/{len(tickers)}] Processing {ticker}...", end=' ')

        # Download data
        df = download_stock_data(ticker, start_date, end_date, delay=delay)

        if df.empty or len(df) < min_history_days:
            if verbose:
                print(f"Insufficient data ({len(df)} days)")
            failed_tickers.append(ticker)
            continue

        # Compute technical indicators
        df = compute_technical_indicators(df)

        # Get sector correlations (optional, slower)
        if include_sector_correlation:
            sector_corr = fetch_sector_etf_correlations(ticker, sector_etfs)
        else:
            sector_corr = {f'corr_{s}': 0.0 for s in sector_etfs.keys()}

        # Create samples for each valid date
        max_horizon = max(h.days for h in HORIZONS)
        sample_count = 0

        for i in range(min_history_days, len(df) - max_horizon):
            sample = {}

            # Metadata
            sample['ticker'] = ticker
            sample['date'] = df['date'].iloc[i]

            # OHLCV features (current day)
            sample['open'] = df['open'].iloc[i]
            sample['high'] = df['high'].iloc[i]
            sample['low'] = df['low'].iloc[i]
            sample['close'] = df['close'].iloc[i]
            sample['volume'] = df['volume'].iloc[i]

            # Technical indicators (current day)
            for col in ['rsi_14', 'macd', 'macd_signal', 'macd_hist',
                       'bb_mid', 'bb_upper', 'bb_lower', 'bb_width',
                       'atr', 'sma_50', 'sma_200',
                       'volume_sma_20', 'volume_ratio',
                       'return_1d', 'return_5d', 'return_20d']:
                sample[col] = df[col].iloc[i]

            # Market context
            sample['vix'] = vix_value
            sample.update(sector_corr)

            # Multi-horizon labels
            all_labels_valid = True
            for horizon in HORIZONS:
                fwd_return = compute_forward_return(df['close'], i, horizon.days)

                if fwd_return is None:
                    all_labels_valid = False
                    break

                # Regression target
                sample[f'target_return_{horizon.name}'] = fwd_return

                # Classification label
                signal = classify_return(fwd_return, horizon)
                sample[f'target_signal_{horizon.name}'] = signal

                # Encoded class (0=SELL, 1=HOLD, 2=BUY)
                signal_map = {'SELL': 0, 'HOLD': 1, 'BUY': 2}
                sample[f'target_class_{horizon.name}'] = signal_map[signal]

            if all_labels_valid:
                all_samples.append(sample)
                sample_count += 1

        if verbose:
            print(f"✓ {sample_count} samples")

    # Create DataFrame
    if not all_samples:
        raise ValueError("No valid samples collected!")

    dataset = pd.DataFrame(all_samples)

    # Summary
    if verbose:
        print(f"\n{'='*60}")
        print(f"Dataset Collection Complete")
        print(f"{'='*60}")
        print(f"Successful tickers: {len(tickers) - len(failed_tickers)}/{len(tickers)}")
        print(f"Total samples: {len(dataset)}")
        print(f"Date range: {dataset['date'].min()} to {dataset['date'].max()}")

        # Class distribution per horizon
        print(f"\nClass Distribution:")
        for horizon in HORIZONS:
            signal_col = f'target_signal_{horizon.name}'
            counts = dataset[signal_col].value_counts()
            print(f"  {horizon.name}:")
            for signal in ['SELL', 'HOLD', 'BUY']:
                count = counts.get(signal, 0)
                pct = (count / len(dataset)) * 100
                print(f"    {signal}: {count} ({pct:.1f}%)")

        print(f"{'='*60}\n")

    return dataset


def save_dataset(dataset: pd.DataFrame, filepath: str):
    """Save dataset to CSV or parquet"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if filepath.suffix == '.parquet':
        dataset.to_parquet(filepath, index=False)
    else:
        dataset.to_csv(filepath, index=False)

    print(f"Dataset saved to: {filepath}")


def load_dataset(filepath: str) -> pd.DataFrame:
    """Load dataset from CSV or parquet"""
    filepath = Path(filepath)

    if filepath.suffix == '.parquet':
        return pd.read_parquet(filepath)
    else:
        return pd.read_csv(filepath, parse_dates=['date'])


if __name__ == '__main__':
    # Example usage
    from .sp500_tickers import TEST_TICKERS, TOP_100_TICKERS

    print("Multi-Horizon Data Collection Script")
    print("="*60)

    # Use test tickers for demo
    tickers = TEST_TICKERS[:5]  # Start small

    # Prepare dataset
    dataset = prepare_dataset(
        tickers=tickers,
        start_date='2020-01-01',
        end_date=None,
        include_sector_correlation=False,  # Set True for full features
        delay=0.5,
        verbose=True
    )

    # Save
    save_dataset(dataset, 'data/training_data_multihorizon.csv')

    print("\nFirst 3 samples:")
    print(dataset.head(3))
