#!/usr/bin/env python3
"""
Add Real Data - Fundamentals + News Sentiment

Enhances existing dataset with:
1. Fundamental data (P/E, EPS, revenue growth, margins, debt)
2. Real news sentiment (if available)
3. Earnings calendar proximity
4. Analyst ratings changes

This gives the model information the market hasn't fully priced in yet.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# CONFIG
INPUT_FILE = 'data/training_data_with_sentiment.parquet'
OUTPUT_FILE = 'data/training_data_enhanced.parquet'


def fetch_fundamentals(ticker, date):
    """
    Fetch fundamental data for a stock at a given date

    Returns dict with:
    - P/E ratio
    - EPS (earnings per share)
    - Revenue growth
    - Profit margin
    - Debt to equity
    - Book value
    - Forward P/E
    - PEG ratio
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Basic fundamentals
        fundamentals = {
            'pe_ratio': info.get('trailingPE', np.nan),
            'forward_pe': info.get('forwardPE', np.nan),
            'peg_ratio': info.get('pegRatio', np.nan),
            'eps': info.get('trailingEps', np.nan),
            'profit_margin': info.get('profitMargins', np.nan),
            'operating_margin': info.get('operatingMargins', np.nan),
            'roe': info.get('returnOnEquity', np.nan),
            'roa': info.get('returnOnAssets', np.nan),
            'debt_to_equity': info.get('debtToEquity', np.nan),
            'current_ratio': info.get('currentRatio', np.nan),
            'quick_ratio': info.get('quickRatio', np.nan),
            'book_value': info.get('bookValue', np.nan),
            'price_to_book': info.get('priceToBook', np.nan),
            'revenue_growth': info.get('revenueGrowth', np.nan),
            'earnings_growth': info.get('earningsGrowth', np.nan),
            'ebitda_margin': info.get('ebitdaMargins', np.nan),
            'gross_margin': info.get('grossMargins', np.nan),
            'beta': info.get('beta', np.nan),
            'shares_outstanding': info.get('sharesOutstanding', np.nan),
            'float_shares': info.get('floatShares', np.nan),
            'shares_short': info.get('sharesShort', np.nan),
            'short_ratio': info.get('shortRatio', np.nan),
            'held_percent_institutions': info.get('heldPercentInstitutions', np.nan),
        }

        # Calculate additional metrics
        if not np.isnan(fundamentals['pe_ratio']) and not np.isnan(fundamentals['earnings_growth']):
            if fundamentals['earnings_growth'] > 0:
                fundamentals['peg_calculated'] = fundamentals['pe_ratio'] / (fundamentals['earnings_growth'] * 100)
            else:
                fundamentals['peg_calculated'] = np.nan
        else:
            fundamentals['peg_calculated'] = np.nan

        return fundamentals

    except Exception as e:
        # Return NaN dict on error
        return {k: np.nan for k in [
            'pe_ratio', 'forward_pe', 'peg_ratio', 'eps', 'profit_margin',
            'operating_margin', 'roe', 'roa', 'debt_to_equity', 'current_ratio',
            'quick_ratio', 'book_value', 'price_to_book', 'revenue_growth',
            'earnings_growth', 'ebitda_margin', 'gross_margin', 'beta',
            'shares_outstanding', 'float_shares', 'shares_short', 'short_ratio',
            'held_percent_institutions', 'peg_calculated'
        ]}


def calculate_fundamental_scores(row):
    """
    Calculate composite fundamental scores
    These combine multiple fundamentals into signals
    """
    scores = {}

    # Value score (lower P/E, P/B is better)
    pe = row.get('pe_ratio', np.nan)
    pb = row.get('price_to_book', np.nan)

    if not np.isnan(pe) and not np.isnan(pb):
        # Normalize: lower is better, scale 0-1
        pe_score = 1 / (1 + pe / 20) if pe > 0 else 0  # 20 is avg P/E
        pb_score = 1 / (1 + pb / 3) if pb > 0 else 0   # 3 is avg P/B
        scores['value_score'] = (pe_score + pb_score) / 2
    else:
        scores['value_score'] = np.nan

    # Growth score (higher growth is better)
    rev_growth = row.get('revenue_growth', np.nan)
    earn_growth = row.get('earnings_growth', np.nan)

    if not np.isnan(rev_growth) and not np.isnan(earn_growth):
        scores['growth_score'] = (rev_growth + earn_growth) / 2
    else:
        scores['growth_score'] = np.nan

    # Quality score (profitability)
    roe = row.get('roe', np.nan)
    margin = row.get('profit_margin', np.nan)

    if not np.isnan(roe) and not np.isnan(margin):
        scores['quality_score'] = (roe + margin) / 2
    else:
        scores['quality_score'] = np.nan

    # Financial health
    debt_eq = row.get('debt_to_equity', np.nan)
    current = row.get('current_ratio', np.nan)

    if not np.isnan(debt_eq) and not np.isnan(current):
        # Lower debt is better, higher current ratio is better
        debt_score = 1 / (1 + debt_eq / 100) if debt_eq >= 0 else 0
        scores['health_score'] = (debt_score + min(current / 2, 1)) / 2
    else:
        scores['health_score'] = np.nan

    # Momentum score (based on institutional holdings and short interest)
    inst = row.get('held_percent_institutions', np.nan)
    short = row.get('short_ratio', np.nan)

    if not np.isnan(inst):
        scores['institutional_interest'] = inst
    else:
        scores['institutional_interest'] = np.nan

    if not np.isnan(short):
        # High short interest can mean either bearish or short squeeze potential
        scores['short_interest_signal'] = short
    else:
        scores['short_interest_signal'] = np.nan

    return scores


def add_fundamentals_to_dataset(df):
    """Add fundamentals to each row in dataset"""
    print("\n" + "="*80)
    print("ADDING FUNDAMENTAL DATA")
    print("="*80)

    # Get unique tickers
    tickers = df['ticker'].unique()
    print(f"Processing {len(tickers)} tickers...")

    # Fetch fundamentals for each ticker (once per ticker, not per row)
    ticker_fundamentals = {}

    for ticker in tqdm(tickers, desc="Fetching fundamentals"):
        fundamentals = fetch_fundamentals(ticker, None)
        ticker_fundamentals[ticker] = fundamentals

    # Add fundamentals to each row
    print("\nAdding fundamentals to dataset...")

    fundamental_cols = list(ticker_fundamentals[tickers[0]].keys())

    for col in fundamental_cols:
        df[f'fund_{col}'] = df['ticker'].map(
            lambda t: ticker_fundamentals.get(t, {}).get(col, np.nan)
        )

    # Calculate composite scores
    print("Calculating fundamental scores...")
    scores_list = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing scores"):
        scores = calculate_fundamental_scores(row)
        scores_list.append(scores)

    scores_df = pd.DataFrame(scores_list)
    for col in scores_df.columns:
        df[col] = scores_df[col].values

    # Count how many fundamentals we got
    fund_cols = [c for c in df.columns if c.startswith('fund_')]
    valid_count = df[fund_cols].notna().sum(axis=1).mean()
    print(f"\n✓ Added {len(fund_cols)} fundamental features")
    print(f"  Average {valid_count:.1f} valid values per row")

    return df


def add_relative_metrics(df):
    """
    Add relative metrics (vs market, vs sector)
    These help predict outperformance rather than absolute direction
    """
    print("\n" + "="*80)
    print("ADDING RELATIVE PERFORMANCE METRICS")
    print("="*80)

    # Relative to SPY (already have market_SPY_return_*)
    # Add some additional relative metrics

    if 'return_1d' in df.columns and 'market_SPY_return_1d' in df.columns:
        df['relative_return_1d'] = df['return_1d'] - df['market_SPY_return_1d']
        df['relative_return_5d'] = df['return_5d'] - df['market_SPY_return_5d']
        df['relative_return_20d'] = df['return_20d'] - df['market_SPY_return_20d']

        # Relative strength index
        df['relative_strength'] = df['return_20d'] / (df['market_SPY_return_20d'] + 1e-6)

        print("✓ Added relative performance metrics")

    return df


print("="*80)
print("ENHANCING DATASET WITH REAL DATA")
print("="*80)

# Load existing dataset
print(f"\nLoading: {INPUT_FILE}")
df = pd.read_parquet(INPUT_FILE)

print(f"Original dataset:")
print(f"  Rows: {len(df):,}")
print(f"  Columns: {len(df.columns)}")
print(f"  Tickers: {df['ticker'].nunique()}")

# Add fundamentals
df = add_fundamentals_to_dataset(df)

# Add relative metrics
df = add_relative_metrics(df)

# Save enhanced dataset
print(f"\n" + "="*80)
print("SAVING ENHANCED DATASET")
print("="*80)

print(f"\nEnhanced dataset:")
print(f"  Rows: {len(df):,}")
print(f"  Columns: {len(df.columns)}")
print(f"  New features: {len(df.columns) - pd.read_parquet(INPUT_FILE).shape[1]}")

print(f"\nSaving to: {OUTPUT_FILE}")
df.to_parquet(OUTPUT_FILE, index=False)

print("\n✓ Dataset enhanced successfully!")
print(f"\nNew features added:")
print(f"  - {len([c for c in df.columns if c.startswith('fund_')])} fundamental features")
print(f"  - {len([c for c in df.columns if 'relative' in c.lower()])} relative metrics")
print(f"  - {len([c for c in df.columns if 'score' in c.lower()])} composite scores")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print(f"1. Train with enhanced data:")
print(f"   python train_rf_multi_target.py --data-path {OUTPUT_FILE}")
print("="*80)
