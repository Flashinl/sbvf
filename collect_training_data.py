"""
Quick script to collect training data for a diverse set of stocks

This collects data across multiple sectors to ensure good model generalization
"""

from stockbot.ml.data_collection import prepare_dataset, save_dataset

# Diverse stock selection across sectors
DIVERSE_STOCKS = [
    # Technology (15)
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AVGO', 'ORCL', 'ADBE',
    'CRM', 'CSCO', 'INTC', 'AMD', 'QCOM',

    # Healthcare (10)
    'UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'PFE', 'BMY',

    # Financials (10)
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'USB',

    # Consumer Discretionary (10)
    'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX', 'BKNG', 'MAR', 'CMG', 'F',

    # Consumer Staples (8)
    'WMT', 'PG', 'KO', 'PEP', 'COST', 'PM', 'MDLZ', 'CL',

    # Energy (8)
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'PSX',

    # Industrials (10)
    'CAT', 'BA', 'HON', 'UNP', 'UPS', 'RTX', 'DE', 'LMT', 'GE', 'MMM',

    # Materials (5)
    'LIN', 'APD', 'SHW', 'ECL', 'NEM',

    # Utilities (5)
    'NEE', 'DUK', 'SO', 'D', 'AEP',

    # Real Estate (5)
    'AMT', 'PLD', 'CCI', 'EQIX', 'PSA',

    # Communications (4)
    'T', 'VZ', 'TMUS', 'NFLX'
]

print("="*80)
print("STOCK TRADING SIGNAL - TRAINING DATA COLLECTION")
print("="*80)
print(f"\nCollecting data for {len(DIVERSE_STOCKS)} stocks across all sectors")
print(f"Sectors: Tech, Healthcare, Finance, Consumer, Energy, Industrial, Materials, Utilities, Real Estate, Communications")
print("\nThis will take approximately 10-20 minutes...")
print("="*80)

# Collect data
dataset = prepare_dataset(
    tickers=DIVERSE_STOCKS,
    start_date='2020-01-01',  # 5 years of data
    end_date=None,  # Up to today
    min_history_days=250,  # Require at least 250 days
    include_sector_correlation=False,  # Set to True for more features (slower)
    delay=0.5,  # Avoid rate limits
    verbose=True
)

# Save to CSV
output_path = 'data/training_data_multihorizon.csv'
save_dataset(dataset, output_path)

print("\n" + "="*80)
print("DATA COLLECTION COMPLETE!")
print("="*80)
print(f"\nDataset saved to: {output_path}")
print(f"Total samples: {len(dataset):,}")
print(f"\nNext step: Train the model")
print("\nRun this command:")
print("  python train_model.py")
print("="*80)
