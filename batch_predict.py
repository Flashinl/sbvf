"""
Batch prediction script - analyze multiple stocks at once

Generates predictions for a list of stocks and creates a comprehensive report
"""

from pathlib import Path
import sys

# Check if model exists
model_path = Path('models/multihorizon/best_model.pth')
if not model_path.exists():
    print("ERROR: Trained model not found!")
    print("\nPlease train the model first:")
    print("  python train_model.py")
    sys.exit(1)

from stockbot.ml.inference import MultiHorizonPredictor
from stockbot.ml.visualization import create_prediction_report
import pandas as pd

# Stocks to analyze (customize this list)
STOCKS_TO_ANALYZE = [
    # FAANG + Tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',

    # Finance
    'JPM', 'BAC', 'GS', 'WFC', 'V', 'MA',

    # Healthcare
    'UNH', 'JNJ', 'PFE', 'ABBV',

    # Consumer
    'WMT', 'HD', 'NKE', 'MCD', 'COST',

    # Industrial
    'CAT', 'BA', 'GE',

    # Energy
    'XOM', 'CVX'
]

print("="*80)
print("BATCH STOCK PREDICTION & ANALYSIS")
print("="*80)
print(f"\nAnalyzing {len(STOCKS_TO_ANALYZE)} stocks...")
print(f"Stocks: {', '.join(STOCKS_TO_ANALYZE[:10])}{'...' if len(STOCKS_TO_ANALYZE) > 10 else ''}")

# Load model
print("\nLoading model...")
predictor = MultiHorizonPredictor(str(model_path))

# Make predictions
print("\nGenerating predictions...")
predictions_df = predictor.predict_to_dataframe(STOCKS_TO_ANALYZE)

# Save predictions
output_dir = Path('predictions')
output_dir.mkdir(exist_ok=True)

predictions_df.to_csv('predictions/latest_predictions.csv', index=False)
print(f"\n✓ Predictions saved to: predictions/latest_predictions.csv")

# Generate visualizations
print("\nGenerating visualizations...")
create_prediction_report(predictions_df, 'predictions/report')

# Show top signals
print("\n" + "="*80)
print("TOP TRADING SIGNALS")
print("="*80)

for horizon in ['1week', '1month', '3month']:
    print(f"\n{horizon.upper().replace('WEEK', ' WEEK').replace('MONTH', ' MONTH')}:")
    print("-" * 80)

    # Top BUY signals
    buys = predictions_df[
        (predictions_df['horizon'] == horizon) &
        (predictions_df['signal'] == 'BUY')
    ].nlargest(5, 'confidence')

    if not buys.empty:
        print("\n  Top BUY Signals:")
        print(f"  {'Ticker':<8} {'Confidence':<12} {'Expected Return':<18} {'Price':<10}")
        print("  " + "-" * 60)
        for _, row in buys.iterrows():
            print(f"  {row['ticker']:<8} {row['confidence']:>10.1%}  "
                  f"{row['expected_return']:>+15.2f}%  ${row['current_price']:>8.2f}")

    # Top SELL signals
    sells = predictions_df[
        (predictions_df['horizon'] == horizon) &
        (predictions_df['signal'] == 'SELL')
    ].nlargest(5, 'confidence')

    if not sells.empty:
        print("\n  Top SELL Signals:")
        print(f"  {'Ticker':<8} {'Confidence':<12} {'Expected Return':<18} {'Price':<10}")
        print("  " + "-" * 60)
        for _, row in sells.iterrows():
            print(f"  {row['ticker']:<8} {row['confidence']:>10.1%}  "
                  f"{row['expected_return']:>+15.2f}%  ${row['current_price']:>8.2f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nResults saved to:")
print("  - predictions/latest_predictions.csv")
print("  - predictions/report/ (visualizations)")
print("="*80)
