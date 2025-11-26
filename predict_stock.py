"""
Simple stock prediction script

Usage:
    python predict_stock.py AAPL
    python predict_stock.py MSFT
"""

import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python predict_stock.py <TICKER>")
    print("\nExample:")
    print("  python predict_stock.py AAPL")
    sys.exit(1)

ticker = sys.argv[1].upper()

# Check if model exists
model_path = Path('models/multihorizon/best_model.pth')
if not model_path.exists():
    print("ERROR: Trained model not found!")
    print("\nPlease train the model first:")
    print("  python train_model.py")
    sys.exit(1)

# Load model and predict
from stockbot.ml.inference import MultiHorizonPredictor

print(f"\nLoading model...")
predictor = MultiHorizonPredictor(str(model_path))

print(f"Fetching data and making predictions for {ticker}...")
predictions = predictor.predict(ticker)

# Display results
print("\n" + "="*70)
print(f"PREDICTIONS FOR {ticker}")
print("="*70)
print(f"Current Price: ${predictions['1week']['current_price']:.2f}")
print(f"Prediction Date: {predictions['1week']['prediction_date'][:10]}")

for horizon in ['1week', '1month', '3month']:
    pred = predictions[horizon]

    print(f"\n{horizon.upper().replace('WEEK', ' WEEK').replace('MONTH', ' MONTH')}:")
    print(f"  Signal: {pred['signal']} (confidence: {pred['confidence']:.1%})")
    print(f"  Expected Return: {pred['expected_return']:+.2f}%")
    print(f"  Probabilities:")
    for signal in ['BUY', 'HOLD', 'SELL']:
        prob = pred['probabilities'][signal]
        bar = '=' * int(prob * 30)
        print(f"    {signal:5s}: {prob:5.1%} {bar}")

print("="*70)
