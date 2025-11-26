"""
Stock Prediction with Catalyst Reasoning

Makes ML predictions and provides explanations based on detected
business catalysts (deals, contracts, partnerships, etc.)

Usage:
    python predict_with_reasoning.py AAPL
    python predict_with_reasoning.py NVDA
"""

import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python predict_with_reasoning.py <TICKER>")
    print("\nExample:")
    print("  python predict_with_reasoning.py AAPL")
    sys.exit(1)

ticker = sys.argv[1].upper()

# Check if model exists
model_path = Path('models/multihorizon/best_model.pth')
if not model_path.exists():
    print("ERROR: Trained model not found!")
    print("\nPlease train the model first:")
    print("  python train_model.py")
    sys.exit(1)

# Load model and make prediction with explanation
from stockbot.ml.inference import MultiHorizonPredictor

print(f"\nLoading model...")
predictor = MultiHorizonPredictor(str(model_path))

print(f"Making prediction for {ticker} with catalyst analysis...")
print("(This will fetch news and detect business catalysts)")
print()

# Get prediction with explanation for 1-month horizon
result = predictor.predict_with_explanation(
    ticker=ticker,
    horizon='1month',
    fetch_news=True
)

# Display results
print("="*70)
print(f"PREDICTION FOR {ticker} - {result['horizon'].upper()}")
print("="*70)

print(f"\nCurrent Price: ${result['predictions_all_horizons']['1month']['current_price']:.2f}")
print(f"Signal: {result['signal']}")
print(f"Expected Return: {result['expected_return']:+.2f}%")
print(f"Overall Confidence: {result['confidence']:.1%}")
print(f"  - ML Model: {result['ml_confidence']:.1%}")
print(f"  - Catalyst Support: {result['catalyst_support']:.1%}")

print(f"\n{'-'*70}")
print("WHY IS IT MOVING?")
print(f"{'-'*70}")
print(result['reasoning'])

if result['catalysts']:
    print(f"\n{'-'*70}")
    print("BUSINESS CATALYSTS DETECTED:")
    print(f"{'-'*70}")

    for i, cat in enumerate(result['catalysts'], 1):
        impact_symbol = {
            'bullish': '[BULLISH]',
            'bearish': '[BEARISH]',
            'neutral': '[NEUTRAL]'
        }[cat['impact_direction']]

        level_stars = {'high': '***', 'medium': '**', 'low': '*'}[cat['impact_level']]

        print(f"\n{i}. {cat['type'].upper()} {impact_symbol} {level_stars}")
        print(f"   {cat['description']}")

        if cat['entities']:
            print(f"   Companies: {', '.join(cat['entities'][:3])}")

        if cat['key_facts']:
            print(f"   Key Facts:")
            for fact in cat['key_facts'][:2]:
                print(f"     - {fact[:100]}...")

        if cat['source_url']:
            print(f"   Source: {cat['source_url'][:70]}...")

if result['key_factors']:
    print(f"\n{'-'*70}")
    print("KEY FACTORS:")
    print(f"{'-'*70}")

    for i, factor in enumerate(result['key_factors'][:5], 1):
        alignment_symbol = {
            'supporting': '[+]',
            'conflicting': '[-]',
            'neutral': '[=]'
        }.get(factor['alignment'], '')

        print(f"\n{i}. {factor['name']} {alignment_symbol}")
        print(f"   Weight: {factor['weight']:.1%} | Category: {factor['category']}")
        print(f"   {factor['description']}")

# Show all horizons
print(f"\n{'-'*70}")
print("ALL HORIZONS:")
print(f"{'-'*70}")

for horizon in ['1week', '1month', '3month']:
    pred = result['predictions_all_horizons'][horizon]
    horizon_display = horizon.replace('week', ' WEEK').replace('month', ' MONTH')

    print(f"\n{horizon_display}:")
    print(f"  Signal: {pred['signal']:5s} | Return: {pred['expected_return']:+6.2f}% | "
          f"Confidence: {pred['confidence']:5.1%}")
    print(f"  Probabilities: BUY {pred['probabilities']['BUY']:.1%} | "
          f"HOLD {pred['probabilities']['HOLD']:.1%} | "
          f"SELL {pred['probabilities']['SELL']:.1%}")

print("\n" + "="*70)
print(f"\nNOTE: This analysis combines ML predictions with news catalyst detection.")
print("Catalysts include: partnerships, contracts, acquisitions, earnings,")
print("product launches, regulatory decisions, and analyst ratings.")
print("\nSet NEWSAPI_KEY or FINNHUB_API_KEY environment variables for better")
print("catalyst detection. Without API keys, predictions rely on technical analysis.")
print("="*70)
