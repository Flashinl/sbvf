"""
Test Catalyst Detection and Explanation Generation

Demonstrates the new catalyst detection system that identifies business
catalysts (deals, contracts, partnerships) and provides reasoning for
stock price movements.
"""

import sys
from pathlib import Path

# Check if model exists
model_path = Path('models/multihorizon/best_model.pth')
if not model_path.exists():
    print("ERROR: Trained model not found!")
    print("\nPlease train the model first:")
    print("  python train_model.py")
    sys.exit(1)

print("="*70)
print("CATALYST DETECTION & REASONING TEST")
print("="*70)

ticker = sys.argv[1].upper() if len(sys.argv) > 1 else 'AAPL'

print(f"\nTesting catalyst detection for {ticker}...")
print("-"*70)

# Step 1: Get ML predictions
print("\n[1/4] Loading ML model and making predictions...")
from stockbot.ml.inference import MultiHorizonPredictor

predictor = MultiHorizonPredictor(str(model_path))
ml_predictions = predictor.predict(ticker)

print(f"  Signal: {ml_predictions['1month']['signal']}")
print(f"  Expected Return: {ml_predictions['1month']['expected_return']:+.2f}%")
print(f"  Confidence: {ml_predictions['1month']['confidence']:.1%}")

# Step 2: Fetch news
print("\n[2/4] Fetching recent news...")
import os
from stockbot.providers.news_newsapi import fetch_news_newsapi
from stockbot.providers.news_finnhub import fetch_news_finnhub

newsapi_key = os.environ.get('NEWSAPI_KEY')
finnhub_key = os.environ.get('FINNHUB_API_KEY')

news_items = []

if newsapi_key:
    news_items.extend(fetch_news_newsapi(ticker, newsapi_key, max_items=10))
    print(f"  NewsAPI: {len([n for n in news_items if n.source])} articles")

if finnhub_key:
    finnhub_news = fetch_news_finnhub(ticker, finnhub_key, max_items=10)
    news_items.extend(finnhub_news)
    print(f"  Finnhub: {len(finnhub_news)} articles")

if not news_items:
    print("  WARNING: No news fetched (API keys not configured)")
    print("  Set NEWSAPI_KEY or FINNHUB_API_KEY environment variables")

print(f"  Total news items: {len(news_items)}")

# Step 3: Detect catalysts
print("\n[3/4] Detecting business catalysts...")
from stockbot.ml.catalyst_detection import CatalystDetector

detector = CatalystDetector()
catalysts = detector.detect_catalysts(news_items, max_articles=10)

print(f"  Detected {len(catalysts)} catalysts")

if catalysts:
    print("\n  Top Catalysts:")
    for i, cat in enumerate(catalysts[:5], 1):
        direction_symbol = {
            'bullish': '[+]',
            'bearish': '[-]',
            'neutral': '[=]'
        }[cat.impact_direction]

        impact_symbol = {
            'high': '***',
            'medium': '**',
            'low': '*'
        }[cat.impact_level]

        print(f"    {i}. {direction_symbol} {impact_symbol} {cat.type.upper()}: {cat.description[:80]}")
        if cat.entities:
            print(f"       Entities: {', '.join(cat.entities[:3])}")
        if cat.key_facts:
            print(f"       Fact: {cat.key_facts[0][:100]}...")
        print()

# Step 4: Generate explanation
print("[4/4] Generating explanation with reasoning...")
from stockbot.ml.catalyst_detection import ExplanationGenerator

generator = ExplanationGenerator()
explanation = generator.generate_explanation(
    ticker=ticker,
    ml_prediction=ml_predictions,
    news=news_items,
    horizon='1month'
)

# Display results
print("\n" + "="*70)
print(f"EXPLANATION FOR {ticker} - {explanation['horizon'].upper()}")
print("="*70)

print(f"\nSignal: {explanation['signal']}")
print(f"Expected Return: {explanation['expected_return']:+.2f}%")
print(f"Overall Confidence: {explanation['confidence']:.1%}")
print(f"  - ML Confidence: {explanation['ml_confidence']:.1%}")
print(f"  - Catalyst Support: {explanation['catalyst_support']:.1%}")

print(f"\n{'-'*70}")
print("REASONING:")
print(f"{'-'*70}")
print(f"{explanation['reasoning']}")

if explanation['key_factors']:
    print(f"\n{'-'*70}")
    print("KEY FACTORS:")
    print(f"{'-'*70}")
    for i, factor in enumerate(explanation['key_factors'][:5], 1):
        alignment = {
            'supporting': '[SUPPORTS]',
            'conflicting': '[CONFLICTS]',
            'neutral': '[NEUTRAL]'
        }.get(factor['alignment'], '')

        print(f"\n{i}. {factor['name']} {alignment}")
        print(f"   Category: {factor['category']}")
        print(f"   Weight: {factor['weight']:.1%}")
        print(f"   {factor['description']}")

if explanation['catalysts']:
    print(f"\n{'-'*70}")
    print("DETECTED CATALYSTS:")
    print(f"{'-'*70}")
    for i, cat in enumerate(explanation['catalysts'][:5], 1):
        print(f"\n{i}. {cat['type'].upper()} - {cat['impact_direction'].upper()} ({cat['impact_level']})")
        print(f"   {cat['description']}")
        if cat['entities']:
            print(f"   Entities: {', '.join(cat['entities'][:3])}")
        if cat['key_facts']:
            for fact in cat['key_facts'][:2]:
                print(f"   - {fact}")
        if cat['source_url']:
            print(f"   Source: {cat['source_url'][:80]}...")

print("\n" + "="*70)

# Test all horizons
print("\nMULTI-HORIZON SUMMARY:")
print("="*70)

for horizon in ['1week', '1month', '3month']:
    exp = generator.generate_explanation(
        ticker=ticker,
        ml_prediction=ml_predictions,
        news=news_items,
        horizon=horizon
    )

    horizon_display = horizon.replace('week', ' WEEK').replace('month', ' MONTH')
    print(f"\n{horizon_display}:")
    print(f"  Signal: {exp['signal']:5s} | Return: {exp['expected_return']:+6.2f}% | "
          f"Confidence: {exp['confidence']:5.1%} | Catalysts: {len(exp['catalysts'])}")
    print(f"  {exp['reasoning'][:100]}...")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
