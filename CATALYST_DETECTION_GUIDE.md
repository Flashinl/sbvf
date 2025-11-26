# Catalyst Detection & Reasoning System

## Overview

The StockBot ML system now includes **catalyst detection and reasoning** capabilities that identify and explain business events affecting stock prices.

### What are Catalysts?

Catalysts are business events that can significantly impact stock prices:
- **Partnerships**: Strategic alliances, collaborations
- **Contracts**: New orders, deals, customer wins
- **Acquisitions**: M&A activity, buyouts
- **Earnings**: Quarterly results, guidance changes
- **Product Launches**: New product announcements
- **Regulatory**: FDA approvals, regulatory decisions
- **Executive Changes**: Leadership appointments/departures
- **Analyst Ratings**: Upgrades, downgrades, price target changes

## Key Features

### 1. Catalyst Detection
Analyzes news articles to identify business catalysts with:
- **Type classification**: Automatically categorizes catalyst type
- **Impact direction**: Bullish, bearish, or neutral
- **Impact level**: High, medium, or low severity
- **Entity extraction**: Identifies companies/organizations involved
- **Key facts**: Extracts supporting evidence from articles
- **Confidence scoring**: How confident we are in the detection

### 2. Reasoning Generation
Provides human-readable explanations for predictions:
- **Why it's moving**: Clear explanation of price drivers
- **Catalyst alignment**: How news supports/conflicts with ML signal
- **Key factors**: Top factors influencing the prediction
- **Multi-source**: Combines ML model + news catalysts + technical analysis

### 3. ML Integration
Seamlessly integrated with existing ML predictions:
- Works with all time horizons (1 week, 1 month, 3 months)
- Enhances confidence scoring based on catalyst support
- No change to core prediction logic

## Quick Start

### Simple Prediction with Reasoning

```python
from stockbot.ml.inference import MultiHorizonPredictor

# Load model
predictor = MultiHorizonPredictor('models/multihorizon/best_model.pth')

# Get prediction with explanation
result = predictor.predict_with_explanation(
    ticker='AAPL',
    horizon='1month',
    fetch_news=True  # Automatically fetches and analyzes news
)

# Display reasoning
print(result['reasoning'])
# "The model predicts a BUY signal for AAPL over the 1 month timeframe
# with +4.2% expected return. Recent positive catalyst: Apple announced
# a partnership (positive development). News catalysts strongly support
# this prediction. The model has high confidence in this prediction."

# Access catalysts
for catalyst in result['catalysts']:
    print(f"{catalyst['type']}: {catalyst['description']}")
    print(f"  Impact: {catalyst['impact_direction']} ({catalyst['impact_level']})")
    print(f"  Entities: {', '.join(catalyst['entities'])}")
```

### Command-Line Usage

```bash
# Simple prediction with reasoning
python predict_with_reasoning.py AAPL

# Detailed catalyst analysis
python test_catalyst_detection.py NVDA
```

### Output Example

```
PREDICTION FOR AAPL - 1 MONTH
======================================================================
Current Price: $184.50
Signal: BUY
Expected Return: +4.2%
Overall Confidence: 82.3%
  - ML Model: 76.5%
  - Catalyst Support: 85.0%

----------------------------------------------------------------------
WHY IS IT MOVING?
----------------------------------------------------------------------
The model predicts a BUY signal for AAPL over the 1 month timeframe
with +4.2% expected return. Recent positive catalyst: Apple announced
a strategic partnership with major cloud provider (positive development).
News catalysts strongly support this prediction. The model has high
confidence in this prediction.

----------------------------------------------------------------------
BUSINESS CATALYSTS DETECTED:
----------------------------------------------------------------------

1. PARTNERSHIP [BULLISH] ***
   Apple announced a strategic partnership with major cloud provider
   Companies: Apple Inc., Microsoft Corporation
   Key Facts:
     - Apple and Microsoft announced a multi-year cloud partnership...
     - The partnership will expand Apple's enterprise offerings...
   Source: https://example.com/news/apple-microsoft-partnership

2. CONTRACT [BULLISH] **
   Apple secured a new contract with major automotive manufacturer
   Companies: Apple Inc., Tesla
   Key Facts:
     - Apple wins major contract to supply chips for autonomous vehicles...
```

## API Integration

### In FastAPI Endpoints

```python
from stockbot.ml_service import get_ml_service

@app.get("/analyze/{ticker}")
async def analyze(ticker: str):
    ml_service = get_ml_service()

    # Get prediction with explanation
    result = await ml_service.predict_with_explanation(
        ticker=ticker,
        horizon='1month'
    )

    return {
        'signal': result['signal'],
        'reasoning': result['reasoning'],
        'catalysts': result['catalysts'],
        'confidence': result['confidence']
    }
```

## How It Works

### 1. Pattern Matching
Uses keyword patterns to detect catalyst types:
- Each catalyst type has specific keywords and action verbs
- Example: "partnership" looks for: partner, collaborate, alliance, joint venture
- Example: "contract" looks for: order, deal, agreement, award, customer win

### 2. NLP Analysis
Leverages spaCy for advanced text analysis:
- Named Entity Recognition (NER) to extract companies
- Sentence tokenization to find key facts
- Part-of-speech tagging for action verbs

### 3. Sentiment Analysis
Determines bullish/bearish direction:
- Scans for positive terms: beat, surge, upgrade, approval, win
- Scans for negative terms: miss, fall, downgrade, lawsuit, delay
- Prioritizes title over body text

### 4. Impact Assessment
Evaluates significance level:
- **High impact**: Major deals, record-breaking, multi-billion dollar
- **Medium impact**: Strategic partnerships, multi-year contracts
- **Low impact**: Smaller announcements

### 5. Explanation Generation
Combines all signals into coherent reasoning:
- ML prediction (signal, expected return, confidence)
- Detected catalysts (type, direction, impact)
- Alignment score (how well catalysts support prediction)
- Key factors ranked by weight

## Configuration

### News API Keys

For best results, set environment variables:

```bash
# Windows (Command Prompt)
set NEWSAPI_KEY=your_key_here
set FINNHUB_API_KEY=your_key_here

# Windows (PowerShell)
$env:NEWSAPI_KEY="your_key_here"
$env:FINNHUB_API_KEY="your_key_here"

# Linux/Mac
export NEWSAPI_KEY=your_key_here
export FINNHUB_API_KEY=your_key_here
```

Get free API keys:
- NewsAPI: https://newsapi.org/register
- Finnhub: https://finnhub.io/register

### Custom Catalyst Detection

```python
from stockbot.ml.catalyst_detection import CatalystDetector

# Initialize detector
detector = CatalystDetector()

# Detect catalysts from news
catalysts = detector.detect_catalysts(
    news_items,
    max_articles=10,
    min_confidence=0.3
)

# Analyze specific catalyst
for catalyst in catalysts:
    if catalyst.type == 'partnership':
        print(f"Partnership detected: {catalyst.description}")
        print(f"Entities: {catalyst.entities}")
        print(f"Key facts: {catalyst.key_facts}")
```

## Advanced Usage

### Multi-Horizon Analysis

```python
# Analyze all time horizons
for horizon in ['1week', '1month', '3month']:
    result = predictor.predict_with_explanation(
        ticker='AAPL',
        horizon=horizon
    )

    print(f"{horizon}: {result['signal']} "
          f"({result['expected_return']:+.1f}%)")
    print(f"  Reasoning: {result['reasoning']}")
```

### Custom News Sources

```python
from stockbot.providers.news_newsapi import NewsItem

# Provide your own news items
custom_news = [
    NewsItem(
        title="Company announces major partnership",
        url="https://...",
        published_at="2025-01-15T10:00:00Z",
        source="Reuters",
        sentiment=0.8
    )
]

# Use custom news
result = predictor.predict_with_explanation(
    ticker='AAPL',
    news=custom_news,
    fetch_news=False  # Don't fetch, use provided news
)
```

### Batch Processing

```python
tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA']

for ticker in tickers:
    result = predictor.predict_with_explanation(ticker)
    print(f"{ticker}: {result['signal']} - {result['reasoning'][:100]}...")
```

## Model Details

### Supported Catalyst Types

| Type | Keywords | Default Impact |
|------|----------|----------------|
| Partnership | partner, alliance, collaboration | Bullish |
| Contract | order, deal, agreement, award | Bullish |
| Acquisition | acquire, merger, buyout | Bullish |
| Earnings | revenue, profit, guidance | Neutral |
| Product Launch | launch, unveil, introduce | Bullish |
| Regulatory | FDA, approval, clearance | Neutral |
| Executive | CEO, CFO, leadership change | Neutral |
| Analyst | upgrade, downgrade, rating | Neutral |

### Confidence Calculation

Overall confidence = ML confidence (60%) + Catalyst support (30%) + Bonus (10%)

- **ML confidence**: Base model probability
- **Catalyst support**: How well catalysts align with signal
- **Bonus**: +10% if catalysts are detected

### Key Factor Weights

Factors are weighted by:
- Catalyst impact level (high=0.9, medium=0.6, low=0.3)
- Model confidence
- Catalyst-signal alignment

## Limitations

1. **News availability**: Quality depends on API access and article availability
2. **Language**: Currently English-only
3. **Temporal lag**: News catalysts may lag actual events by hours/days
4. **False positives**: Some non-financial news may be misclassified
5. **Sentiment accuracy**: Basic keyword-based sentiment (not transformer-based)

## Future Enhancements

- [ ] Transformer-based sentiment analysis
- [ ] Multi-language support
- [ ] Real-time news streaming
- [ ] Historical catalyst tracking
- [ ] Catalyst impact backtesting
- [ ] Custom catalyst pattern definitions
- [ ] Integration with SEC filings (8-K, 10-K)
- [ ] Social media sentiment (Twitter, Reddit)

## Troubleshooting

### No catalysts detected
- Check that news API keys are configured
- Verify news items are being fetched (check logs)
- Try lowering `min_confidence` threshold
- Ensure articles are recent (< 2 weeks old)

### Low catalyst support score
- Catalysts may conflict with ML signal (both could be valid)
- Check catalyst `impact_direction` vs signal
- Consider it a "proceed with caution" signal

### Performance issues
- Limit `max_articles` to 10 or fewer
- Disable news fetching if using cached predictions
- Use batch processing for multiple tickers

## Examples

See example scripts:
- `predict_with_reasoning.py` - Simple prediction with explanation
- `test_catalyst_detection.py` - Detailed catalyst analysis
- `batch_predict.py` - Batch predictions (can be extended with explanations)

## Technical Reference

### Class: `CatalystDetector`
Location: `stockbot/ml/catalyst_detection.py`

Main methods:
- `detect_catalysts(news, max_articles, min_confidence)` - Detect catalysts from news
- `_analyze_text(text, title, url, published_at)` - Analyze single article
- `_determine_impact_direction(text, title, catalyst_type)` - Assess bullish/bearish
- `_extract_entities(text)` - Extract company names
- `_extract_key_facts(text, catalyst_type)` - Find supporting facts

### Class: `ExplanationGenerator`
Location: `stockbot/ml/catalyst_detection.py`

Main methods:
- `generate_explanation(ticker, ml_prediction, news, horizon)` - Full explanation
- `_assess_catalyst_support(signal, catalysts)` - Calculate alignment
- `_generate_reasoning(...)` - Create human-readable text
- `_identify_key_factors(...)` - Rank driving factors

### Class: `MultiHorizonPredictor`
Location: `stockbot/ml/inference.py`

New method:
- `predict_with_explanation(ticker, news, horizon, fetch_news)` - Combined prediction + catalysts

## License & Attribution

This catalyst detection system is part of the StockBot ML platform.
Uses spaCy for NLP, trafilatura for article extraction.
