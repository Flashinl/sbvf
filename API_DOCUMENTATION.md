# StockBot VF API Documentation

## Base URL
```
Production: https://your-app.onrender.com
Development: http://localhost:8000
```

## Authentication

All endpoints (except `/health`, `/auth/*`) require authentication.

### Register
```http
POST /auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "secure_password"
}
```

### Login
```http
POST /auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "secure_password"
}
```

Returns session cookie for subsequent requests.

## Core Endpoints

### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "ok"
}
```

---

### 2. Stock Analysis (with ML Predictions)

Comprehensive stock analysis combining traditional technical analysis with ML predictions and catalyst detection.

```http
GET /analyze?ticker=AAPL&risk=medium&max_news=10
```

**Parameters:**
- `ticker` (required): Stock symbol (e.g., AAPL, MSFT)
- `risk` (optional): Risk tolerance - `low`, `medium`, `high` (default: `medium`)
- `max_news` (optional): Maximum news articles (default: 10, max: 25)

**Response:**
```json
{
  "price": {
    "ticker": "AAPL",
    "price": 184.50,
    "currency": "USD",
    "market_cap": 2850000000000,
    "trailing_pe": 28.5,
    "eps_ttm": 6.46,
    "dividend_yield": 0.0048,
    "fifty_two_week_high": 199.62,
    "fifty_two_week_low": 164.08,
    "earnings_date": "2025-02-01T00:00:00",
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "long_name": "Apple Inc."
  },
  "technicals": {
    "sma20": 182.30,
    "sma50": 178.45,
    "sma200": 175.20,
    "rsi14": 62.5,
    "trend_score": 0.75
  },
  "news": [
    {
      "title": "Apple announces new partnership...",
      "url": "https://...",
      "published_at": "2025-01-15T10:00:00Z",
      "source": "Reuters",
      "sentiment": 0.8
    }
  ],
  "recommendation": {
    "label": "BUY",
    "confidence": 0.82,
    "rationale": "Strong fundamentals with positive momentum...",
    "ai_analysis": "Detailed analysis...",
    "predicted_price": 195.00
  },
  "ml_prediction": {
    "signal": "BUY",
    "expected_return": 4.2,
    "confidence": 0.823,
    "ml_confidence": 0.765,
    "catalyst_support": 0.85,
    "reasoning": "The model predicts a BUY signal for AAPL over the 1 month timeframe with +4.2% expected return. Recent positive catalyst: Apple announced a strategic partnership (positive development). News catalysts strongly support this prediction.",
    "catalysts": [
      {
        "type": "partnership",
        "description": "Apple announced a strategic partnership with major cloud provider",
        "impact_direction": "bullish",
        "impact_level": "high",
        "entities": ["Apple Inc.", "Microsoft Corporation"],
        "key_facts": [
          "Apple and Microsoft announced a multi-year cloud partnership..."
        ],
        "source_url": "https://..."
      }
    ],
    "key_factors": [
      {
        "category": "business_catalyst",
        "name": "Partnership",
        "description": "Apple announced a strategic partnership...",
        "direction": "bullish",
        "weight": 0.9,
        "alignment": "supporting"
      },
      {
        "category": "model_confidence",
        "name": "ML Model Confidence",
        "description": "Model confidence: 76.5%",
        "direction": "positive",
        "weight": 0.765,
        "alignment": "supporting"
      }
    ],
    "horizons": {
      "1week": {
        "signal": "HOLD",
        "expected_return": 1.2,
        "confidence": 0.68
      },
      "1month": {
        "signal": "BUY",
        "expected_return": 4.2,
        "confidence": 0.76
      },
      "3month": {
        "signal": "BUY",
        "expected_return": 8.5,
        "confidence": 0.71
      }
    }
  }
}
```

---

### 3. ML Prediction (Standalone)

Get ML predictions with catalyst detection and reasoning without full analysis.

```http
GET /predict/ml?ticker=AAPL&horizon=1month&fetch_news=true
```

**Parameters:**
- `ticker` (required): Stock symbol
- `horizon` (optional): Time horizon - `1week`, `1month`, `3month` (default: `1month`)
- `fetch_news` (optional): Fetch and analyze news catalysts (default: `true`)

**Response:**
```json
{
  "ticker": "AAPL",
  "horizon": "1month",
  "signal": "BUY",
  "expected_return": 4.2,
  "confidence": 0.823,
  "ml_confidence": 0.765,
  "catalyst_support": 0.85,
  "reasoning": "The model predicts a BUY signal for AAPL over the 1 month timeframe with +4.2% expected return. Recent positive catalyst: Apple announced a strategic partnership (positive development). News catalysts strongly support this prediction. The model has high confidence in this prediction.",
  "catalysts": [
    {
      "type": "partnership",
      "description": "Apple announced a strategic partnership with major cloud provider",
      "impact_direction": "bullish",
      "impact_level": "high",
      "entities": ["Apple Inc.", "Microsoft Corporation"],
      "key_facts": ["Apple and Microsoft announced a multi-year cloud partnership..."],
      "source_url": "https://..."
    }
  ],
  "key_factors": [
    {
      "category": "business_catalyst",
      "name": "Partnership",
      "description": "Apple announced a strategic partnership...",
      "direction": "bullish",
      "weight": 0.9,
      "alignment": "supporting"
    }
  ],
  "all_horizons": {
    "1week": {
      "signal": "HOLD",
      "probabilities": {"BUY": 0.42, "HOLD": 0.45, "SELL": 0.13},
      "expected_return": 1.2,
      "confidence": 0.68,
      "current_price": 184.50,
      "prediction_date": "2025-01-15T10:00:00"
    },
    "1month": {
      "signal": "BUY",
      "probabilities": {"BUY": 0.76, "HOLD": 0.18, "SELL": 0.06},
      "expected_return": 4.2,
      "confidence": 0.76,
      "current_price": 184.50,
      "prediction_date": "2025-01-15T10:00:00"
    },
    "3month": {
      "signal": "BUY",
      "probabilities": {"BUY": 0.71, "HOLD": 0.22, "SELL": 0.07},
      "expected_return": 8.5,
      "confidence": 0.71,
      "current_price": 184.50,
      "prediction_date": "2025-01-15T10:00:00"
    }
  }
}
```

---

### 4. Batch ML Predictions

Get predictions for multiple tickers in one request.

```http
GET /predict/ml/batch?tickers=AAPL,MSFT,GOOGL&horizon=1month
```

**Parameters:**
- `tickers` (required): Comma-separated ticker symbols (max 20)
- `horizon` (optional): Time horizon - `1week`, `1month`, `3month` (default: `1month`)

**Response:**
```json
{
  "horizon": "1month",
  "count": 3,
  "results": {
    "AAPL": {
      "signal": "BUY",
      "expected_return": 4.2,
      "confidence": 0.76,
      "reasoning": "The model predicts a BUY signal..."
    },
    "MSFT": {
      "signal": "HOLD",
      "expected_return": 2.1,
      "confidence": 0.64,
      "reasoning": "The model predicts a HOLD signal..."
    },
    "GOOGL": {
      "signal": "BUY",
      "expected_return": 5.8,
      "confidence": 0.81,
      "reasoning": "The model predicts a BUY signal..."
    }
  }
}
```

**Note:** News catalyst detection is disabled for batch requests to improve performance.

---

### 5. ML Model Information

Get details about the loaded ML model.

```http
GET /ml/model-info
```

**Response:**
```json
{
  "available": true,
  "encoder_type": "transformer",
  "d_model": 128,
  "sequence_length": 60,
  "horizons": ["1week", "1month", "3month"],
  "num_features": 42
}
```

If model not loaded:
```json
{
  "available": false,
  "message": "ML model not loaded"
}
```

---

## Catalyst Types

The system detects 8 types of business catalysts:

| Type | Description | Examples |
|------|-------------|----------|
| `partnership` | Strategic partnerships, collaborations | Joint ventures, alliances |
| `contract` | New contracts, orders, deals | Customer wins, supply agreements |
| `acquisition` | M&A activity | Buyouts, mergers, strategic investments |
| `earnings` | Quarterly results, guidance | Earnings beats/misses, forecast changes |
| `product_launch` | New product announcements | Product releases, unveilings |
| `regulatory` | Regulatory decisions | FDA approvals, patent grants |
| `executive` | Leadership changes | CEO appointments, resignations |
| `analyst` | Analyst ratings | Upgrades, downgrades, price targets |

## Error Responses

### 400 Bad Request
```json
{
  "error": "No valid tickers provided"
}
```

### 401 Unauthorized
```json
{
  "error": "Not authenticated"
}
```

### 404 Not Found
```json
{
  "error": "Ticker not found or no data available. Try a valid symbol like AAPL or NVDA."
}
```

### 500 Internal Server Error
```json
{
  "error": "Analysis failed. Please try another ticker or try again shortly."
}
```

### 503 Service Unavailable
```json
{
  "error": "ML prediction service not available",
  "message": "The ML model is not loaded. Please contact support."
}
```

### 504 Gateway Timeout
```json
{
  "error": "Timed out"
}
```

## Rate Limits

- **Free tier**: No explicit limits, but may be throttled under heavy load
- **Authenticated**: Higher priority in queue
- **Batch requests**: Max 20 tickers per request

## Best Practices

### Performance
1. **Use batch endpoints** when possible for multiple tickers
2. **Cache results** - predictions don't change frequently
3. **Set appropriate timeouts** - ML predictions can take 3-10 seconds
4. **Disable news fetching** for batch requests (already done)

### Reliability
1. **Handle timeouts gracefully** - retry with exponential backoff
2. **Check ML model availability** via `/ml/model-info` before bulk operations
3. **Validate ticker symbols** before API calls

### Security
1. **Use HTTPS** in production (automatic with Render)
2. **Store API credentials securely** - never commit to git
3. **Implement rate limiting** on client side
4. **Log out inactive sessions**

## Example Client Code

### Python
```python
import requests

# Login
session = requests.Session()
session.post(
    'https://your-app.onrender.com/auth/login',
    json={'email': 'user@example.com', 'password': 'password'}
)

# Get prediction
response = session.get(
    'https://your-app.onrender.com/predict/ml',
    params={'ticker': 'AAPL', 'horizon': '1month'}
)
data = response.json()

print(f"Signal: {data['signal']}")
print(f"Expected Return: {data['expected_return']:.2f}%")
print(f"Reasoning: {data['reasoning']}")

for catalyst in data['catalysts']:
    print(f"\n{catalyst['type'].upper()}: {catalyst['description']}")
```

### JavaScript
```javascript
const API_BASE = 'https://your-app.onrender.com';

// Login
const loginResponse = await fetch(`${API_BASE}/auth/login`, {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    email: 'user@example.com',
    password: 'password'
  }),
  credentials: 'include'
});

// Get prediction
const response = await fetch(
  `${API_BASE}/predict/ml?ticker=AAPL&horizon=1month`,
  {credentials: 'include'}
);
const data = await response.json();

console.log(`Signal: ${data.signal}`);
console.log(`Expected Return: ${data.expected_return}%`);
console.log(`Reasoning: ${data.reasoning}`);
```

### cURL
```bash
# Login
curl -c cookies.txt -X POST https://your-app.onrender.com/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"password"}'

# Get prediction
curl -b cookies.txt "https://your-app.onrender.com/predict/ml?ticker=AAPL&horizon=1month"
```

## Changelog

### v0.2.0 (Current)
- ✨ Added ML prediction endpoints
- ✨ Catalyst detection and reasoning
- ✨ Multi-horizon predictions (1 week, 1 month, 3 months)
- ✨ Batch prediction support
- 🔧 Improved error handling
- 📚 Comprehensive API documentation

### v0.1.0
- 🎉 Initial release
- ✅ Stock analysis endpoint
- ✅ Authentication system
- ✅ News integration
