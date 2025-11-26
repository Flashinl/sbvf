# Windows Quick Start Guide

Easy-to-use scripts for Windows users. No complex command-line arguments needed!

## Prerequisites

```bash
pip install -r requirements.txt
```

## Complete Workflow (3 Simple Steps)

### Step 1: Collect Training Data (10-20 minutes)

Collects data for **90+ stocks** across all major sectors (Tech, Healthcare, Finance, Energy, etc.)

```bash
python collect_training_data.py
```

**What it does:**
- Downloads 5 years of historical data (2020-2025)
- Computes technical indicators (RSI, MACD, Bollinger Bands, ATR, SMAs)
- Adds VIX market volatility
- Creates multi-horizon labels (1 week, 1 month, 3 months)
- Saves to `data/training_data_multihorizon.csv`

**Stocks covered:**
- 15 Technology stocks (AAPL, MSFT, GOOGL, NVDA, etc.)
- 10 Healthcare stocks (UNH, JNJ, PFE, etc.)
- 10 Financial stocks (JPM, BAC, GS, etc.)
- 10 Consumer stocks (HD, WMT, NKE, etc.)
- 8 Energy stocks (XOM, CVX, COP, etc.)
- 10 Industrial stocks (CAT, BA, GE, etc.)
- And more across Materials, Utilities, Real Estate, Communications

### Step 2: Train Model (15-45 minutes)

Trains a Transformer model with walk-forward validation.

```bash
python train_model.py
```

**What it does:**
- Loads training data
- Splits into train/validation/test sets (temporal order preserved)
- Trains Transformer model with:
  - Multi-horizon prediction heads (1 week, 1 month, 3 months)
  - Classification (BUY/HOLD/SELL) + Regression (expected return)
  - Early stopping
  - Class weighting for imbalanced data
- Saves best model to `models/multihorizon/best_model.pth`

**Training time:**
- GPU: ~15 minutes
- CPU (8 cores): ~30-45 minutes

### Step 3: Make Predictions

#### Option A: Single Stock Prediction

```bash
python predict_stock.py AAPL
python predict_stock.py MSFT
python predict_stock.py GOOGL
```

**Example output:**
```
======================================================================
PREDICTIONS FOR AAPL
======================================================================
Current Price: $178.50
Prediction Date: 2025-11-25

1 WEEK:
  Signal: BUY (confidence: 65.0%)
  Expected Return: +4.20%
  Probabilities:
    BUY  : 65.0% ███████████████████
    HOLD : 25.0% ███████
    SELL : 10.0% ███

1 MONTH:
  Signal: BUY (confidence: 58.0%)
  Expected Return: +6.80%
  Probabilities:
    BUY  : 58.0% █████████████████
    HOLD : 30.0% █████████
    SELL : 12.0% ███

3 MONTH:
  Signal: BUY (confidence: 62.0%)
  Expected Return: +12.50%
  Probabilities:
    BUY  : 62.0% ██████████████████
    HOLD : 28.0% ████████
    SELL : 10.0% ███
======================================================================
```

#### Option B: Batch Prediction (30+ stocks)

```bash
python batch_predict.py
```

**What it does:**
- Analyzes 30+ stocks across all sectors
- Generates predictions for all horizons
- Creates comprehensive report with visualizations
- Shows top BUY and SELL signals ranked by confidence

**Output:**
- `predictions/latest_predictions.csv` - All predictions in CSV format
- `predictions/report/` - Folder with visualizations:
  - Dashboard overview
  - Calibration curves
  - Confusion matrices
  - Distribution plots
  - Top signals charts

## Additional Tools

### View Training History

```bash
python visualize_training.py
```

Shows training/validation loss curves and learning rate schedule.

### Customize Stock List

Edit `batch_predict.py` and modify the `STOCKS_TO_ANALYZE` list:

```python
STOCKS_TO_ANALYZE = [
    'AAPL', 'MSFT', 'GOOGL',  # Your stocks here
    # Add more...
]
```

## Advanced: Using Command-Line Directly

If you want to use the full command-line interface, use this syntax (no backslashes on Windows):

```bash
python -m stockbot.ml.train_multihorizon --data-path data/training_data_multihorizon.csv --encoder-type transformer --epochs 100 --batch-size 32
```

Or with PowerShell (use backtick for line continuation):

```powershell
python -m stockbot.ml.train_multihorizon `
    --data-path data/training_data_multihorizon.csv `
    --encoder-type transformer `
    --epochs 100 `
    --batch-size 32
```

## Configuration

### Change Model Type

Edit `train_model.py` and change:

```python
model_config = {
    'encoder_type': 'lstm',  # Change to 'lstm' for faster training
    'd_model': 128,
    # ...
}
```

### Adjust Training Parameters

Edit `train_model.py`:

```python
training_config = {
    'epochs': 150,  # More epochs
    'batch_size': 64,  # Larger batch (if you have GPU memory)
    'learning_rate': 0.0005,  # Lower learning rate
    # ...
}
```

### Collect More/Less Data

Edit `collect_training_data.py`:

```python
# Change date range
start_date='2018-01-01',  # More historical data

# Add more stocks
DIVERSE_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL',
    # Add your stocks here
]

# Enable sector correlations (slower but more features)
include_sector_correlation=True,
```

## Expected Performance

### Training Metrics (typical results)
- **Accuracy**: 55-65% (vs 33% random baseline)
- **Directional Accuracy**: 60-70%
- **F1-Score**: 0.50-0.60 (macro average)
- **Calibration ECE**: 0.05-0.15 (lower is better)

### Prediction Confidence
- High confidence (>70%): Most reliable signals
- Medium confidence (60-70%): Moderate reliability
- Low confidence (<60%): Less reliable, use caution

### Signal Distribution (typical)
- BUY: 30-35%
- HOLD: 35-45%
- SELL: 25-30%

## Troubleshooting

### "Out of memory" error
**Solution:** Edit `train_model.py` and reduce `batch_size` to 16 or 8:
```python
'batch_size': 16,  # or 8
```

### "CUDA not available" warning
**Solution:** This is fine! The model will use CPU. It just takes longer (~30-45 min vs ~15 min on GPU).

### "Rate limit exceeded" during data collection
**Solution:** Edit `collect_training_data.py` and increase delay:
```python
delay=1.0,  # Increase from 0.5 to 1.0
```

### Training is too slow
**Solution:** Use LSTM instead of Transformer. Edit `train_model.py`:
```python
'encoder_type': 'lstm',  # Faster than transformer
```

### Poor prediction accuracy
**Solutions:**
1. Collect more training data (earlier start_date)
2. Train for more epochs (150-200)
3. Use larger model (d_model=256)
4. Enable sector correlations in data collection

## Next Steps

After getting predictions, you can:

1. **Backtest the strategy** - See how signals would have performed historically
2. **Integrate with existing API** - Add predictions to your stockbot API
3. **Schedule regular retraining** - Retrain monthly with latest data
4. **Create custom alerts** - Get notified of high-confidence signals
5. **Build portfolio** - Use signals for automated portfolio construction

See `TRAINING_PIPELINE_GUIDE.md` for advanced features like backtesting.

## File Structure

```
StockBotVF/
├── collect_training_data.py    # Step 1: Data collection
├── train_model.py               # Step 2: Model training
├── predict_stock.py             # Step 3a: Single prediction
├── batch_predict.py             # Step 3b: Batch predictions
├── visualize_training.py        # View training curves
├── data/
│   └── training_data_multihorizon.csv
├── models/
│   └── multihorizon/
│       ├── best_model.pth
│       └── training_results.json
└── predictions/
    ├── latest_predictions.csv
    └── report/
        ├── dashboard.png
        ├── calibration_curves.png
        └── ...
```

## Quick Reference

```bash
# 1. Collect data (run once)
python collect_training_data.py

# 2. Train model (run once, or monthly to update)
python train_model.py

# 3. Predict single stock
python predict_stock.py AAPL

# 4. Analyze multiple stocks
python batch_predict.py

# 5. View training history
python visualize_training.py
```

## Support

- Full documentation: `TRAINING_PIPELINE_GUIDE.md`
- Detailed examples: `QUICKSTART.md`
- Code documentation: See docstrings in `stockbot/ml/` modules
