# Multi-Horizon Stock Prediction Training Pipeline

Complete guide for training, evaluating, and deploying multi-horizon stock prediction models.

## Overview

This pipeline provides:
- **Multi-horizon predictions**: 1 week, 1 month, 3 months
- **Dual outputs per horizon**:
  - Classification: BUY/HOLD/SELL with calibrated probabilities
  - Regression: Expected return percentage
- **Architectures**: Transformer encoder or LSTM
- **Walk-forward validation**: Prevents lookahead bias
- **Class weighting**: Handles imbalanced data
- **Comprehensive evaluation**: Calibration plots, per-class metrics, backtesting

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Collect Training Data

```bash
# Collect data for S&P 500 stocks (this will take a while)
python -m stockbot.ml.data_collection
```

Or customize the data collection:

```python
from stockbot.ml.data_collection import prepare_dataset, save_dataset
from stockbot.ml.sp500_tickers import TOP_100_TICKERS

# Collect data for top 100 S&P 500 stocks
dataset = prepare_dataset(
    tickers=TOP_100_TICKERS,
    start_date='2020-01-01',
    end_date=None,  # Today
    include_sector_correlation=False,  # Set True for full features (slower)
    delay=0.5,  # Delay between API calls to avoid rate limits
    verbose=True
)

# Save dataset
save_dataset(dataset, 'data/training_data_multihorizon.csv')
```

**Output**: CSV file with columns:
- OHLCV features (open, high, low, close, volume)
- Technical indicators (RSI, MACD, Bollinger Bands, ATR, SMAs, etc.)
- Market context (VIX, sector correlations)
- Multi-horizon labels:
  - `target_return_1week`, `target_signal_1week`, `target_class_1week`
  - `target_return_1month`, `target_signal_1month`, `target_class_1month`
  - `target_return_3month`, `target_signal_3month`, `target_class_3month`

### 2. Train Model

```bash
# Train with default settings (Transformer, 100 epochs)
python -m stockbot.ml.train_multihorizon \
    --data-path data/training_data_multihorizon.csv \
    --save-dir models/multihorizon \
    --epochs 100

# Train with LSTM encoder
python -m stockbot.ml.train_multihorizon \
    --data-path data/training_data_multihorizon.csv \
    --encoder-type lstm \
    --save-dir models/multihorizon_lstm \
    --epochs 100

# Train with custom settings
python -m stockbot.ml.train_multihorizon \
    --data-path data/training_data_multihorizon.csv \
    --encoder-type transformer \
    --d-model 256 \
    --batch-size 64 \
    --learning-rate 0.0005 \
    --epochs 150 \
    --early-stopping-patience 20
```

**Training Features**:
- Walk-forward validation (expanding window)
- Class weighting for imbalanced data
- Early stopping based on validation loss
- Learning rate scheduling (ReduceLROnPlateau)
- Gradient clipping
- Model checkpointing (saves best model)

**Output**:
- `models/multihorizon/split_1/best_model.pth` - Trained model checkpoint
- `models/multihorizon/split_1/training_results.json` - Training history

### 3. Evaluate Model

```python
from stockbot.ml.evaluation import evaluate_model_predictions
from stockbot.ml.visualization import plot_training_history

# Plot training curves
plot_training_history(
    'models/multihorizon/split_1/training_results.json',
    save_path='results/training_history.png'
)

# Evaluate on test set (predictions_dict from training)
# This is automatically done during training, but you can also do it separately
metrics = evaluate_model_predictions(
    test_predictions,
    output_dir='results/evaluation',
    show_plots=True
)
```

**Evaluation Metrics**:

**Classification (per horizon)**:
- Per-class precision, recall, F1-score
- Confusion matrices
- Overall accuracy

**Regression (per horizon)**:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- Directional accuracy (did we predict the right direction?)
- R² score

**Calibration (per horizon)**:
- Expected Calibration Error (ECE)
- Reliability diagrams
- Bin-wise confidence vs accuracy

### 4. Make Predictions

```bash
# Predict for a single stock
python -m stockbot.ml.inference AAPL models/multihorizon/split_1/best_model.pth
```

Or use the API:

```python
from stockbot.ml.inference import MultiHorizonPredictor

# Load model
predictor = MultiHorizonPredictor('models/multihorizon/split_1/best_model.pth')

# Predict for single stock
predictions = predictor.predict('AAPL')
print(predictions)
# {
#   '1week': {
#     'signal': 'BUY',
#     'probabilities': {'BUY': 0.65, 'HOLD': 0.25, 'SELL': 0.10},
#     'expected_return': 4.2,
#     'confidence': 0.65,
#     'ticker': 'AAPL',
#     'current_price': 178.50,
#     'prediction_date': '2025-11-25T...'
#   },
#   '1month': {...},
#   '3month': {...}
# }

# Predict for multiple stocks
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
batch_predictions = predictor.predict_batch(tickers, show_progress=True)

# Get as DataFrame
predictions_df = predictor.predict_to_dataframe(tickers)
predictions_df.to_csv('predictions/latest.csv', index=False)
```

### 5. Visualize Predictions

```python
from stockbot.ml.visualization import (
    plot_prediction_summary_dashboard,
    plot_multi_horizon_comparison,
    plot_top_signals,
    create_prediction_report
)

# Dashboard of all predictions
plot_prediction_summary_dashboard(
    predictions_df,
    save_path='results/dashboard.png'
)

# Compare horizons for single stock
plot_multi_horizon_comparison(
    predictions_df,
    ticker='AAPL',
    save_path='results/aapl_comparison.png'
)

# Top BUY signals
plot_top_signals(
    predictions_df,
    horizon='1month',
    top_n=15,
    signal_type='BUY',
    save_path='results/top_buys.png'
)

# Generate complete report
create_prediction_report(predictions_df, output_dir='results/prediction_report')
```

### 6. Backtest Strategy

```python
from stockbot.ml.backtest import (
    backtest_signals,
    backtest_buy_and_hold,
    compare_strategies,
    plot_backtest_results,
    BacktestConfig
)

# Prepare data for backtesting
# predictions_df should have: date, ticker, signal_1month, confidence_1month
# price_data should have: date, ticker, close

config = BacktestConfig(
    initial_capital=100000.0,
    transaction_cost=0.001,  # 0.1%
    position_size=0.1,  # 10% per position
    min_confidence=0.6,
    stop_loss=-0.10,
    take_profit=0.20
)

# Backtest signal-based strategy
signal_portfolio, signal_equity = backtest_signals(
    predictions_df,
    price_data,
    config,
    horizon='1month'
)

# Backtest buy-and-hold
buyhold_final, buyhold_equity = backtest_buy_and_hold(
    price_data,
    config.initial_capital
)

# Compare strategies
comparison = compare_strategies(
    signal_portfolio,
    signal_equity,
    buyhold_equity,
    config
)

# Plot results
plot_backtest_results(
    signal_equity,
    buyhold_equity,
    comparison,
    save_path='results/backtest.png'
)

print(f"Signal Strategy Return: {comparison['signal_strategy']['total_return_pct']:.2f}%")
print(f"Buy & Hold Return: {comparison['buy_and_hold']['total_return_pct']:.2f}%")
print(f"Outperformance: {comparison['outperformance']:.2f}%")
```

## Architecture Details

### Model Architecture

```
Input: (batch_size, sequence_length=60, num_features)
  │
  ├─> Transformer Encoder (or LSTM)
  │     - 3 encoder layers
  │     - 8 attention heads (Transformer)
  │     - d_model = 128
  │     - Positional encoding
  │
  └─> Shared Feature Representation (batch_size, d_model)
        │
        ├─> 1-Week Head
        │     ├─> Classification (3 classes): BUY/HOLD/SELL logits
        │     └─> Regression (1 value): Expected return %
        │
        ├─> 1-Month Head
        │     ├─> Classification: BUY/HOLD/SELL logits
        │     └─> Regression: Expected return %
        │
        └─> 3-Month Head
              ├─> Classification: BUY/HOLD/SELL logits
              └─> Regression: Expected return %
```

### Loss Function

Combined multi-task loss:
```
Total Loss = Σ [horizon_weight_h * (α * CrossEntropy_h + β * MSE_h)]
             for h in [1week, 1month, 3month]

where:
  α = classification_weight (default: 1.0)
  β = regression_weight (default: 1.0)
  horizon_weight_h = importance weight for horizon h (default: equal)
```

Class weights are computed using scikit-learn's `compute_class_weight` with `class_weight='balanced'` to handle imbalanced data.

## Configuration Options

### Data Collection

| Parameter | Description | Default |
|-----------|-------------|---------|
| `start_date` | Start date for historical data | '2020-01-01' |
| `end_date` | End date (None = today) | None |
| `min_history_days` | Minimum days of history | 250 |
| `include_sector_correlation` | Fetch sector ETF correlations | False |
| `delay` | Delay between API calls (seconds) | 0.5 |

### Training

| Parameter | Description | Default |
|-----------|-------------|---------|
| `encoder_type` | 'transformer' or 'lstm' | 'transformer' |
| `d_model` | Model hidden dimension | 128 |
| `sequence_length` | Input sequence length | 60 |
| `epochs` | Maximum training epochs | 100 |
| `batch_size` | Batch size | 32 |
| `learning_rate` | Initial learning rate | 0.001 |
| `weight_decay` | L2 regularization | 1e-5 |
| `early_stopping_patience` | Patience for early stopping | 15 |
| `classification_weight` | Weight for classification loss | 1.0 |
| `regression_weight` | Weight for regression loss | 1.0 |

### Signal Thresholds

| Threshold | Value | Description |
|-----------|-------|-------------|
| BUY | return > +3% | Positive outlook |
| HOLD | -3% ≤ return ≤ +3% | Neutral outlook |
| SELL | return < -3% | Negative outlook |

These can be customized in `stockbot/ml/data_collection.py` by modifying the `HorizonConfig` dataclass.

## Files and Modules

```
stockbot/ml/
├── data_collection.py          # Data fetching and labeling
├── models/
│   └── transformer_multihorizon.py  # Model architecture
├── train_multihorizon.py       # Training pipeline
├── evaluation.py               # Evaluation metrics
├── backtest.py                 # Backtesting framework
├── inference.py                # Prediction interface
└── visualization.py            # Plotting utilities
```

## Performance Tips

### Training Speed
- Use GPU if available (automatically detected)
- Reduce `d_model` for faster training (e.g., 64 or 128)
- Use LSTM encoder (faster than Transformer)
- Reduce `sequence_length` to 30 or 40

### Memory Usage
- Reduce `batch_size` if running out of memory
- Use gradient checkpointing (modify model code)

### Data Collection
- Use `include_sector_correlation=False` for faster collection
- Increase `delay` if hitting rate limits
- Collect data in batches (e.g., 50 tickers at a time)

## Troubleshooting

### Issue: "Insufficient data for ticker X"
**Solution**: Some tickers may not have enough historical data. Filter them out or use more recent tickers.

### Issue: "CUDA out of memory"
**Solution**: Reduce `batch_size` or use CPU by setting device='cpu' in model creation.

### Issue: "Rate limit exceeded"
**Solution**: Increase `delay` parameter in data collection (e.g., `delay=1.0`).

### Issue: "Model not converging"
**Solution**:
- Try different learning rates (e.g., 0.0001 or 0.0005)
- Increase model capacity (`d_model=256`)
- Check data quality (ensure no NaN values)
- Increase training data size

## Next Steps

1. **Hyperparameter Tuning**: Experiment with different architectures and hyperparameters
2. **Feature Engineering**: Add more features (e.g., news sentiment, SEC filings)
3. **Ensemble Models**: Combine Transformer and LSTM predictions
4. **Online Learning**: Retrain periodically with new data
5. **Production Deployment**: Integrate with existing API (`stockbot/api.py`)

## Example: Complete Workflow

```python
# 1. Collect data
from stockbot.ml.data_collection import prepare_dataset, save_dataset
from stockbot.ml.sp500_tickers import TOP_100_TICKERS

dataset = prepare_dataset(
    tickers=TOP_100_TICKERS[:50],  # Top 50 stocks
    start_date='2020-01-01',
    verbose=True
)
save_dataset(dataset, 'data/my_training_data.csv')

# 2. Train model (run in terminal)
# python -m stockbot.ml.train_multihorizon --data-path data/my_training_data.csv --epochs 100

# 3. Make predictions
from stockbot.ml.inference import MultiHorizonPredictor

predictor = MultiHorizonPredictor('models/multihorizon/split_1/best_model.pth')
predictions_df = predictor.predict_to_dataframe(['AAPL', 'MSFT', 'GOOGL'])

# 4. Visualize
from stockbot.ml.visualization import create_prediction_report
create_prediction_report(predictions_df, 'results/my_report')

# 5. Backtest (prepare your price data first)
# See backtest section above
```

## References

- **Walk-Forward Validation**: Prevents lookahead bias by using expanding window
- **Calibration**: Ensures predicted probabilities match actual frequencies
- **Multi-Task Learning**: Shared representation for multiple prediction horizons
- **Transformer**: Attention-based architecture for sequence modeling

## Support

For questions or issues, please refer to the code documentation or create an issue in the repository.
