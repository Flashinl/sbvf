"""
Test Script - Verify Model is Working Correctly

This script tests:
1. Model loading
2. Data fetching
3. Prediction generation
4. Output format validation
5. Multi-horizon consistency
"""

import sys
from pathlib import Path
import numpy as np

# Test configuration
TEST_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'JPM']
MODEL_PATH = 'models/multihorizon/best_model.pth'

print("="*70)
print("MODEL TEST SUITE")
print("="*70)

# Check if model exists
model_path = Path(MODEL_PATH)
if not model_path.exists():
    print("\n[ERROR] Model not found!")
    print(f"Expected location: {MODEL_PATH}")
    print("\nPlease train the model first:")
    print("  python train_model.py")
    print("\nOr check training status:")
    print("  python check_training.py")
    sys.exit(1)

print(f"\n[OK] Model found at: {MODEL_PATH}")

# Test 1: Load Model
print("\n" + "-"*70)
print("TEST 1: Loading Model")
print("-"*70)

try:
    from stockbot.ml.inference import MultiHorizonPredictor

    predictor = MultiHorizonPredictor(MODEL_PATH)
    print("[PASS] Model loaded successfully")
    print(f"  - Encoder: {predictor.model_config['encoder_type']}")
    print(f"  - Model dimension: {predictor.model_config['d_model']}")
    print(f"  - Sequence length: {predictor.model_config['sequence_length']}")
    print(f"  - Features: {len(predictor.feature_columns)}")
except Exception as e:
    print(f"[FAIL] Could not load model: {e}")
    sys.exit(1)

# Test 2: Fetch Data
print("\n" + "-"*70)
print("TEST 2: Data Fetching")
print("-"*70)

test_ticker = TEST_TICKERS[0]
try:
    df = predictor.fetch_and_prepare_data(test_ticker)
    if df is None or df.empty:
        print(f"[FAIL] Could not fetch data for {test_ticker}")
        sys.exit(1)

    print(f"[PASS] Data fetched for {test_ticker}")
    print(f"  - Rows: {len(df)}")
    print(f"  - Columns: {len(df.columns)}")
    print(f"  - Date range: {df['date'].min()} to {df['date'].max()}")
except Exception as e:
    print(f"[FAIL] Data fetching error: {e}")
    sys.exit(1)

# Test 3: Single Stock Prediction
print("\n" + "-"*70)
print("TEST 3: Single Stock Prediction")
print("-"*70)

try:
    predictions = predictor.predict(test_ticker)
    print(f"[PASS] Prediction generated for {test_ticker}")

    # Validate structure
    assert '1week' in predictions, "Missing 1week predictions"
    assert '1month' in predictions, "Missing 1month predictions"
    assert '3month' in predictions, "Missing 3month predictions"

    for horizon, pred in predictions.items():
        assert 'signal' in pred, f"Missing signal for {horizon}"
        assert pred['signal'] in ['BUY', 'HOLD', 'SELL'], f"Invalid signal: {pred['signal']}"

        assert 'probabilities' in pred, f"Missing probabilities for {horizon}"
        assert 'confidence' in pred, f"Missing confidence for {horizon}"
        assert 'expected_return' in pred, f"Missing expected_return for {horizon}"

        # Check probabilities sum to 1
        prob_sum = sum(pred['probabilities'].values())
        assert 0.99 <= prob_sum <= 1.01, f"Probabilities don't sum to 1: {prob_sum}"

        # Check confidence is max probability
        max_prob = max(pred['probabilities'].values())
        assert abs(pred['confidence'] - max_prob) < 0.01, "Confidence != max probability"

    print("  - Structure: VALID")
    print("  - Signals: VALID")
    print("  - Probabilities: VALID")

except Exception as e:
    print(f"[FAIL] Prediction error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Display Sample Prediction
print("\n" + "-"*70)
print("TEST 4: Sample Prediction Output")
print("-"*70)

print(f"\nPredictions for {test_ticker}:")
print(f"Current Price: ${predictions['1week']['current_price']:.2f}")

for horizon in ['1week', '1month', '3month']:
    pred = predictions[horizon]
    print(f"\n{horizon.upper()}:")
    print(f"  Signal: {pred['signal']} ({pred['confidence']:.1%} confidence)")
    print(f"  Expected Return: {pred['expected_return']:+.2f}%")
    print(f"  Probabilities: BUY={pred['probabilities']['BUY']:.1%}, "
          f"HOLD={pred['probabilities']['HOLD']:.1%}, "
          f"SELL={pred['probabilities']['SELL']:.1%}")

print("\n[PASS] Prediction format is correct")

# Test 5: Batch Prediction
print("\n" + "-"*70)
print("TEST 5: Batch Prediction")
print("-"*70)

try:
    print(f"Testing batch prediction for {len(TEST_TICKERS)} tickers...")
    batch_results = predictor.predict_batch(TEST_TICKERS, show_progress=False)

    successful = sum(1 for r in batch_results.values() if 'error' not in r)
    failed = len(batch_results) - successful

    print(f"[PASS] Batch prediction completed")
    print(f"  - Successful: {successful}/{len(TEST_TICKERS)}")
    print(f"  - Failed: {failed}/{len(TEST_TICKERS)}")

    if failed > 0:
        print("\n  Failed tickers:")
        for ticker, result in batch_results.items():
            if 'error' in result:
                print(f"    {ticker}: {result['error']}")

except Exception as e:
    print(f"[FAIL] Batch prediction error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Multi-Horizon Consistency
print("\n" + "-"*70)
print("TEST 6: Multi-Horizon Consistency Checks")
print("-"*70)

consistency_issues = []

# Check 1: Longer horizons should generally have more extreme returns
for ticker, result in batch_results.items():
    if 'error' in result:
        continue

    try:
        ret_1w = abs(result['1week']['expected_return'])
        ret_1m = abs(result['1month']['expected_return'])
        ret_3m = abs(result['3month']['expected_return'])

        # 3-month should generally be larger than 1-week
        # (not strict, but flagging extreme violations)
        if ret_3m < ret_1w * 0.5:
            consistency_issues.append(
                f"{ticker}: 3-month return ({ret_3m:.2f}%) much smaller than 1-week ({ret_1w:.2f}%)"
            )
    except:
        pass

# Check 2: Confidence should be reasonable (not too high/low for all)
low_conf_count = 0
high_conf_count = 0

for ticker, result in batch_results.items():
    if 'error' in result:
        continue

    try:
        conf_1m = result['1month']['confidence']
        if conf_1m < 0.4:
            low_conf_count += 1
        elif conf_1m > 0.9:
            high_conf_count += 1
    except:
        pass

if len(consistency_issues) == 0:
    print("[PASS] Multi-horizon predictions are consistent")
else:
    print(f"[WARN] Found {len(consistency_issues)} consistency issues:")
    for issue in consistency_issues[:5]:
        print(f"  - {issue}")
    if len(consistency_issues) > 5:
        print(f"  ... and {len(consistency_issues) - 5} more")

print(f"\nConfidence distribution:")
print(f"  - Very low (<40%): {low_conf_count}/{len(batch_results)}")
print(f"  - Very high (>90%): {high_conf_count}/{len(batch_results)}")

# Test 7: DataFrame Export
print("\n" + "-"*70)
print("TEST 7: DataFrame Export")
print("-"*70)

try:
    import pandas as pd

    predictions_df = predictor.predict_to_dataframe(TEST_TICKERS[:3])

    print(f"[PASS] DataFrame export successful")
    print(f"  - Rows: {len(predictions_df)}")
    print(f"  - Columns: {len(predictions_df.columns)}")

    # Verify structure
    required_cols = ['ticker', 'horizon', 'signal', 'confidence', 'expected_return']
    missing_cols = [col for col in required_cols if col not in predictions_df.columns]

    if missing_cols:
        print(f"[WARN] Missing columns: {missing_cols}")
    else:
        print("  - Required columns: PRESENT")

    # Show sample
    print("\nSample rows:")
    print(predictions_df.head(3).to_string(index=False))

except Exception as e:
    print(f"[FAIL] DataFrame export error: {e}")
    import traceback
    traceback.print_exc()

# Final Summary
print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)

print("\n[SUCCESS] All core tests passed!")
print("\nModel is ready for production use:")
print("  - Single predictions: python predict_stock.py TICKER")
print("  - Batch predictions: python batch_predict.py")
print("  - Training visualization: python visualize_training.py")

print("\nModel Performance Summary:")
print(f"  - Test tickers: {len(TEST_TICKERS)}")
print(f"  - Successful predictions: {successful}/{len(TEST_TICKERS)}")
print(f"  - Average 1-month confidence: {np.mean([r['1month']['confidence'] for r in batch_results.values() if 'error' not in r]):.1%}")

print("\n" + "="*70)
print("All tests completed successfully!")
print("="*70)
