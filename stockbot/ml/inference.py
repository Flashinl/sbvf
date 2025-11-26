"""
Multi-Horizon Inference Module

Loads trained model and makes predictions for any stock ticker

Usage:
    from stockbot.ml.inference import MultiHorizonPredictor

    predictor = MultiHorizonPredictor('models/multihorizon/split_1/best_model.pth')
    result = predictor.predict('AAPL')

    print(result)
    # {
    #   '1week': {'signal': 'BUY', 'probabilities': {...}, 'expected_return': 4.2, ...},
    #   '1month': {'signal': 'HOLD', 'probabilities': {...}, 'expected_return': 1.5, ...},
    #   '3month': {'signal': 'BUY', 'probabilities': {...}, 'expected_return': 8.1, ...}
    # }
"""

import torch
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

from .models.transformer_multihorizon import MultiHorizonStockModel, create_multihorizon_model
from .data_collection import compute_technical_indicators, fetch_vix
from .catalyst_detection import ExplanationGenerator


class MultiHorizonPredictor:
    """
    Predictor for multi-horizon stock signals

    Handles:
    - Model loading
    - Data fetching and preprocessing
    - Feature normalization
    - Prediction generation
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        temperature: float = 1.0
    ):
        """
        Args:
            model_path: Path to trained model checkpoint
            device: 'cpu' or 'cuda' (auto-detect if None)
            temperature: Temperature for probability calibration
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.temperature = temperature

        # Load checkpoint
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Extract config
        self.model_config = checkpoint['model_config']
        self.feature_columns = checkpoint['feature_columns']
        self.feature_mean = checkpoint['feature_mean']
        self.feature_std = checkpoint['feature_std']

        # Create and load model
        self.model = create_multihorizon_model(
            input_dim=len(self.feature_columns),
            encoder_type=self.model_config['encoder_type'],
            d_model=self.model_config['d_model'],
            device=self.device
        )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"Model loaded successfully ({self.model_config['encoder_type']} encoder)")
        print(f"Sequence length: {self.model_config['sequence_length']}")
        print(f"Features: {len(self.feature_columns)}")

    def fetch_and_prepare_data(
        self,
        ticker: str,
        sequence_length: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical data and compute features

        Args:
            ticker: Stock symbol
            sequence_length: Required sequence length (defaults to model config)

        Returns:
            DataFrame with features, or None if fetch failed
        """
        if sequence_length is None:
            sequence_length = self.model_config['sequence_length']

        # Fetch extra data to ensure we have enough after computing indicators
        lookback_days = sequence_length + 250  # Extra for technical indicators

        try:
            # Download OHLCV data
            stock = yf.Ticker(ticker)
            hist = stock.history(period=f"{lookback_days}d", auto_adjust=False)

            if hist.empty or len(hist) < sequence_length + 50:
                print(f"Insufficient data for {ticker}")
                return None

            # Rename columns
            hist = hist.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # Keep only OHLCV
            hist = hist[['open', 'high', 'low', 'close', 'volume']].copy()
            hist['date'] = hist.index
            hist.reset_index(drop=True, inplace=True)

            # Compute technical indicators
            hist = compute_technical_indicators(hist)

            # Add market context
            hist['vix'] = fetch_vix()

            # Add placeholder sector correlations (in production, fetch real values)
            for sector in ['tech', 'health', 'finance', 'energy', 'consumer']:
                hist[f'corr_{sector}'] = 0.0

            return hist

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None

    def prepare_features(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Prepare features from DataFrame

        Args:
            df: DataFrame with technical indicators

        Returns:
            Tensor of shape (1, sequence_length, num_features)
        """
        sequence_length = self.model_config['sequence_length']

        # Get last sequence_length rows
        if len(df) < sequence_length:
            raise ValueError(f"Insufficient data: {len(df)} < {sequence_length}")

        df_sequence = df.iloc[-sequence_length:].copy()

        # Extract features in correct order
        features = []
        for col in self.feature_columns:
            if col in df_sequence.columns:
                features.append(df_sequence[col].values)
            else:
                # Missing column, fill with zeros
                features.append(np.zeros(sequence_length))

        features = np.array(features).T  # Shape: (sequence_length, num_features)

        # Handle NaN
        features = np.nan_to_num(features, nan=0.0)

        # Normalize using fitted parameters
        if self.feature_mean is not None:
            features = (features - self.feature_mean) / self.feature_std

        # Convert to tensor and add batch dimension
        features_tensor = torch.FloatTensor(features).unsqueeze(0)  # (1, seq_len, features)

        return features_tensor

    def predict(
        self,
        ticker: str,
        return_raw: bool = False
    ) -> Dict[str, Dict]:
        """
        Make multi-horizon prediction for a ticker

        Args:
            ticker: Stock symbol
            return_raw: If True, return raw model outputs

        Returns:
            Dictionary mapping horizon -> prediction dict with:
            {
                'signal': 'BUY' | 'HOLD' | 'SELL',
                'probabilities': {'BUY': 0.6, 'HOLD': 0.3, 'SELL': 0.1},
                'expected_return': float,
                'confidence': float (max probability)
            }
        """
        # Fetch and prepare data
        df = self.fetch_and_prepare_data(ticker)
        if df is None:
            raise ValueError(f"Could not fetch data for {ticker}")

        # Get current price
        current_price = df['close'].iloc[-1]

        # Prepare features
        features = self.prepare_features(df).to(self.device)

        # Make prediction
        with torch.no_grad():
            predictions = self.model.predict(features, temperature=self.temperature)

        # Add current price and ticker to results
        for horizon in predictions:
            predictions[horizon]['ticker'] = ticker
            predictions[horizon]['current_price'] = float(current_price)
            predictions[horizon]['prediction_date'] = datetime.now().isoformat()

        if return_raw:
            # Also return raw model outputs
            raw_outputs = self.model.forward(features, return_probabilities=True)
            return {
                'predictions': predictions,
                'raw_outputs': raw_outputs,
                'current_price': current_price
            }

        return predictions

    def predict_batch(
        self,
        tickers: List[str],
        show_progress: bool = True
    ) -> Dict[str, Dict]:
        """
        Predict for multiple tickers

        Args:
            tickers: List of stock symbols
            show_progress: Print progress

        Returns:
            Dict mapping ticker -> predictions
        """
        results = {}

        for idx, ticker in enumerate(tickers, 1):
            if show_progress:
                print(f"[{idx}/{len(tickers)}] Predicting {ticker}...", end=' ')

            try:
                predictions = self.predict(ticker)
                results[ticker] = predictions

                if show_progress:
                    # Show 1-month signal as preview
                    signal_1m = predictions['1month']['signal']
                    conf_1m = predictions['1month']['confidence']
                    print(f"[OK] (1M: {signal_1m} @ {conf_1m:.2f})")

            except Exception as e:
                if show_progress:
                    print(f"[FAIL] Error: {e}")
                results[ticker] = {'error': str(e)}

        return results

    def predict_to_dataframe(
        self,
        tickers: List[str]
    ) -> pd.DataFrame:
        """
        Predict for multiple tickers and return as DataFrame

        Args:
            tickers: List of stock symbols

        Returns:
            DataFrame with columns: ticker, horizon, signal, confidence, expected_return, etc.
        """
        results = self.predict_batch(tickers, show_progress=True)

        rows = []
        for ticker, predictions in results.items():
            if 'error' in predictions:
                continue

            for horizon, pred_dict in predictions.items():
                if horizon in ['1week', '1month', '3month']:
                    row = {
                        'ticker': ticker,
                        'horizon': horizon,
                        'signal': pred_dict['signal'],
                        'confidence': pred_dict['confidence'],
                        'expected_return': pred_dict['expected_return'],
                        'current_price': pred_dict.get('current_price', 0),
                        'prediction_date': pred_dict.get('prediction_date', ''),
                        'prob_buy': pred_dict['probabilities'].get('BUY', 0),
                        'prob_hold': pred_dict['probabilities'].get('HOLD', 0),
                        'prob_sell': pred_dict['probabilities'].get('SELL', 0)
                    }
                    rows.append(row)

        return pd.DataFrame(rows)

    def predict_with_explanation(
        self,
        ticker: str,
        news: Optional[List] = None,
        horizon: str = '1month',
        fetch_news: bool = True
    ) -> Dict[str, Any]:
        """
        Make prediction with catalyst detection and reasoning

        Args:
            ticker: Stock symbol
            news: Optional list of NewsItem objects (if None, will fetch)
            horizon: Time horizon ('1week', '1month', '3month')
            fetch_news: Whether to fetch news if not provided

        Returns:
            Dict with predictions + catalysts + reasoning
        """
        # Make ML prediction
        predictions = self.predict(ticker)

        # Fetch news if not provided
        if news is None and fetch_news:
            import os
            from ..providers.news_newsapi import fetch_news_newsapi
            from ..providers.news_finnhub import fetch_news_finnhub

            news_items = []

            newsapi_key = os.environ.get('NEWSAPI_KEY')
            if newsapi_key:
                news_items.extend(
                    fetch_news_newsapi(ticker, newsapi_key, max_items=10)
                )

            finnhub_key = os.environ.get('FINNHUB_API_KEY')
            if finnhub_key:
                news_items.extend(
                    fetch_news_finnhub(ticker, finnhub_key, max_items=10)
                )

            news = news_items

        # Generate explanation with catalysts
        if news:
            generator = ExplanationGenerator()
            explanation = generator.generate_explanation(
                ticker=ticker,
                ml_prediction=predictions,
                news=news,
                horizon=horizon
            )
        else:
            # No news available, return predictions with basic explanation
            pred = predictions.get(horizon, {})
            explanation = {
                'ticker': ticker,
                'signal': pred.get('signal', 'HOLD'),
                'horizon': horizon,
                'reasoning': f"Prediction based on technical analysis. No news catalysts available.",
                'catalysts': [],
                'key_factors': [
                    {
                        'category': 'model_confidence',
                        'name': 'ML Model Confidence',
                        'description': f"Model confidence: {pred.get('confidence', 0.5):.1%}",
                        'direction': 'positive',
                        'weight': pred.get('confidence', 0.5),
                        'alignment': 'supporting'
                    }
                ],
                'confidence': pred.get('confidence', 0.5),
                'ml_confidence': pred.get('confidence', 0.5),
                'catalyst_support': 0.5,
                'expected_return': pred.get('expected_return', 0.0)
            }

        # Add full prediction data
        explanation['predictions_all_horizons'] = predictions

        return explanation


def predict_single_stock(
    ticker: str,
    model_path: str,
    verbose: bool = True
) -> Dict:
    """
    Convenience function to predict for a single stock

    Args:
        ticker: Stock symbol
        model_path: Path to trained model
        verbose: Print results

    Returns:
        Predictions dictionary
    """
    predictor = MultiHorizonPredictor(model_path)
    predictions = predictor.predict(ticker)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Predictions for {ticker}")
        print(f"{'='*60}")
        print(f"Current Price: ${predictions['1week']['current_price']:.2f}")
        print(f"Prediction Date: {predictions['1week']['prediction_date']}")

        for horizon in ['1week', '1month', '3month']:
            pred = predictions[horizon]
            print(f"\n{horizon.upper()}:")
            print(f"  Signal: {pred['signal']} (confidence: {pred['confidence']:.2%})")
            print(f"  Expected Return: {pred['expected_return']:+.2f}%")
            print(f"  Probabilities:")
            for signal, prob in pred['probabilities'].items():
                print(f"    {signal}: {prob:.2%}")

        print(f"{'='*60}\n")

    return predictions


if __name__ == '__main__':
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m stockbot.ml.inference <TICKER> [MODEL_PATH]")
        print("\nExample:")
        print("  python -m stockbot.ml.inference AAPL models/multihorizon/split_1/best_model.pth")
        sys.exit(1)

    ticker = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else 'models/multihorizon/split_1/best_model.pth'

    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print("\nTrain a model first:")
        print("  python -m stockbot.ml.train_multihorizon --data-path data/training_data.csv")
        sys.exit(1)

    # Make prediction
    predictions = predict_single_stock(ticker, model_path, verbose=True)
