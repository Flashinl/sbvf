"""
ML Service - Integration layer between ML models and API

Provides a clean interface for the API to get ML predictions without
dealing with model loading, error handling, etc.
"""

from typing import Dict, Optional
from pathlib import Path
import asyncio
from functools import lru_cache

from .ml.inference import MultiHorizonPredictor


class MLService:
    """
    Singleton service for ML predictions

    Handles model loading, caching, and provides async interface for API
    """

    _instance: Optional['MLService'] = None
    _predictor: Optional[MultiHorizonPredictor] = None
    _model_loaded: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the ML service"""
        pass

    def load_model(self, model_path: str = 'models/multihorizon/best_model.pth'):
        """
        Load the ML model (call once at startup)

        Args:
            model_path: Path to the trained model
        """
        if self._model_loaded:
            return

        model_file = Path(model_path)
        if not model_file.exists():
            print(f"[WARNING] ML model not found at {model_path}")
            print("ML predictions will not be available.")
            print("Train a model with: python train_model.py")
            return

        try:
            self._predictor = MultiHorizonPredictor(model_path)
            self._model_loaded = True
            print(f"[OK] ML model loaded from {model_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load ML model: {e}")
            self._model_loaded = False

    def is_available(self) -> bool:
        """Check if ML predictions are available"""
        return self._model_loaded and self._predictor is not None

    async def predict(self, ticker: str) -> Optional[Dict]:
        """
        Get ML predictions for a ticker (async)

        Args:
            ticker: Stock symbol

        Returns:
            Predictions dict or None if unavailable
        """
        if not self.is_available():
            return None

        try:
            # Run prediction in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            predictions = await loop.run_in_executor(
                None,
                self._predictor.predict,
                ticker
            )
            return predictions
        except Exception as e:
            print(f"[ERROR] ML prediction failed for {ticker}: {e}")
            return None

    async def predict_batch(self, tickers: list) -> Dict[str, Dict]:
        """
        Get ML predictions for multiple tickers (async)

        Args:
            tickers: List of stock symbols

        Returns:
            Dict mapping ticker -> predictions
        """
        if not self.is_available():
            return {}

        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                self._predictor.predict_batch,
                tickers,
                False  # show_progress=False
            )
            return results
        except Exception as e:
            print(f"[ERROR] Batch ML prediction failed: {e}")
            return {}

    async def predict_with_explanation(
        self,
        ticker: str,
        news: Optional[list] = None,
        horizon: str = '1month',
        fetch_news: bool = True
    ) -> Optional[Dict]:
        """
        Get ML prediction with catalyst detection and reasoning (async)

        Args:
            ticker: Stock symbol
            news: Optional list of NewsItem objects (if None, will fetch if fetch_news=True)
            horizon: Time horizon ('1week', '1month', '3month')
            fetch_news: Whether to fetch and analyze news if not provided

        Returns:
            Dict with:
            - signal: BUY/HOLD/SELL
            - reasoning: Human-readable explanation
            - catalysts: Detected business catalysts
            - confidence: Overall confidence score
            - key_factors: Top factors driving prediction
        """
        if not self.is_available():
            return None

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._predictor.predict_with_explanation,
                ticker,
                news,
                horizon,
                fetch_news
            )
            return result
        except Exception as e:
            print(f"[ERROR] ML prediction with explanation failed for {ticker}: {e}")
            return None

    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        if not self.is_available():
            return {
                'available': False,
                'message': 'ML model not loaded'
            }

        return {
            'available': True,
            'encoder_type': self._predictor.model_config['encoder_type'],
            'd_model': self._predictor.model_config['d_model'],
            'sequence_length': self._predictor.model_config['sequence_length'],
            'horizons': self._predictor.model_config['horizons'],
            'num_features': len(self._predictor.feature_columns)
        }


# Global instance
ml_service = MLService()


def init_ml_service(model_path: str = 'models/multihorizon/best_model.pth'):
    """
    Initialize ML service at application startup

    Usage:
        from stockbot.ml_service import init_ml_service
        init_ml_service()
    """
    ml_service.load_model(model_path)


def get_ml_service() -> MLService:
    """
    Get the ML service instance

    Usage:
        from stockbot.ml_service import get_ml_service

        ml = get_ml_service()
        if ml.is_available():
            predictions = await ml.predict('AAPL')
    """
    return ml_service
