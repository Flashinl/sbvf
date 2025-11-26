"""
Main Prediction Engine

Orchestrates the entire prediction pipeline:
1. Fetch historical data
2. Extract features
3. Run ensemble model
4. Generate SHAP explanations
5. Create specific, actionable explanations
6. Save to database
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

from .feature_engineering import extract_all_features
from .models.ensemble import EnsembleStockModel, EnsemblePrediction, create_ensemble_model
from .shap_explainer import create_explainer, FEATURE_METADATA
from .explainer import SpecificExplainer, ExplanationFactor


class StockPredictor:
    """
    Main prediction engine - combines all ML components

    Usage:
        predictor = StockPredictor()
        predictor.load_models('models/lstm.pth', 'models/xgboost.pkl')
        result = predictor.predict('AAPL')
    """

    def __init__(
        self,
        ensemble_model: Optional[EnsembleStockModel] = None,
        models_dir: str = 'models'
    ):
        self.ensemble_model = ensemble_model
        self.models_dir = Path(models_dir)
        self.shap_explainer = None

    def load_models(
        self,
        lstm_path: Optional[str] = None,
        xgboost_path: Optional[str] = None
    ):
        """Load pre-trained models"""
        if self.ensemble_model is None:
            self.ensemble_model = create_ensemble_model(
                lstm_path=lstm_path,
                xgboost_path=xgboost_path
            )
        else:
            self.ensemble_model.load_models(lstm_path or "", xgboost_path or "")

    def predict(
        self,
        ticker: str,
        news_items: Optional[List[Dict]] = None,
        historical_data: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Make complete prediction with explanations

        Args:
            ticker: Stock symbol
            news_items: Recent news articles (for sentiment)
            historical_data: Historical OHLCV data (if available)

        Returns:
            Complete prediction dict with:
            - signal: BUY/SELL/HOLD
            - predicted_return: %
            - predicted_price: $
            - confidence: 0-1
            - confidence_interval: (low, high)
            - risk_score: 1-10
            - explanations: List of ExplanationFactor objects
            - model_metadata: Model details
        """
        if self.ensemble_model is None:
            raise ValueError("Models not loaded. Call load_models() first.")

        # 1. Extract features
        print(f"Extracting features for {ticker}...")
        features = extract_all_features(
            ticker=ticker,
            news_items=news_items,
            lookback_days=252
        )

        if not features:
            raise ValueError(f"Failed to extract features for {ticker}")

        # Get current price
        current_price = features.get('price_current', 0)
        if current_price == 0:
            raise ValueError(f"Could not fetch current price for {ticker}")

        # 2. Prepare historical data for LSTM
        if historical_data is None:
            # Fetch from yfinance if not provided
            import yfinance as yf
            stock = yf.Ticker(ticker)
            historical_data = stock.history(period='1y')

        if historical_data.empty:
            raise ValueError(f"No historical data available for {ticker}")

        # 3. Make ensemble prediction
        print(f"Generating prediction for {ticker}...")
        prediction = self.ensemble_model.predict(
            ticker=ticker,
            sequence_data=historical_data,
            features=features,
            current_price=current_price
        )

        # 4. Extract SHAP values for explainability
        print(f"Generating explanations...")
        shap_values = self._get_shap_values(features, prediction)

        # 5. Generate specific, actionable explanations
        explainer = SpecificExplainer(
            ticker=ticker,
            features=features,
            shap_values=shap_values
        )

        explanation_factors = explainer.generate_top_factors(n=5)

        # 6. Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(historical_data, prediction)

        # 7. Find similar historical patterns
        historical_matches = self._find_historical_patterns(ticker, features, prediction)

        # 8. Assemble complete result
        result = {
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),

            # Prediction
            'signal': prediction.signal,
            'predicted_return_pct': prediction.predicted_return,
            'predicted_price': prediction.predicted_price,
            'current_price': current_price,

            # Confidence & Risk
            'confidence': prediction.confidence,
            'confidence_interval_low': prediction.confidence_interval_low,
            'confidence_interval_high': prediction.confidence_interval_high,
            'risk_score': prediction.risk_score,

            # Risk metrics
            'sharpe_ratio': risk_metrics['sharpe_ratio'],
            'max_drawdown': risk_metrics['max_drawdown'],
            'volatility': risk_metrics['volatility'],

            # Model details
            'model_contributions': prediction.model_contributions,
            'ensemble_agreement': prediction.ensemble_agreement,

            # Explanations
            'explanations': [self._format_explanation(f) for f in explanation_factors],

            # Historical patterns
            'similar_patterns': historical_matches,

            # Raw features (for debugging)
            'features': features
        }

        return result

    def _get_shap_values(
        self,
        features: Dict[str, float],
        prediction: EnsemblePrediction
    ) -> Dict[str, float]:
        """Get SHAP values from XGBoost model"""
        if prediction.feature_contributions:
            return prediction.feature_contributions

        # Fallback: use XGBoost feature contributions
        if self.ensemble_model.xgboost_model:
            try:
                features_df = pd.DataFrame([features])
                _, contributions = self.ensemble_model.xgboost_model.predict(
                    features_df,
                    return_feature_contributions=True
                )
                return contributions or {}
            except Exception as e:
                print(f"Error getting SHAP values: {e}")
                return {}

        return {}

    def _calculate_risk_metrics(
        self,
        historical_data: pd.DataFrame,
        prediction: EnsemblePrediction
    ) -> Dict[str, float]:
        """Calculate risk-adjusted metrics"""
        try:
            # Calculate daily returns
            returns = historical_data['Close'].pct_change().dropna()

            # Sharpe ratio (annualized)
            if len(returns) > 0 and returns.std() > 0:
                sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0

            # Max drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = abs(drawdown.min()) * 100  # Convert to %

            # Volatility (annualized)
            volatility = returns.std() * np.sqrt(252) * 100  # Convert to %

            return {
                'sharpe_ratio': round(sharpe_ratio, 2),
                'max_drawdown': round(max_drawdown, 2),
                'volatility': round(volatility, 2)
            }
        except Exception as e:
            print(f"Error calculating risk metrics: {e}")
            return {
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0
            }

    def _find_historical_patterns(
        self,
        ticker: str,
        features: Dict[str, float],
        prediction: EnsemblePrediction,
        lookback_years: int = 3
    ) -> List[Dict]:
        """
        Find similar historical patterns

        TODO: Implement proper pattern matching using:
        - Feature similarity (cosine distance)
        - Price pattern matching (DTW)
        - Outcome tracking
        """
        # Placeholder - in production, query database for similar patterns
        return [
            {
                'date': '2023-06-15',
                'outcome': '+8.2% in 30 days',
                'similarity': 0.85,
                'context': 'Similar RSI oversold condition with positive news sentiment'
            },
            {
                'date': '2023-09-22',
                'outcome': '+5.7% in 30 days',
                'similarity': 0.78,
                'context': 'Breakout above 50-day MA with volume confirmation'
            }
        ]

    def _format_explanation(self, factor: ExplanationFactor) -> Dict:
        """Format ExplanationFactor for API response"""
        return {
            'category': factor.category,
            'name': factor.name,
            'weight': round(factor.weight, 2),
            'direction': factor.direction,
            'expert_explanation': factor.expert_explanation,
            'beginner_explanation': factor.beginner_explanation,
            'supporting_data': factor.supporting_data,
            'source_url': factor.source_url,
            'event_date': factor.event_date
        }

    def predict_batch(
        self,
        tickers: List[str],
        max_workers: int = 4
    ) -> Dict[str, Dict]:
        """
        Predict for multiple stocks in parallel

        Args:
            tickers: List of stock symbols
            max_workers: Number of parallel workers

        Returns:
            Dict mapping ticker -> prediction result
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(self.predict, ticker): ticker
                for ticker in tickers
            }

            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    results[ticker] = future.result()
                    print(f"✓ {ticker} predicted successfully")
                except Exception as e:
                    print(f"✗ {ticker} failed: {e}")
                    results[ticker] = {'error': str(e)}

        return results

    def save_prediction_to_db(
        self,
        prediction_result: Dict,
        db_session
    ):
        """
        Save prediction to database

        Args:
            prediction_result: Output from predict()
            db_session: SQLAlchemy session
        """
        from ..models import (
            Prediction, PredictionExplanation, ConfidenceBreakdown,
            HistoricalPatternMatch, SignalType, FactorCategory, FactorDirection
        )
        from datetime import date

        # Create Prediction record
        pred = Prediction(
            stock_symbol=prediction_result['ticker'],
            prediction_date=date.today(),
            signal=SignalType[prediction_result['signal']],
            predicted_change_pct=prediction_result['predicted_return_pct'],
            predicted_price=prediction_result['predicted_price'],
            current_price=prediction_result['current_price'],
            confidence_score=prediction_result['confidence'],
            confidence_interval_low=prediction_result['confidence_interval_low'],
            confidence_interval_high=prediction_result['confidence_interval_high'],
            risk_score=prediction_result['risk_score'],
            sharpe_ratio=prediction_result['sharpe_ratio'],
            max_drawdown=prediction_result['max_drawdown'],
            volatility=prediction_result['volatility'],
            model_version='v1.0',
            ensemble_agreement=prediction_result['ensemble_agreement']
        )

        db_session.add(pred)
        db_session.flush()  # Get prediction ID

        # Create Explanation records
        for i, explanation in enumerate(prediction_result['explanations'], 1):
            exp = PredictionExplanation(
                prediction_id=pred.id,
                factor_rank=i,
                factor_category=FactorCategory[explanation['category'].upper()],
                factor_name=explanation['name'],
                factor_weight=explanation['weight'],
                factor_direction=FactorDirection[explanation['direction'].upper()],
                expert_explanation=explanation['expert_explanation'],
                beginner_explanation=explanation['beginner_explanation'],
                supporting_data=explanation['supporting_data']
            )
            db_session.add(exp)

        # Create Confidence Breakdown
        conf = ConfidenceBreakdown(
            prediction_id=pred.id,
            model_agreement_score=prediction_result['ensemble_agreement'],
            historical_accuracy=0.68,  # TODO: Calculate from actual history
            data_quality_score=1.0,
            volatility_adjustment=1.0 - (prediction_result['volatility'] / 100),
            final_confidence=prediction_result['confidence']
        )
        db_session.add(conf)

        # Create Historical Pattern Matches
        for pattern in prediction_result.get('similar_patterns', []):
            match = HistoricalPatternMatch(
                prediction_id=pred.id,
                historical_date=datetime.strptime(pattern['date'], '%Y-%m-%d').date(),
                historical_outcome=pattern['outcome'],
                pattern_similarity_score=pattern['similarity'],
                historical_context=pattern['context']
            )
            db_session.add(match)

        db_session.commit()
        print(f"Prediction saved to database (ID: {pred.id})")

        return pred.id


def create_predictor(
    lstm_model_path: Optional[str] = None,
    xgboost_model_path: Optional[str] = None
) -> StockPredictor:
    """
    Factory function to create predictor

    Args:
        lstm_model_path: Path to LSTM model
        xgboost_model_path: Path to XGBoost model

    Returns:
        StockPredictor instance
    """
    predictor = StockPredictor()

    if lstm_model_path or xgboost_model_path:
        predictor.load_models(lstm_model_path, xgboost_model_path)

    return predictor
