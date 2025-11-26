"""
SHAP Integration for Model Explainability

Extracts feature importance using SHAP (SHapley Additive exPlanations)
to understand WHY the model made a specific prediction.

SHAP values show the contribution of each feature to the final prediction.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not installed. Install with: pip install shap")


class SHAPExplainer:
    """
    SHAP-based explainability for stock predictions

    Generates feature importance and contributions for individual predictions
    """

    def __init__(self, model, model_type: str = 'xgboost'):
        """
        Initialize SHAP explainer

        Args:
            model: Trained model (XGBoost, LSTM, etc.)
            model_type: Type of model ('xgboost', 'pytorch', 'sklearn')
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required. Install with: pip install shap")

        self.model = model
        self.model_type = model_type
        self.explainer = None
        self.background_data = None

    def initialize_explainer(
        self,
        background_data: pd.DataFrame,
        max_samples: int = 100
    ):
        """
        Initialize SHAP explainer with background data

        Args:
            background_data: Representative sample of training data
            max_samples: Max samples for background (more = slower but more accurate)
        """
        # Sample background data if too large
        if len(background_data) > max_samples:
            background_data = background_data.sample(n=max_samples, random_state=42)

        self.background_data = background_data

        # Create appropriate explainer based on model type
        if self.model_type == 'xgboost':
            self.explainer = shap.TreeExplainer(self.model.model)
        elif self.model_type == 'pytorch':
            # For neural networks, use DeepExplainer or KernelExplainer
            self.explainer = shap.KernelExplainer(
                self.model.predict,
                background_data.values
            )
        else:
            # Generic explainer for any model
            self.explainer = shap.Explainer(
                self.model.predict,
                background_data
            )

        print(f"SHAP explainer initialized with {len(background_data)} background samples")

    def explain_prediction(
        self,
        features: pd.DataFrame,
        top_n: int = 10
    ) -> Dict[str, float]:
        """
        Get SHAP values for a single prediction

        Args:
            features: Single row DataFrame with features
            top_n: Return top N features by absolute SHAP value

        Returns:
            Dict mapping feature_name -> SHAP_value
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer() first.")

        # Calculate SHAP values
        if self.model_type == 'xgboost':
            shap_values = self.explainer.shap_values(features)
        else:
            shap_values = self.explainer(features).values

        # Handle different SHAP output formats
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]  # Take first prediction if batch

        # Create dict of feature -> SHAP value
        feature_contributions = dict(zip(features.columns, shap_values))

        # Sort by absolute value and take top N
        top_features = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_n]

        return dict(top_features)

    def explain_with_categories(
        self,
        features: pd.DataFrame,
        feature_categories: Dict[str, List[str]]
    ) -> Dict[str, Dict]:
        """
        Explain prediction grouped by feature categories

        Args:
            features: Single row DataFrame
            feature_categories: Dict mapping category -> list of features
                Example: {'technical': ['rsi_14', 'macd'], 'fundamental': ['pe_ratio']}

        Returns:
            Dict mapping category -> {features: {feat: shap_val}, total_contribution: float}
        """
        # Get all SHAP values
        all_shap = self.explain_prediction(features, top_n=len(features.columns))

        # Group by category
        categorized = {}
        for category, feature_list in feature_categories.items():
            category_shap = {
                feat: all_shap.get(feat, 0.0)
                for feat in feature_list
                if feat in all_shap
            }

            total_contribution = sum(abs(v) for v in category_shap.values())

            categorized[category] = {
                'features': category_shap,
                'total_contribution': total_contribution,
                'direction': 'positive' if sum(category_shap.values()) > 0 else 'negative'
            }

        return categorized

    def get_top_factors(
        self,
        features: pd.DataFrame,
        feature_metadata: Dict[str, Dict],
        n: int = 5
    ) -> List[Dict]:
        """
        Get top N factors with rich metadata for explanations

        Args:
            features: Single row DataFrame
            feature_metadata: Dict mapping feature_name -> {category, display_name, etc.}
            n: Number of top factors to return

        Returns:
            List of dicts with factor details
        """
        shap_values = self.explain_prediction(features, top_n=n)

        factors = []
        for feature_name, shap_value in shap_values.items():
            metadata = feature_metadata.get(feature_name, {})

            factor = {
                'feature_name': feature_name,
                'feature_value': features[feature_name].iloc[0],
                'shap_value': shap_value,
                'importance_pct': 0.0,  # Will calculate below
                'direction': 'positive' if shap_value > 0 else 'negative',
                'category': metadata.get('category', 'unknown'),
                'display_name': metadata.get('display_name', feature_name)
            }
            factors.append(factor)

        # Calculate importance percentages
        total_importance = sum(abs(f['shap_value']) for f in factors)
        if total_importance > 0:
            for factor in factors:
                factor['importance_pct'] = abs(factor['shap_value']) / total_importance * 100

        return factors


class FastExplainer:
    """
    Fallback explainer when SHAP is too slow or unavailable

    Uses model's native feature importance (XGBoost) or gradient-based importance
    """

    def __init__(self, model, model_type: str = 'xgboost'):
        self.model = model
        self.model_type = model_type

    def explain_prediction(
        self,
        features: pd.DataFrame,
        top_n: int = 10
    ) -> Dict[str, float]:
        """
        Fast approximate explanation using feature importance

        Args:
            features: Single row DataFrame
            top_n: Top N features

        Returns:
            Dict of feature -> pseudo-SHAP value
        """
        if self.model_type == 'xgboost' and hasattr(self.model, 'feature_importances_'):
            # Use XGBoost's native feature importance
            importance = self.model.feature_importances_
            feature_names = features.columns

            # Pseudo-SHAP: importance * feature_value (normalized)
            contributions = {}
            for i, feat in enumerate(feature_names):
                feat_value = features[feat].iloc[0]
                # Normalize by feature magnitude
                normalized_value = feat_value / (abs(feat_value) + 1)
                contributions[feat] = importance[i] * normalized_value

            # Sort and take top N
            top_features = sorted(
                contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:top_n]

            return dict(top_features)
        else:
            # Generic fallback: equal importance, scaled by feature value
            contributions = {}
            for feat in features.columns:
                feat_value = features[feat].iloc[0]
                contributions[feat] = feat_value / 10.0  # Simple scaling

            top_features = sorted(
                contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:top_n]

            return dict(top_features)


def create_explainer(
    model,
    model_type: str = 'xgboost',
    background_data: Optional[pd.DataFrame] = None,
    use_shap: bool = True
) -> SHAPExplainer | FastExplainer:
    """
    Factory function to create explainer

    Args:
        model: Trained model
        model_type: Type of model
        background_data: Background data for SHAP
        use_shap: If False, use fast explainer

    Returns:
        Explainer instance
    """
    if use_shap and SHAP_AVAILABLE:
        explainer = SHAPExplainer(model, model_type)
        if background_data is not None:
            explainer.initialize_explainer(background_data)
        return explainer
    else:
        print("Using fast explainer (SHAP not available or disabled)")
        return FastExplainer(model, model_type)


# Feature metadata for explanation generation
FEATURE_METADATA = {
    # Technical indicators
    'rsi_14': {
        'category': 'technical',
        'display_name': 'RSI (Relative Strength Index)',
        'description': 'Momentum oscillator measuring overbought/oversold conditions'
    },
    'macd_histogram': {
        'category': 'technical',
        'display_name': 'MACD Momentum',
        'description': 'Difference between MACD line and signal line'
    },
    'price_vs_sma50': {
        'category': 'technical',
        'display_name': 'Position vs 50-day MA',
        'description': 'How far price is from 50-day moving average'
    },
    'volume_ratio': {
        'category': 'technical',
        'display_name': 'Trading Volume',
        'description': 'Current volume vs 20-day average'
    },
    'bb_width': {
        'category': 'technical',
        'display_name': 'Bollinger Band Width',
        'description': 'Volatility measure - wider bands = higher volatility'
    },
    'distance_from_52w_high': {
        'category': 'technical',
        'display_name': 'Distance from 52-Week High',
        'description': 'How far stock has fallen from yearly peak'
    },

    # Fundamental metrics
    'pe_ratio': {
        'category': 'fundamental',
        'display_name': 'P/E Ratio (Valuation)',
        'description': 'Price-to-Earnings ratio - valuation metric'
    },
    'revenue_growth': {
        'category': 'fundamental',
        'display_name': 'Revenue Growth',
        'description': 'Year-over-year revenue growth percentage'
    },
    'profit_margin': {
        'category': 'fundamental',
        'display_name': 'Profit Margin',
        'description': 'Net profit as percentage of revenue'
    },
    'debt_to_equity': {
        'category': 'fundamental',
        'display_name': 'Debt-to-Equity Ratio',
        'description': 'Financial leverage - total debt / shareholder equity'
    },
    'roe': {
        'category': 'fundamental',
        'display_name': 'Return on Equity',
        'description': 'Profitability measure - net income / equity'
    },

    # Sentiment
    'avg_sentiment': {
        'category': 'sentiment',
        'display_name': 'News Sentiment',
        'description': 'Average sentiment of recent news articles'
    },
    'catalyst_count': {
        'category': 'sentiment',
        'display_name': 'Business Catalysts',
        'description': 'Number of major catalysts (contracts, deals, approvals)'
    },
    'positive_ratio': {
        'category': 'sentiment',
        'display_name': 'Positive News Ratio',
        'description': 'Percentage of news articles with positive sentiment'
    },

    # Market context
    'vix': {
        'category': 'market_context',
        'display_name': 'Market Volatility (VIX)',
        'description': 'CBOE Volatility Index - market fear gauge'
    },
    'is_small_cap': {
        'category': 'market_context',
        'display_name': 'Small Cap Stock',
        'description': 'Market cap < $2B (higher growth potential, higher risk)'
    }
}
