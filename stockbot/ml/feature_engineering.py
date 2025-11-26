"""
Feature Engineering for Stock Prediction

Extracts 50+ features from stock data:
- Technical indicators (RSI, MACD, Bollinger Bands, ATR, OBV, etc.)
- Fundamental metrics (P/E, P/B, debt ratio, margins, growth rates)
- Sentiment features (news sentiment, social media, analyst ratings)
- Market context (VIX, sector performance, correlation)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta
import yfinance as yf


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """Calculate MACD indicator"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> tuple:
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range (volatility)"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate On-Balance Volume"""
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> tuple:
    """Calculate Stochastic Oscillator"""
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=3).mean()
    return k_percent, d_percent


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average Directional Index (trend strength)"""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    return adx


def extract_technical_features(ticker: str, lookback_days: int = 252) -> Dict[str, float]:
    """
    Extract technical indicators from price data
    Returns 25+ technical features
    """
    try:
        # Fetch historical data
        stock = yf.Ticker(ticker)
        hist = stock.history(period=f"{lookback_days}d")

        if hist.empty or len(hist) < 50:
            return {}

        close = hist['Close']
        high = hist['High']
        low = hist['Low']
        volume = hist['Volume']

        features = {}

        # Price momentum features
        features['price_current'] = float(close.iloc[-1])
        features['price_change_1d'] = float((close.iloc[-1] / close.iloc[-2] - 1) * 100) if len(close) > 1 else 0
        features['price_change_5d'] = float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(close) > 5 else 0
        features['price_change_20d'] = float((close.iloc[-1] / close.iloc[-21] - 1) * 100) if len(close) > 20 else 0
        features['price_change_60d'] = float((close.iloc[-1] / close.iloc[-61] - 1) * 100) if len(close) > 60 else 0

        # Moving averages
        features['sma_10'] = float(close.rolling(window=10).mean().iloc[-1])
        features['sma_20'] = float(close.rolling(window=20).mean().iloc[-1])
        features['sma_50'] = float(close.rolling(window=50).mean().iloc[-1])
        features['sma_200'] = float(close.rolling(window=200).mean().iloc[-1]) if len(close) >= 200 else 0

        # SMA relationships (critical for trend detection)
        features['price_vs_sma20'] = float((close.iloc[-1] / features['sma_20'] - 1) * 100)
        features['price_vs_sma50'] = float((close.iloc[-1] / features['sma_50'] - 1) * 100)
        features['price_vs_sma200'] = float((close.iloc[-1] / features['sma_200'] - 1) * 100) if features['sma_200'] > 0 else 0
        features['sma20_vs_sma50'] = float((features['sma_20'] / features['sma_50'] - 1) * 100)

        # RSI
        rsi = calculate_rsi(close)
        features['rsi_14'] = float(rsi.iloc[-1])
        features['rsi_oversold'] = 1.0 if features['rsi_14'] < 30 else 0.0
        features['rsi_overbought'] = 1.0 if features['rsi_14'] > 70 else 0.0

        # MACD
        macd_line, signal_line, histogram = calculate_macd(close)
        features['macd'] = float(macd_line.iloc[-1])
        features['macd_signal'] = float(signal_line.iloc[-1])
        features['macd_histogram'] = float(histogram.iloc[-1])
        features['macd_bullish_crossover'] = 1.0 if features['macd'] > features['macd_signal'] else 0.0

        # Bollinger Bands
        bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(close)
        features['bb_upper'] = float(bb_upper.iloc[-1])
        features['bb_lower'] = float(bb_lower.iloc[-1])
        features['bb_width'] = float((bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_mid.iloc[-1] * 100)
        features['price_vs_bb'] = float((close.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]) * 100)

        # ATR (volatility)
        atr = calculate_atr(high, low, close)
        features['atr'] = float(atr.iloc[-1])
        features['atr_pct'] = float(atr.iloc[-1] / close.iloc[-1] * 100)

        # Volume indicators
        features['volume_current'] = float(volume.iloc[-1])
        features['volume_sma_20'] = float(volume.rolling(window=20).mean().iloc[-1])
        features['volume_ratio'] = float(volume.iloc[-1] / features['volume_sma_20'])

        # OBV
        obv = calculate_obv(close, volume)
        features['obv'] = float(obv.iloc[-1])
        features['obv_trend'] = float((obv.iloc[-1] - obv.iloc[-21]) / abs(obv.iloc[-21]) * 100) if len(obv) > 20 else 0

        # Stochastic Oscillator
        stoch_k, stoch_d = calculate_stochastic(high, low, close)
        features['stochastic_k'] = float(stoch_k.iloc[-1])
        features['stochastic_d'] = float(stoch_d.iloc[-1])

        # ADX (trend strength)
        adx = calculate_adx(high, low, close)
        features['adx'] = float(adx.iloc[-1])
        features['strong_trend'] = 1.0 if features['adx'] > 25 else 0.0

        # 52-week high/low
        features['52w_high'] = float(high.rolling(window=252).max().iloc[-1]) if len(high) >= 252 else float(high.max())
        features['52w_low'] = float(low.rolling(window=252).min().iloc[-1]) if len(low) >= 252 else float(low.min())
        features['distance_from_52w_high'] = float((features['52w_high'] - close.iloc[-1]) / features['52w_high'] * 100)
        features['distance_from_52w_low'] = float((close.iloc[-1] - features['52w_low']) / features['52w_low'] * 100)

        return features

    except Exception as e:
        print(f"Error extracting technical features for {ticker}: {e}")
        return {}


def extract_fundamental_features(ticker: str) -> Dict[str, float]:
    """
    Extract fundamental metrics
    Returns 15+ fundamental features
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        features = {}

        # Valuation metrics
        features['market_cap'] = float(info.get('marketCap', 0))
        features['pe_ratio'] = float(info.get('trailingPE', 0)) if info.get('trailingPE') else 0
        features['forward_pe'] = float(info.get('forwardPE', 0)) if info.get('forwardPE') else 0
        features['peg_ratio'] = float(info.get('pegRatio', 0)) if info.get('pegRatio') else 0
        features['pb_ratio'] = float(info.get('priceToBook', 0)) if info.get('priceToBook') else 0
        features['ps_ratio'] = float(info.get('priceToSalesTrailing12Months', 0)) if info.get('priceToSalesTrailing12Months') else 0

        # Profitability metrics
        features['profit_margin'] = float(info.get('profitMargins', 0) * 100) if info.get('profitMargins') else 0
        features['operating_margin'] = float(info.get('operatingMargins', 0) * 100) if info.get('operatingMargins') else 0
        features['roe'] = float(info.get('returnOnEquity', 0) * 100) if info.get('returnOnEquity') else 0
        features['roa'] = float(info.get('returnOnAssets', 0) * 100) if info.get('returnOnAssets') else 0

        # Growth metrics
        features['revenue_growth'] = float(info.get('revenueGrowth', 0) * 100) if info.get('revenueGrowth') else 0
        features['earnings_growth'] = float(info.get('earningsGrowth', 0) * 100) if info.get('earningsGrowth') else 0
        features['revenue_per_share'] = float(info.get('revenuePerShare', 0)) if info.get('revenuePerShare') else 0

        # Financial health
        features['debt_to_equity'] = float(info.get('debtToEquity', 0)) if info.get('debtToEquity') else 0
        features['current_ratio'] = float(info.get('currentRatio', 0)) if info.get('currentRatio') else 0
        features['quick_ratio'] = float(info.get('quickRatio', 0)) if info.get('quickRatio') else 0

        # Dividend metrics
        features['dividend_yield'] = float(info.get('dividendYield', 0) * 100) if info.get('dividendYield') else 0
        features['payout_ratio'] = float(info.get('payoutRatio', 0) * 100) if info.get('payoutRatio') else 0

        # Analyst sentiment
        features['target_price'] = float(info.get('targetMeanPrice', 0)) if info.get('targetMeanPrice') else 0
        features['num_analysts'] = float(info.get('numberOfAnalystOpinions', 0)) if info.get('numberOfAnalystOpinions') else 0
        features['recommendation_score'] = float(info.get('recommendationMean', 0)) if info.get('recommendationMean') else 0

        return features

    except Exception as e:
        print(f"Error extracting fundamental features for {ticker}: {e}")
        return {}


def extract_sentiment_features(news_items: list, ticker: str) -> Dict[str, float]:
    """
    Extract sentiment features from news
    Returns 10+ sentiment features
    """
    try:
        features = {}

        if not news_items:
            return {
                'news_count': 0.0,
                'avg_sentiment': 0.0,
                'positive_ratio': 0.0,
                'negative_ratio': 0.0,
                'sentiment_trend': 0.0,
            }

        sentiments = [item.get('sentiment', 0) for item in news_items if item.get('sentiment') is not None]

        features['news_count'] = float(len(news_items))
        features['avg_sentiment'] = float(np.mean(sentiments)) if sentiments else 0.0
        features['sentiment_std'] = float(np.std(sentiments)) if sentiments else 0.0
        features['positive_ratio'] = float(sum(1 for s in sentiments if s > 0.1) / len(sentiments)) if sentiments else 0.0
        features['negative_ratio'] = float(sum(1 for s in sentiments if s < -0.1) / len(sentiments)) if sentiments else 0.0
        features['neutral_ratio'] = float(sum(1 for s in sentiments if abs(s) <= 0.1) / len(sentiments)) if sentiments else 0.0

        # Sentiment trend (recent vs. older news)
        if len(sentiments) >= 5:
            recent_sentiment = np.mean(sentiments[:len(sentiments)//2])
            older_sentiment = np.mean(sentiments[len(sentiments)//2:])
            features['sentiment_trend'] = float(recent_sentiment - older_sentiment)
        else:
            features['sentiment_trend'] = 0.0

        # Catalyst detection
        catalyst_keywords = ['contract', 'partnership', 'deal', 'acquisition', 'merger', 'approval', 'award', 'funding']
        catalyst_count = sum(1 for item in news_items if any(kw in item.get('title', '').lower() for kw in catalyst_keywords))
        features['catalyst_count'] = float(catalyst_count)
        features['has_major_catalyst'] = 1.0 if catalyst_count > 0 else 0.0

        return features

    except Exception as e:
        print(f"Error extracting sentiment features: {e}")
        return {}


def extract_market_context_features(ticker: str) -> Dict[str, float]:
    """
    Extract market context features (sector, VIX, correlation)
    Returns 5+ context features
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        features = {}

        # Fetch VIX (market volatility)
        try:
            vix = yf.Ticker("^VIX")
            vix_history = vix.history(period="5d")
            if not vix_history.empty:
                features['vix'] = float(vix_history['Close'].iloc[-1])
                features['vix_high'] = 1.0 if features['vix'] > 20 else 0.0
            else:
                features['vix'] = 15.0  # Default
                features['vix_high'] = 0.0
        except:
            features['vix'] = 15.0
            features['vix_high'] = 0.0

        # Sector classification
        sector = info.get('sector', '').lower()
        features['is_tech'] = 1.0 if 'technology' in sector else 0.0
        features['is_healthcare'] = 1.0 if 'healthcare' in sector else 0.0
        features['is_financial'] = 1.0 if 'financial' in sector else 0.0
        features['is_energy'] = 1.0 if 'energy' in sector else 0.0

        # Market cap category
        market_cap = info.get('marketCap', 0)
        features['is_large_cap'] = 1.0 if market_cap > 10e9 else 0.0
        features['is_mid_cap'] = 1.0 if 2e9 <= market_cap <= 10e9 else 0.0
        features['is_small_cap'] = 1.0 if market_cap < 2e9 else 0.0

        return features

    except Exception as e:
        print(f"Error extracting market context features: {e}")
        return {}


def extract_all_features(ticker: str, news_items: list = None, lookback_days: int = 252) -> Dict[str, float]:
    """
    Extract ALL features for a stock
    Returns 50+ features combined from all sources
    """
    all_features = {}

    # Technical features (25+)
    technical = extract_technical_features(ticker, lookback_days)
    all_features.update(technical)

    # Fundamental features (15+)
    fundamental = extract_fundamental_features(ticker)
    all_features.update(fundamental)

    # Sentiment features (10+)
    sentiment = extract_sentiment_features(news_items or [], ticker)
    all_features.update(sentiment)

    # Market context features (5+)
    context = extract_market_context_features(ticker)
    all_features.update(context)

    return all_features


def features_to_dataframe(features: Dict[str, float]) -> pd.DataFrame:
    """Convert feature dict to DataFrame for model input"""
    return pd.DataFrame([features])


# Feature importance mapping for explanations
FEATURE_CATEGORIES = {
    'technical': [
        'rsi_14', 'macd_histogram', 'price_vs_sma20', 'price_vs_sma50', 'bb_width',
        'volume_ratio', 'obv_trend', 'stochastic_k', 'adx', 'atr_pct'
    ],
    'fundamental': [
        'pe_ratio', 'pb_ratio', 'profit_margin', 'roe', 'revenue_growth',
        'debt_to_equity', 'current_ratio', 'recommendation_score'
    ],
    'sentiment': [
        'avg_sentiment', 'positive_ratio', 'catalyst_count', 'sentiment_trend'
    ],
    'market_context': [
        'vix', 'is_tech', 'is_small_cap'
    ]
}
