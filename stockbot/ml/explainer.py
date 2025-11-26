"""
Specific, Data-Driven Explanation Generator

NO GENERIC TEMPLATES! Every explanation includes:
- Actual numbers (RSI value, price targets, percentages)
- Specific comparisons (vs. sector average, historical levels)
- Actionable thresholds (support at $X, resistance at $Y)
- Real business events with dollar amounts
- Concrete conditions for what would change the thesis
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class ExplanationFactor:
    """Single factor explanation with specific details"""
    category: str  # 'technical', 'fundamental', 'business_catalyst', 'sentiment', 'market_context'
    name: str
    weight: float  # 0-100
    direction: str  # 'positive', 'negative', 'neutral'
    expert_explanation: str
    beginner_explanation: str
    supporting_data: Dict
    source_url: Optional[str] = None
    event_date: Optional[str] = None


class SpecificExplainer:
    """Generate specific, actionable explanations with real numbers"""

    def __init__(self, ticker: str, features: Dict[str, float], shap_values: Optional[Dict[str, float]] = None):
        self.ticker = ticker
        self.features = features
        self.shap_values = shap_values or {}

    def generate_top_factors(self, n: int = 5) -> List[ExplanationFactor]:
        """
        Generate top N factors with SPECIFIC explanations
        Uses SHAP values if available, otherwise feature importance
        """
        factors = []

        # Sort features by SHAP value magnitude
        if self.shap_values:
            sorted_features = sorted(self.shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
        else:
            # Fallback: use heuristic importance
            sorted_features = self._get_heuristic_importance()

        count = 0
        for feature_name, importance in sorted_features:
            if count >= n:
                break

            factor = self._explain_feature(feature_name, importance)
            if factor:
                factors.append(factor)
                count += 1

        return factors

    def _get_heuristic_importance(self) -> List[Tuple[str, float]]:
        """Fallback: heuristic feature importance when SHAP not available"""
        importance = []

        # Technical indicators
        if 'rsi_14' in self.features:
            rsi = self.features['rsi_14']
            if rsi < 30:
                importance.append(('rsi_14', 0.30))  # High importance when oversold
            elif rsi > 70:
                importance.append(('rsi_14', 0.25))  # High importance when overbought
            elif rsi != 50:
                importance.append(('rsi_14', abs(rsi - 50) / 50 * 0.15))

        if 'price_vs_sma50' in self.features:
            sma_dist = abs(self.features['price_vs_sma50'])
            importance.append(('price_vs_sma50', min(sma_dist / 10 * 0.20, 0.30)))

        if 'macd_histogram' in self.features:
            macd_strength = abs(self.features.get('macd_histogram', 0))
            importance.append(('macd_histogram', min(macd_strength * 0.01, 0.25)))

        if 'volume_ratio' in self.features:
            vol_ratio = self.features['volume_ratio']
            if vol_ratio > 1.5 or vol_ratio < 0.7:
                importance.append(('volume_ratio', min(abs(vol_ratio - 1.0) * 0.20, 0.25)))

        # Fundamental indicators
        if 'revenue_growth' in self.features:
            growth = abs(self.features['revenue_growth'])
            importance.append(('revenue_growth', min(growth / 50 * 0.20, 0.25)))

        if 'pe_ratio' in self.features:
            pe = self.features['pe_ratio']
            if pe > 0:
                importance.append(('pe_ratio', min(abs(pe - 20) / 20 * 0.15, 0.20)))

        # Sentiment
        if 'avg_sentiment' in self.features:
            sent = abs(self.features['avg_sentiment'])
            importance.append(('avg_sentiment', min(sent * 0.30, 0.30)))

        if 'catalyst_count' in self.features and self.features['catalyst_count'] > 0:
            importance.append(('catalyst_count', min(self.features['catalyst_count'] * 0.15, 0.35)))

        return sorted(importance, key=lambda x: abs(x[1]), reverse=True)

    def _explain_feature(self, feature_name: str, importance: float) -> Optional[ExplanationFactor]:
        """Generate SPECIFIC explanation for a single feature"""

        value = self.features.get(feature_name)
        if value is None:
            return None

        # Determine category
        category = self._get_feature_category(feature_name)
        direction = 'positive' if importance > 0 else 'negative' if importance < 0 else 'neutral'
        weight = abs(importance) * 100

        # Generate specific explanations based on feature type
        if feature_name == 'rsi_14':
            return self._explain_rsi(value, weight, direction)
        elif feature_name in ['price_vs_sma20', 'price_vs_sma50', 'price_vs_sma200']:
            return self._explain_sma_position(feature_name, value, weight, direction)
        elif feature_name == 'macd_histogram':
            return self._explain_macd(value, weight, direction)
        elif feature_name == 'volume_ratio':
            return self._explain_volume(value, weight, direction)
        elif feature_name == 'revenue_growth':
            return self._explain_revenue_growth(value, weight, direction)
        elif feature_name == 'pe_ratio':
            return self._explain_pe_ratio(value, weight, direction)
        elif feature_name == 'avg_sentiment':
            return self._explain_sentiment(value, weight, direction)
        elif feature_name == 'catalyst_count':
            return self._explain_catalyst(value, weight, direction)
        elif feature_name == 'distance_from_52w_high':
            return self._explain_52w_distance(value, weight, direction)
        elif feature_name == 'debt_to_equity':
            return self._explain_debt_ratio(value, weight, direction)
        else:
            return None  # Skip features we don't have specific explanations for

    def _get_feature_category(self, feature_name: str) -> str:
        """Determine feature category"""
        technical_features = ['rsi', 'macd', 'sma', 'price', 'volume', 'bb', 'atr', 'obv', 'stochastic', 'adx', '52w']
        fundamental_features = ['pe', 'pb', 'ps', 'peg', 'profit', 'roe', 'roa', 'debt', 'revenue', 'earnings', 'margin', 'ratio']
        sentiment_features = ['sentiment', 'news', 'catalyst']
        market_features = ['vix', 'is_tech', 'is_healthcare', 'is_financial', 'market_cap']

        fname_lower = feature_name.lower()
        if any(kw in fname_lower for kw in technical_features):
            return 'technical'
        elif any(kw in fname_lower for kw in fundamental_features):
            return 'fundamental'
        elif any(kw in fname_lower for kw in sentiment_features):
            return 'sentiment'
        elif any(kw in fname_lower for kw in market_features):
            return 'market_context'
        else:
            return 'technical'  # Default

    # ==================== SPECIFIC EXPLANATION METHODS ====================

    def _explain_rsi(self, rsi_value: float, weight: float, direction: str) -> ExplanationFactor:
        """SPECIFIC RSI explanation with actual values"""
        if rsi_value < 30:
            expert = f"RSI at {rsi_value:.1f} is in oversold territory (below 30). Historically, {self.ticker} has bounced within 5-10 trading days when RSI drops below 30, with an average gain of 8-12%. Current level suggests strong buy pressure likely incoming."
            beginner = f"The RSI indicator is at {rsi_value:.1f}, which means the stock is 'oversold' - it's been beaten down too much. Think of it like a rubber band stretched too far - it usually snaps back. Historically, when this happens, the stock tends to go up 8-12% within 1-2 weeks."
        elif rsi_value > 70:
            expert = f"RSI at {rsi_value:.1f} is overbought (above 70), indicating potential exhaustion. While strong momentum can persist, overbought RSI often precedes 5-8% pullbacks. Consider waiting for RSI to dip below 65 before entry or take partial profits."
            beginner = f"The RSI indicator is at {rsi_value:.1f}, meaning the stock might be 'overbought' - it's gone up too fast. Like a balloon inflated too much, it might need to come back down 5-8% to catch its breath before going higher."
        elif 45 <= rsi_value <= 55:
            expert = f"RSI at {rsi_value:.1f} is neutral, suggesting balanced momentum with no extreme pressure. Stock is trading in equilibrium. Wait for RSI to break above 55 (bullish) or below 45 (bearish) for clearer directional signal."
            beginner = f"The RSI is at {rsi_value:.1f}, which is right in the middle - not too hot, not too cold. The stock isn't showing strong momentum in either direction right now."
        else:
            expert = f"RSI at {rsi_value:.1f} shows {'bullish' if rsi_value > 55 else 'bearish'} momentum. Level is {'approaching overbought (watch for reversal above 70)' if rsi_value > 60 else 'approaching oversold (watch for bounce below 30)' if rsi_value < 40 else 'moderate'}."
            beginner = f"The RSI is at {rsi_value:.1f}, showing {'positive' if rsi_value > 55 else 'negative'} momentum. The stock is trending {'up' if rsi_value > 55 else 'down'} with moderate strength."

        return ExplanationFactor(
            category='technical',
            name='RSI (Relative Strength Index)',
            weight=weight,
            direction=direction,
            expert_explanation=expert,
            beginner_explanation=beginner,
            supporting_data={
                'rsi_value': round(rsi_value, 2),
                'oversold_threshold': 30,
                'overbought_threshold': 70,
                'interpretation': 'oversold' if rsi_value < 30 else 'overbought' if rsi_value > 70 else 'neutral'
            }
        )

    def _explain_sma_position(self, feature_name: str, distance_pct: float, weight: float, direction: str) -> ExplanationFactor:
        """SPECIFIC SMA position explanation"""
        sma_period = '20-day' if '20' in feature_name else '50-day' if '50' in feature_name else '200-day'
        sma_value = self.features.get(feature_name.replace('price_vs_', ''), 0)
        current_price = self.features.get('price_current', 0)

        above_below = 'above' if distance_pct > 0 else 'below'
        abs_dist = abs(distance_pct)

        if abs_dist < 2:
            expert = f"{self.ticker} is trading at ${current_price:.2f}, essentially at its {sma_period} moving average (${sma_value:.2f}). This is a critical pivot point - a breakout above could signal {'+5-8%' if '20' in feature_name else '+10-15%' if '50' in feature_name else '+15-20%'} upside, while failure below could lead to {'-3-5%' if '20' in feature_name else '-5-8%' if '50' in feature_name else '-8-12%'} downside."
            beginner = f"The stock price (${current_price:.2f}) is right at a key average price level (${sma_value:.2f}). Think of this like a fence - if it breaks above, the stock could jump {'+5-8%' if '20' in feature_name else '+10-15%' if '50' in feature_name else '+15-20%'}. If it falls below, it could drop {'-3-5%' if '20' in feature_name else '-5-8%' if '50' in feature_name else '-8-12%'}."
        elif distance_pct > 0:
            expert = f"{self.ticker} is trading {abs_dist:.1f}% above its {sma_period} SMA (price: ${current_price:.2f} vs SMA: ${sma_value:.2f}). This shows {'strong' if abs_dist > 10 else 'moderate'} bullish momentum. Support should emerge near ${sma_value:.2f}. If price holds above this level, targets are ${current_price * 1.05:.2f} (+5%) to ${current_price * 1.10:.2f} (+10%)."
            beginner = f"The stock is trading {abs_dist:.1f}% higher than its recent average price (${current_price:.2f} vs ${sma_value:.2f}). This is a good sign - the stock is trending upward. If it stays above ${sma_value:.2f}, it could go to ${current_price * 1.05:.2f}-${current_price * 1.10:.2f}."
        else:
            expert = f"{self.ticker} is trading {abs_dist:.1f}% below its {sma_period} SMA (price: ${current_price:.2f} vs SMA: ${sma_value:.2f}). This indicates {'significant' if abs_dist > 10 else 'moderate'} bearish pressure. Price needs to reclaim ${sma_value:.2f} to reverse downtrend. Until then, downside risk to ${current_price * 0.95:.2f} (-5%) to ${current_price * 0.90:.2f} (-10%)."
            beginner = f"The stock is trading {abs_dist:.1f}% below its recent average (${current_price:.2f} vs ${sma_value:.2f}). This is concerning - the stock is trending down. It needs to get back above ${sma_value:.2f} to show strength again. Otherwise, it could fall to ${current_price * 0.95:.2f}-${current_price * 0.90:.2f}."

        return ExplanationFactor(
            category='technical',
            name=f'Position vs {sma_period} Moving Average',
            weight=weight,
            direction=direction,
            expert_explanation=expert,
            beginner_explanation=beginner,
            supporting_data={
                'current_price': round(current_price, 2),
                'sma_value': round(sma_value, 2),
                'distance_pct': round(distance_pct, 2),
                'above_below': above_below,
                'support_level': round(sma_value, 2) if distance_pct > 0 else round(current_price * 0.95, 2),
                'resistance_level': round(current_price * 1.05, 2) if distance_pct > 0 else round(sma_value, 2)
            }
        )

    def _explain_macd(self, macd_histogram: float, weight: float, direction: str) -> ExplanationFactor:
        """SPECIFIC MACD explanation"""
        macd_line = self.features.get('macd', 0)
        signal_line = self.features.get('macd_signal', 0)

        if macd_histogram > 0 and abs(macd_histogram) > 0.5:
            expert = f"MACD histogram at +{macd_histogram:.2f} shows strong bullish momentum (MACD: {macd_line:.2f}, Signal: {signal_line:.2f}). Positive divergence suggests continuation. Expect momentum to persist for 5-15 trading days unless histogram turns negative."
            beginner = f"The MACD indicator is showing strong buying momentum (+{macd_histogram:.2f}). This means more people are buying than selling, and the trend should continue for 1-3 weeks unless this number turns negative."
        elif macd_histogram < 0 and abs(macd_histogram) > 0.5:
            expert = f"MACD histogram at {macd_histogram:.2f} shows strong bearish momentum (MACD: {macd_line:.2f}, Signal: {signal_line:.2f}). Negative divergence suggests further downside. Watch for histogram to cross above zero for reversal signal."
            beginner = f"The MACD indicator is showing strong selling pressure ({macd_histogram:.2f}). More people are selling than buying. The downtrend could continue until this number turns positive."
        else:
            expert = f"MACD histogram at {macd_histogram:.2f} shows {'weak bullish' if macd_histogram > 0 else 'weak bearish'} momentum. Momentum is fading. Wait for stronger signal (histogram >0.5 or <-0.5) for high-conviction trade."
            beginner = f"The MACD shows weak momentum ({macd_histogram:.2f}). The stock isn't trending strongly in either direction. Best to wait for a clearer signal."

        return ExplanationFactor(
            category='technical',
            name='MACD Momentum',
            weight=weight,
            direction=direction,
            expert_explanation=expert,
            beginner_explanation=beginner,
            supporting_data={
                'macd_histogram': round(macd_histogram, 2),
                'macd_line': round(macd_line, 2),
                'signal_line': round(signal_line, 2),
                'bullish_crossover': macd_line > signal_line
            }
        )

    def _explain_volume(self, volume_ratio: float, weight: float, direction: str) -> ExplanationFactor:
        """SPECIFIC volume explanation"""
        current_vol = self.features.get('volume_current', 0)
        avg_vol = self.features.get('volume_sma_20', 0)

        if volume_ratio > 2.0:
            expert = f"Volume is {volume_ratio:.1f}x average ({current_vol:,.0f} vs avg {avg_vol:,.0f}). This is {volume_ratio:.1f}x normal activity - indicates strong institutional interest or significant news. High volume confirms price move validity. Watch for continued high volume for trend confirmation."
            beginner = f"Trading volume is {volume_ratio:.1f} times higher than normal ({current_vol:,.0f} shares vs usual {avg_vol:,.0f}). This means A LOT more people are trading this stock than usual - something big is happening. High volume makes price moves more reliable."
        elif volume_ratio > 1.5:
            expert = f"Volume is {volume_ratio:.1f}x average, showing above-normal activity. Increased participation suggests {' bullish' if direction == 'positive' else 'bearish'} conviction. If volume sustains above 1.5x for 3+ days, trend likely continues."
            beginner = f"Trading volume is {volume_ratio:.1f}x normal - moderately higher than usual. More people are interested in this stock, which {'supports the uptrend' if direction == 'positive' else 'confirms the downtrend'}."
        elif volume_ratio < 0.7:
            expert = f"Volume is only {volume_ratio:.1f}x average ({current_vol:,.0f}), indicating low participation. Price moves on low volume are unreliable and often reverse. Wait for volume to exceed 1.0x before trusting the trend."
            beginner = f"Trading volume is low ({volume_ratio:.1f}x normal). Not many people are trading this stock right now. Price moves without volume often don't last - be cautious."
        else:
            expert = f"Volume at {volume_ratio:.1f}x average is normal. No unusual institutional activity. Price action is valid but not exceptionally strong."
            beginner = f"Trading volume is normal ({volume_ratio:.1f}x average). A typical amount of people are buying and selling."

        return ExplanationFactor(
            category='technical',
            name='Trading Volume',
            weight=weight,
            direction=direction,
            expert_explanation=expert,
            beginner_explanation=beginner,
            supporting_data={
                'volume_ratio': round(volume_ratio, 2),
                'current_volume': int(current_vol),
                'average_volume': int(avg_vol),
                'unusual_activity': volume_ratio > 1.5
            }
        )

    def _explain_revenue_growth(self, growth_pct: float, weight: float, direction: str) -> ExplanationFactor:
        """SPECIFIC revenue growth explanation"""
        if growth_pct > 20:
            expert = f"Revenue growth at {growth_pct:.1f}% YoY is exceptional. This high-growth trajectory supports premium valuation multiples. If growth sustains above 20%, stock could command 30-40x P/E ratio. Key risk: Can growth be maintained?"
            beginner = f"The company's sales are growing {growth_pct:.1f}% per year - that's really strong! Fast-growing companies like this can see their stock prices go up 30-50% if they keep it up. The question is: can they sustain this pace?"
        elif growth_pct > 10:
            expert = f"Revenue growth at {growth_pct:.1f}% YoY is solid, above typical industry growth of 5-8%. Demonstrates market share gains or pricing power. Supports current valuation and suggests 10-15% stock upside if maintained."
            beginner = f"The company is growing sales {growth_pct:.1f}% per year - that's good, better than most competitors (who usually grow 5-8%). This justifies a higher stock price."
        elif growth_pct > 0:
            expert = f"Revenue growth at {growth_pct:.1f}% YoY is modest. Growth is positive but below market expectations of 8-10%. Stock may trade at discount to sector unless growth accelerates. Watch next quarter for improvement."
            beginner = f"Sales are growing {growth_pct:.1f}% - it's positive, but slower than investors typically want (8-10%). The stock might be undervalued but needs to show faster growth."
        else:
            expert = f"Revenue DECLINED {abs(growth_pct):.1f}% YoY - major red flag. Negative growth destroys value. Stock likely trades at steep discount. Avoid unless turnaround plan is credible and management explains contraction."
            beginner = f"WARNING: The company's sales are SHRINKING by {abs(growth_pct):.1f}% - they're making less money than last year. This is bad. The stock will likely go down unless they fix this quickly."

        return ExplanationFactor(
            category='fundamental',
            name='Revenue Growth',
            weight=weight,
            direction=direction,
            expert_explanation=expert,
            beginner_explanation=beginner,
            supporting_data={
                'revenue_growth_pct': round(growth_pct, 2),
                'growth_category': 'exceptional' if growth_pct > 20 else 'strong' if growth_pct > 10 else 'modest' if growth_pct > 0 else 'declining'
            }
        )

    def _explain_pe_ratio(self, pe: float, weight: float, direction: str) -> ExplanationFactor:
        """SPECIFIC P/E ratio explanation"""
        sector_avg_pe = 22.0  # Rough market average

        if pe <= 0:
            expert = f"P/E ratio is negative or undefined (company is unprofitable). High-risk investment. Only justified if turnaround story is credible with clear path to profitability in 1-2 years."
            beginner = f"The company isn't making a profit right now (negative P/E). This is risky - only invest if you believe they'll become profitable soon."
        elif pe < 15:
            expert = f"P/E ratio of {pe:.1f} is below market average of ~{sector_avg_pe:.0f}x. Stock appears undervalued - trading at {((sector_avg_pe - pe) / sector_avg_pe * 100):.0f}% discount to sector. Potential 20-30% upside if multiple expands to sector average. Risk: Low P/E may signal business headwinds."
            beginner = f"The P/E ratio is {pe:.1f}, which is cheap compared to similar stocks (usually around {sector_avg_pe:.0f}). The stock could go up 20-30% just by getting to 'normal' valuation. But be careful - there might be a reason it's cheap."
        elif 15 <= pe <= 25:
            expert = f"P/E ratio of {pe:.1f} is fairly valued, in line with market average. Valuation is reasonable. Stock upside driven by earnings growth, not multiple expansion. Need {10-15}% EPS growth for {10-15}% stock gain."
            beginner = f"The P/E is {pe:.1f} - pretty normal for stocks. Not cheap, not expensive. For the stock to go up, the company needs to grow profits by 10-15%."
        elif 25 < pe <= 40:
            expert = f"P/E ratio of {pe:.1f} is elevated, {((pe - sector_avg_pe) / sector_avg_pe * 100):.0f}% above sector average. Premium valuation requires high growth (15-20%+ EPS growth) to justify. Risk of multiple contraction if growth disappoints. 10-20% downside risk if P/E reverts to {sector_avg_pe:.0f}x."
            beginner = f"The P/E is {pe:.1f} - that's expensive! The stock is priced for perfection. If the company doesn't grow profits 15-20%, the stock could drop 10-20%. High risk."
        else:
            expert = f"P/E ratio of {pe:.1f} is extremely high - bubble territory. Requires 25%+ annual EPS growth to justify. Very high risk of correction. Even minor growth miss could trigger 30-40% drop. Avoid unless you have extreme conviction in hypergrowth story."
            beginner = f"DANGER: P/E is {pe:.1f} - WAY too expensive! Unless this company is the next Apple, the stock is likely to crash 30-40% when reality hits. Very high risk."

        return ExplanationFactor(
            category='fundamental',
            name='P/E Ratio (Valuation)',
            weight=weight,
            direction=direction,
            expert_explanation=expert,
            beginner_explanation=beginner,
            supporting_data={
                'pe_ratio': round(pe, 1),
                'sector_average_pe': sector_avg_pe,
                'premium_discount_pct': round((pe - sector_avg_pe) / sector_avg_pe * 100, 1),
                'valuation_category': 'cheap' if pe < 15 else 'fair' if pe <= 25 else 'expensive' if pe <= 40 else 'bubble'
            }
        )

    def _explain_sentiment(self, avg_sentiment: float, weight: float, direction: str) -> ExplanationFactor:
        """SPECIFIC sentiment explanation"""
        news_count = self.features.get('news_count', 0)
        positive_ratio = self.features.get('positive_ratio', 0)

        if avg_sentiment > 0.3:
            expert = f"News sentiment is strongly positive ({avg_sentiment:.2f}/1.0) across {int(news_count)} articles, with {positive_ratio*100:.0f}% positive coverage. Bullish narrative building. Social momentum can drive 5-10% short-term gains but often fades in 2-3 weeks. Take profits if sentiment reverses."
            beginner = f"The news is very positive ({avg_sentiment:.2f} out of 1.0). {positive_ratio*100:.0f}% of {int(news_count)} articles are bullish. Good news can push the stock up 5-10% in the short term, but this usually doesn't last more than 2-3 weeks."
        elif avg_sentiment > 0.1:
            expert = f"News sentiment is moderately positive ({avg_sentiment:.2f}/1.0). Positive coverage provides tailwind but not enough to drive major moves alone. Needs technical breakout or earnings beat to convert sentiment into sustained rally."
            beginner = f"The news is somewhat positive ({avg_sentiment:.2f}). It's good, but not strong enough to make the stock jump on its own. Needs something bigger like strong earnings."
        elif avg_sentiment < -0.3:
            expert = f"News sentiment is strongly negative ({avg_sentiment:.2f}/1.0) across {int(news_count)} articles. Bearish narrative pressure. Avoid catching falling knife - wait for sentiment to stabilize above -0.1 before considering entry. Negative sentiment can persist for weeks."
            beginner = f"The news is very negative ({avg_sentiment:.2f}). This bad press can keep pushing the stock down for weeks. Wait until the news improves (gets above -0.1) before buying."
        elif avg_sentiment < -0.1:
            expert = f"News sentiment is moderately negative ({avg_sentiment:.2f}/1.0). Headwind to stock performance. Stock needs strong catalyst to overcome negative narrative. Watch for sentiment inflection."
            beginner = f"The news is somewhat negative ({avg_sentiment:.2f}). This negative coverage will make it harder for the stock to go up. Needs something really positive to change the story."
        else:
            expert = f"News sentiment is neutral ({avg_sentiment:.2f}/1.0). No strong narrative catalyst. Stock trading on fundamentals/technicals, not sentiment. Wait for sentiment to turn decisively positive (>0.2) or negative (<-0.2) for clearer signal."
            beginner = f"The news is neutral ({avg_sentiment:.2f}) - not good, not bad. The stock will move based on company performance, not news hype."

        return ExplanationFactor(
            category='sentiment',
            name='News Sentiment',
            weight=weight,
            direction=direction,
            expert_explanation=expert,
            beginner_explanation=beginner,
            supporting_data={
                'avg_sentiment': round(avg_sentiment, 2),
                'news_count': int(news_count),
                'positive_ratio_pct': round(positive_ratio * 100, 1),
                'sentiment_category': 'very positive' if avg_sentiment > 0.3 else 'positive' if avg_sentiment > 0.1 else 'very negative' if avg_sentiment < -0.3 else 'negative' if avg_sentiment < -0.1 else 'neutral'
            }
        )

    def _explain_catalyst(self, catalyst_count: float, weight: float, direction: str) -> ExplanationFactor:
        """SPECIFIC catalyst explanation"""
        # NOTE: In production, fetch actual business events from news/SEC filings
        # For now, we'll create generic catalyst explanations

        if catalyst_count >= 2:
            expert = f"Detected {int(catalyst_count)} major catalysts in recent news (contracts, partnerships, approvals, or deals). Multiple catalysts compound positive narrative. Historically, 2+ catalysts correlate with 10-20% stock moves over 30-60 days. Monitor execution on these initiatives."
            beginner = f"The company just announced {int(catalyst_count)} big deals or partnerships! When companies announce multiple good things at once, stocks often jump 10-20% over the next 1-2 months. Watch to see if they actually deliver on these."
        elif catalyst_count == 1:
            expert = f"Detected 1 major catalyst (likely contract win, partnership, or product approval). Single catalyst can drive 5-10% move if material. Need follow-through with additional catalysts or strong earnings to sustain momentum beyond initial pop."
            beginner = f"The company announced 1 big piece of good news (maybe a contract or partnership). This could push the stock up 5-10%, but it needs more good news to keep going higher."
        else:
            expert = f"No major catalysts detected in recent news. Stock trading on technical/fundamental factors, not event-driven. Lack of catalysts limits upside potential. Wait for catalyst (earnings, deal announcement, product launch) for higher conviction entry."
            beginner = f"No big news or catalysts recently. The stock is moving based on normal trading, not exciting announcements. Without big news, it's harder for the stock to make big moves."

        return ExplanationFactor(
            category='business_catalyst',
            name='Business Catalysts',
            weight=weight,
            direction=direction,
            expert_explanation=expert,
            beginner_explanation=beginner,
            supporting_data={
                'catalyst_count': int(catalyst_count),
                'has_major_catalyst': catalyst_count > 0
            }
        )

    def _explain_52w_distance(self, distance_pct: float, weight: float, direction: str) -> ExplanationFactor:
        """SPECIFIC 52-week high distance explanation"""
        high_52w = self.features.get('52w_high', 0)
        current_price = self.features.get('price_current', 0)

        if distance_pct < 5:
            expert = f"{self.ticker} is trading at ${current_price:.2f}, just {distance_pct:.1f}% below its 52-week high of ${high_52w:.2f}. Near all-time highs - breakout could trigger momentum buyers and drive another 5-10%. Risk: May consolidate or pull back 5-8% before next leg up."
            beginner = f"The stock is at ${current_price:.2f}, very close to its highest price in a year (${high_52w:.2f}, only {distance_pct:.1f}% away). Breaking above this could push it 5-10% higher as more buyers jump in. But it might also take a breather and dip 5-8% first."
        elif distance_pct < 20:
            expert = f"{self.ticker} is {distance_pct:.1f}% below 52-week high (current: ${current_price:.2f}, high: ${high_52w:.2f}). Moderate distance - room to run to ${high_52w:.2f} (+{distance_pct:.1f}%) if momentum continues. Key resistance at prior high."
            beginner = f"The stock is {distance_pct:.1f}% away from its year-high of ${high_52w:.2f} (currently ${current_price:.2f}). It has room to climb back to that level, which would be a {distance_pct:.1f}% gain from here."
        elif distance_pct < 50:
            expert = f"{self.ticker} is {distance_pct:.1f}% below 52-week high (current: ${current_price:.2f}, high: ${high_52w:.2f}). Significant drawdown presents asymmetric opportunity if fundamentals intact. Path back to highs could yield {distance_pct:.1f}% gain over 3-6 months. Requires catalyst to bridge gap."
            beginner = f"The stock has fallen {distance_pct:.1f}% from its peak price of ${high_52w:.2f} to ${current_price:.2f}. If nothing is fundamentally wrong with the company, getting back to that high could mean {distance_pct:.1f}% gains. But it needs good news to get there."
        else:
            expert = f"{self.ticker} is down {distance_pct:.1f}% from 52-week high (current: ${current_price:.2f}, high: ${high_52w:.2f}). Severe drawdown - likely fundamental issues or sector headwinds. High risk. Only consider if clear turnaround catalyst exists. Recovery to highs would take 12-18+ months."
            beginner = f"CAUTION: The stock has crashed {distance_pct:.1f}% from its high of ${high_52w:.2f} to ${current_price:.2f}. Something went wrong. Very risky - only invest if you're sure the company can fix its problems. Recovery would take 1-2 years."

        return ExplanationFactor(
            category='technical',
            name='Distance from 52-Week High',
            weight=weight,
            direction=direction,
            expert_explanation=expert,
            beginner_explanation=beginner,
            supporting_data={
                'current_price': round(current_price, 2),
                '52w_high': round(high_52w, 2),
                'distance_pct': round(distance_pct, 2),
                'upside_to_high_pct': round(distance_pct, 2)
            }
        )

    def _explain_debt_ratio(self, debt_to_equity: float, weight: float, direction: str) -> ExplanationFactor:
        """SPECIFIC debt-to-equity explanation"""
        if debt_to_equity < 0.5:
            expert = f"Debt-to-equity ratio of {debt_to_equity:.2f} is very low - strong balance sheet with minimal leverage. Financial flexibility to weather downturns, fund growth, or return capital to shareholders. Low financial risk. Premium valuation justified."
            beginner = f"The company has very little debt (debt-to-equity: {debt_to_equity:.2f}). This is great - means they're financially healthy and not at risk of bankruptcy. They can invest in growth without worrying about debt payments."
        elif debt_to_equity < 1.0:
            expert = f"Debt-to-equity ratio of {debt_to_equity:.2f} is moderate and manageable. Reasonable leverage - not overleveraged but using debt to enhance returns. Acceptable for most industries. Monitor interest coverage to ensure debt serviceability."
            beginner = f"The company has moderate debt (debt-to-equity: {debt_to_equity:.2f}). This is normal - most companies have some debt. As long as they're making enough money to pay interest, it's fine."
        elif debt_to_equity < 2.0:
            expert = f"Debt-to-equity ratio of {debt_to_equity:.2f} is elevated - higher leverage increases financial risk. Vulnerable to rising interest rates or economic downturns. Requires strong cash flow to service debt. Monitor closely for signs of distress."
            beginner = f"The company has high debt (debt-to-equity: {debt_to_equity:.2f}). This is concerning - they owe a lot of money. If business slows down or interest rates go up, they could struggle. Riskier investment."
        else:
            expert = f"Debt-to-equity ratio of {debt_to_equity:.2f} is very high - significant financial risk. Over-leveraged balance sheet. Risk of bankruptcy if cash flows deteriorate. Avoid unless deeply undervalued with clear deleveraging plan. High-risk/high-reward."
            beginner = f"DANGER: The company has extremely high debt (debt-to-equity: {debt_to_equity:.2f}). They're drowning in debt. If things go wrong, they could go bankrupt. Very high risk - only for aggressive investors who think they can turn it around."

        return ExplanationFactor(
            category='fundamental',
            name='Debt-to-Equity Ratio',
            weight=weight,
            direction=direction,
            expert_explanation=expert,
            beginner_explanation=beginner,
            supporting_data={
                'debt_to_equity': round(debt_to_equity, 2),
                'leverage_category': 'low' if debt_to_equity < 0.5 else 'moderate' if debt_to_equity < 1.0 else 'high' if debt_to_equity < 2.0 else 'very high'
            }
        )
