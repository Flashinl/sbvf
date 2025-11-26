"""
Catalyst Detection and Reasoning Module

Analyzes news to identify and classify business catalysts (deals, contracts,
partnerships, etc.) and provides explanations for stock price movements.

Integrates with:
- Existing news providers (NewsAPI, Finnhub, Polygon)
- NLP infrastructure (spacy, trafilatura)
- ML predictions (to explain model outputs)
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import re

import spacy
from ..providers.news_newsapi import NewsItem
from ..nlp import _get_nlp, _fetch_text


@dataclass
class Catalyst:
    """Structured catalyst event"""
    type: str  # 'partnership', 'contract', 'acquisition', 'earnings', etc.
    description: str  # Human-readable description
    impact_direction: str  # 'bullish', 'bearish', 'neutral'
    impact_level: str  # 'low', 'medium', 'high'
    confidence: float  # 0-1, how confident we are in this detection
    source_url: Optional[str] = None
    published_at: Optional[str] = None
    entities: List[str] = None  # Companies/organizations involved
    key_facts: List[str] = None  # Supporting facts

    def __post_init__(self):
        if self.entities is None:
            self.entities = []
        if self.key_facts is None:
            self.key_facts = []


class CatalystDetector:
    """
    Detects business catalysts from news articles

    Catalyst types:
    - partnership: Strategic partnerships, collaborations
    - contract: New contracts, orders, deals
    - acquisition: M&A activity
    - earnings: Earnings beats/misses, guidance changes
    - product_launch: New product announcements
    - regulatory: FDA approvals, regulatory decisions
    - executive: Leadership changes
    - analyst: Analyst upgrades/downgrades
    """

    def __init__(self):
        self.nlp = _get_nlp()

        # Catalyst type patterns
        self.patterns = {
            'partnership': {
                'keywords': [
                    'partner', 'partnership', 'collaborate', 'collaboration',
                    'alliance', 'joint venture', 'strategic partnership',
                    'teamed up', 'teaming up', 'joins forces'
                ],
                'action_verbs': ['announce', 'sign', 'form', 'enter', 'launch'],
                'impact_default': 'bullish'
            },
            'contract': {
                'keywords': [
                    'contract', 'order', 'deal', 'agreement', 'award',
                    'customer', 'win', 'secured', 'lands', 'wins bid',
                    'multi-year', 'supply agreement', 'purchase order'
                ],
                'action_verbs': ['sign', 'award', 'secure', 'win', 'announce', 'land'],
                'impact_default': 'bullish'
            },
            'acquisition': {
                'keywords': [
                    'acquire', 'acquisition', 'merger', 'buyout', 'takeover',
                    'purchase', 'buys', 'bought', 'acquire stake', 'strategic investment'
                ],
                'action_verbs': ['announce', 'complete', 'close', 'agree to'],
                'impact_default': 'bullish'
            },
            'earnings': {
                'keywords': [
                    'earnings', 'revenue', 'profit', 'beat', 'miss', 'eps',
                    'guidance', 'outlook', 'forecast', 'raises guidance',
                    'cuts guidance', 'quarterly results', 'fiscal quarter'
                ],
                'action_verbs': ['report', 'post', 'announce', 'beat', 'miss', 'raise', 'cut'],
                'impact_default': 'neutral'  # Depends on context
            },
            'product_launch': {
                'keywords': [
                    'launch', 'unveil', 'introduce', 'release', 'debut',
                    'new product', 'product line', 'rollout', 'announces product'
                ],
                'action_verbs': ['launch', 'unveil', 'introduce', 'announce', 'release'],
                'impact_default': 'bullish'
            },
            'regulatory': {
                'keywords': [
                    'fda', 'approval', 'approved', 'clearance', 'regulatory',
                    'clinical trial', 'phase', 'patent', 'intellectual property',
                    'regulator', 'compliance'
                ],
                'action_verbs': ['approve', 'grant', 'clear', 'authorize', 'deny', 'reject'],
                'impact_default': 'neutral'  # Depends on outcome
            },
            'executive': {
                'keywords': [
                    'ceo', 'cfo', 'coo', 'president', 'executive', 'board',
                    'appoint', 'hire', 'resign', 'depart', 'leadership'
                ],
                'action_verbs': ['appoint', 'hire', 'name', 'resign', 'step down', 'join'],
                'impact_default': 'neutral'
            },
            'analyst': {
                'keywords': [
                    'upgrade', 'downgrade', 'rating', 'price target',
                    'analyst', 'initiate', 'coverage', 'outperform', 'underperform'
                ],
                'action_verbs': ['upgrade', 'downgrade', 'raise', 'lower', 'initiate'],
                'impact_default': 'neutral'
            }
        }

        # Sentiment indicators for impact direction
        self.bullish_terms = {
            'beat', 'beats', 'surge', 'soar', 'strong', 'growth', 'outperform',
            'upgrade', 'raise', 'raises', 'record', 'expansion', 'win', 'wins',
            'approved', 'approval', 'success', 'breakthrough', 'major deal'
        }

        self.bearish_terms = {
            'miss', 'misses', 'fall', 'drop', 'weak', 'decline', 'downgrade',
            'cut', 'cuts', 'slowdown', 'slump', 'lawsuit', 'investigation',
            'delay', 'denied', 'reject', 'loss', 'losses', 'concern'
        }

    def detect_catalysts(
        self,
        news: List[NewsItem],
        max_articles: int = 10,
        min_confidence: float = 0.3
    ) -> List[Catalyst]:
        """
        Detect catalysts from news articles

        Args:
            news: List of news items
            max_articles: Maximum articles to analyze
            min_confidence: Minimum confidence threshold

        Returns:
            List of detected catalysts, sorted by confidence
        """
        catalysts = []

        for item in news[:max_articles]:
            # Analyze title and description for quick detection
            title = item.title or ""

            # Try to fetch full text for better analysis
            full_text = ""
            if item.url:
                full_text = _fetch_text(item.url)

            # Use title + snippet of full text
            analysis_text = title
            if full_text:
                # Take first 1000 chars for efficiency
                analysis_text += " " + full_text[:1000]

            # Detect catalyst type and extract info
            detected = self._analyze_text(
                analysis_text,
                title,
                item.url,
                item.published_at
            )

            if detected and detected.confidence >= min_confidence:
                catalysts.append(detected)

        # Sort by confidence and impact level
        impact_weights = {'high': 3, 'medium': 2, 'low': 1}
        catalysts.sort(
            key=lambda c: (impact_weights.get(c.impact_level, 0), c.confidence),
            reverse=True
        )

        return catalysts

    def _analyze_text(
        self,
        text: str,
        title: str,
        url: Optional[str],
        published_at: Optional[str]
    ) -> Optional[Catalyst]:
        """Analyze text to detect catalyst"""
        text_lower = text.lower()
        title_lower = title.lower()

        # Find matching catalyst type
        best_match = None
        best_score = 0.0

        for cat_type, pattern in self.patterns.items():
            # Count keyword matches
            keyword_matches = sum(
                1 for kw in pattern['keywords']
                if kw in text_lower
            )

            # Count action verb matches
            verb_matches = sum(
                1 for verb in pattern['action_verbs']
                if verb in text_lower
            )

            # Calculate score (prefer keywords in title)
            score = keyword_matches + verb_matches * 0.5
            if any(kw in title_lower for kw in pattern['keywords']):
                score += 2.0  # Boost if keyword in title

            if score > best_score:
                best_score = score
                best_match = cat_type

        if not best_match or best_score < 1.0:
            return None

        # Determine impact direction
        impact_direction = self._determine_impact_direction(
            text_lower,
            title_lower,
            best_match
        )

        # Determine impact level
        impact_level = self._determine_impact_level(
            title_lower,
            text_lower,
            best_match
        )

        # Extract entities (companies, organizations)
        entities = self._extract_entities(text)

        # Extract key facts
        key_facts = self._extract_key_facts(text, best_match)

        # Generate description
        description = self._generate_description(
            best_match,
            title,
            entities,
            impact_direction
        )

        # Calculate confidence
        confidence = min(best_score / 5.0, 1.0)  # Normalize

        return Catalyst(
            type=best_match,
            description=description,
            impact_direction=impact_direction,
            impact_level=impact_level,
            confidence=confidence,
            source_url=url,
            published_at=published_at,
            entities=entities,
            key_facts=key_facts
        )

    def _determine_impact_direction(
        self,
        text: str,
        title: str,
        catalyst_type: str
    ) -> str:
        """Determine bullish/bearish/neutral direction"""
        # Count sentiment words (prioritize title)
        title_bullish = sum(1 for term in self.bullish_terms if term in title)
        title_bearish = sum(1 for term in self.bearish_terms if term in title)
        text_bullish = sum(1 for term in self.bullish_terms if term in text)
        text_bearish = sum(1 for term in self.bearish_terms if term in text)

        # Weight title more heavily
        bullish_score = title_bullish * 2 + text_bullish
        bearish_score = title_bearish * 2 + text_bearish

        if bullish_score > bearish_score + 1:
            return 'bullish'
        elif bearish_score > bullish_score + 1:
            return 'bearish'
        else:
            # Fall back to default for catalyst type
            return self.patterns[catalyst_type]['impact_default']

    def _determine_impact_level(
        self,
        title: str,
        text: str,
        catalyst_type: str
    ) -> str:
        """Determine impact level (high/medium/low)"""
        # High impact indicators
        high_impact_terms = {
            'major', 'significant', 'massive', 'historic', 'record-breaking',
            'multi-billion', 'largest', 'breakthrough', 'transformative',
            '$1 billion', '$1b', 'billion-dollar'
        }

        # Medium impact indicators
        medium_impact_terms = {
            'multi-year', 'strategic', 'expanded', 'key', 'important',
            'notable', 'substantial', 'multi-million'
        }

        combined = title + " " + text

        if any(term in combined for term in high_impact_terms):
            return 'high'
        elif any(term in combined for term in medium_impact_terms):
            return 'medium'
        elif catalyst_type in ['acquisition', 'earnings', 'regulatory']:
            # These types are inherently higher impact
            return 'medium'
        else:
            return 'low'

    def _extract_entities(self, text: str) -> List[str]:
        """Extract company/organization names"""
        doc = self.nlp(text[:2000])  # Limit for performance
        entities = []

        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PERSON', 'PRODUCT']:
                # Clean up entity text
                clean = ent.text.strip()
                if len(clean) > 2 and clean not in entities:
                    entities.append(clean)

        return entities[:5]  # Top 5 entities

    def _extract_key_facts(self, text: str, catalyst_type: str) -> List[str]:
        """Extract key facts related to catalyst"""
        doc = self.nlp(text[:2000])
        facts = []

        pattern = self.patterns[catalyst_type]
        keywords = set(pattern['keywords'])

        # Find sentences containing catalyst keywords
        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_lower = sent_text.lower()

            # Skip if too short or too long
            if len(sent_text) < 20 or len(sent_text) > 200:
                continue

            # Skip if too many numbers (likely financial data)
            digit_ratio = sum(c.isdigit() for c in sent_text) / max(len(sent_text), 1)
            if digit_ratio > 0.25:
                continue

            # Check if contains catalyst keywords
            if any(kw in sent_lower for kw in keywords):
                facts.append(sent_text)

            if len(facts) >= 3:
                break

        return facts

    def _generate_description(
        self,
        catalyst_type: str,
        title: str,
        entities: List[str],
        impact_direction: str
    ) -> str:
        """Generate human-readable description"""
        # Extract main entity (usually first one)
        main_entity = entities[0] if entities else "Company"

        # Type-specific templates
        templates = {
            'partnership': f"{main_entity} announced a partnership",
            'contract': f"{main_entity} secured a new contract",
            'acquisition': f"{main_entity} announced an acquisition",
            'earnings': f"{main_entity} reported earnings",
            'product_launch': f"{main_entity} launched a new product",
            'regulatory': f"{main_entity} received regulatory news",
            'executive': f"{main_entity} announced leadership changes",
            'analyst': f"Analysts updated their view on {main_entity}"
        }

        base = templates.get(catalyst_type, f"{main_entity} in the news")

        # Add sentiment context
        if impact_direction == 'bullish':
            base += " (positive development)"
        elif impact_direction == 'bearish':
            base += " (concerning development)"

        # If title is concise, use it directly
        if len(title) < 100:
            return title

        return base


class ExplanationGenerator:
    """
    Generates explanations for ML predictions using detected catalysts
    """

    def __init__(self):
        self.catalyst_detector = CatalystDetector()

    def generate_explanation(
        self,
        ticker: str,
        ml_prediction: Dict[str, Any],
        news: List[NewsItem],
        horizon: str = '1month'
    ) -> Dict[str, Any]:
        """
        Generate explanation combining ML prediction with catalysts

        Args:
            ticker: Stock symbol
            ml_prediction: ML model prediction dict
            news: List of news items
            horizon: Time horizon ('1week', '1month', '3month')

        Returns:
            Explanation dict with:
            - signal: BUY/HOLD/SELL
            - reasoning: Human-readable explanation
            - catalysts: List of detected catalysts
            - confidence: Overall confidence
            - key_factors: Top factors driving prediction
        """
        # Get prediction for specified horizon
        pred = ml_prediction.get(horizon, {})
        signal = pred.get('signal', 'HOLD')
        ml_confidence = pred.get('confidence', 0.5)
        expected_return = pred.get('expected_return', 0.0)

        # Detect catalysts
        catalysts = self.catalyst_detector.detect_catalysts(news, max_articles=10)

        # Analyze catalyst alignment with ML prediction
        catalyst_support = self._assess_catalyst_support(signal, catalysts)

        # Generate reasoning
        reasoning = self._generate_reasoning(
            ticker,
            signal,
            expected_return,
            ml_confidence,
            catalysts,
            catalyst_support,
            horizon
        )

        # Identify key factors
        key_factors = self._identify_key_factors(
            signal,
            catalysts,
            ml_confidence,
            catalyst_support
        )

        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            ml_confidence,
            catalyst_support,
            len(catalysts)
        )

        return {
            'ticker': ticker,
            'signal': signal,
            'horizon': horizon,
            'reasoning': reasoning,
            'catalysts': [
                {
                    'type': c.type,
                    'description': c.description,
                    'impact_direction': c.impact_direction,
                    'impact_level': c.impact_level,
                    'entities': c.entities,
                    'key_facts': c.key_facts,
                    'source_url': c.source_url
                }
                for c in catalysts[:5]  # Top 5 catalysts
            ],
            'key_factors': key_factors,
            'confidence': overall_confidence,
            'ml_confidence': ml_confidence,
            'catalyst_support': catalyst_support,
            'expected_return': expected_return
        }

    def _assess_catalyst_support(
        self,
        signal: str,
        catalysts: List[Catalyst]
    ) -> float:
        """
        Assess how well catalysts support the ML signal

        Returns:
            Score from 0 to 1
        """
        if not catalysts:
            return 0.5  # Neutral if no catalysts

        # Count supporting vs conflicting catalysts
        supporting = 0
        conflicting = 0

        for catalyst in catalysts:
            if signal == 'BUY':
                if catalyst.impact_direction == 'bullish':
                    weight = {'high': 1.0, 'medium': 0.7, 'low': 0.4}[catalyst.impact_level]
                    supporting += weight
                elif catalyst.impact_direction == 'bearish':
                    weight = {'high': 1.0, 'medium': 0.7, 'low': 0.4}[catalyst.impact_level]
                    conflicting += weight

            elif signal == 'SELL':
                if catalyst.impact_direction == 'bearish':
                    weight = {'high': 1.0, 'medium': 0.7, 'low': 0.4}[catalyst.impact_level]
                    supporting += weight
                elif catalyst.impact_direction == 'bullish':
                    weight = {'high': 1.0, 'medium': 0.7, 'low': 0.4}[catalyst.impact_level]
                    conflicting += weight

        # Calculate support score
        total = supporting + conflicting
        if total == 0:
            return 0.5

        return supporting / total

    def _generate_reasoning(
        self,
        ticker: str,
        signal: str,
        expected_return: float,
        ml_confidence: float,
        catalysts: List[Catalyst],
        catalyst_support: float,
        horizon: str
    ) -> str:
        """Generate human-readable reasoning"""
        parts = []

        # Opening statement
        horizon_text = horizon.replace('week', ' week').replace('month', ' month')
        parts.append(
            f"The model predicts a {signal} signal for {ticker} over the {horizon_text} "
            f"timeframe with {expected_return:+.1f}% expected return."
        )

        # Catalyst analysis
        if catalysts:
            bullish_cats = [c for c in catalysts if c.impact_direction == 'bullish']
            bearish_cats = [c for c in catalysts if c.impact_direction == 'bearish']

            if bullish_cats:
                top_bullish = bullish_cats[0]
                parts.append(
                    f"Recent positive catalyst: {top_bullish.description}."
                )

            if bearish_cats:
                top_bearish = bearish_cats[0]
                parts.append(
                    f"Potential concern: {top_bearish.description}."
                )

            # Alignment assessment
            if catalyst_support > 0.7:
                parts.append(
                    "News catalysts strongly support this prediction."
                )
            elif catalyst_support < 0.3:
                parts.append(
                    "News catalysts suggest caution with this prediction."
                )
        else:
            parts.append(
                "Limited recent news catalysts detected. Prediction based primarily on technical patterns."
            )

        # Confidence context
        if ml_confidence > 0.8:
            parts.append(
                "The model has high confidence in this prediction."
            )
        elif ml_confidence < 0.5:
            parts.append(
                "The model has lower confidence - consider this a weaker signal."
            )

        return " ".join(parts)

    def _identify_key_factors(
        self,
        signal: str,
        catalysts: List[Catalyst],
        ml_confidence: float,
        catalyst_support: float
    ) -> List[Dict[str, Any]]:
        """Identify key factors driving the prediction"""
        factors = []

        # Add top catalysts as factors
        for catalyst in catalysts[:3]:
            # Determine if catalyst supports or conflicts with signal
            alignment = 'supporting' if (
                (signal == 'BUY' and catalyst.impact_direction == 'bullish') or
                (signal == 'SELL' and catalyst.impact_direction == 'bearish')
            ) else 'conflicting' if (
                (signal == 'BUY' and catalyst.impact_direction == 'bearish') or
                (signal == 'SELL' and catalyst.impact_direction == 'bullish')
            ) else 'neutral'

            factors.append({
                'category': 'business_catalyst',
                'name': catalyst.type.replace('_', ' ').title(),
                'description': catalyst.description,
                'direction': catalyst.impact_direction,
                'weight': {'high': 0.9, 'medium': 0.6, 'low': 0.3}[catalyst.impact_level],
                'alignment': alignment
            })

        # Add ML model confidence as a factor
        factors.append({
            'category': 'model_confidence',
            'name': 'ML Model Confidence',
            'description': f"Model confidence: {ml_confidence:.1%}",
            'direction': 'positive' if ml_confidence > 0.6 else 'neutral',
            'weight': ml_confidence,
            'alignment': 'supporting'
        })

        # Add catalyst alignment as a factor
        if catalysts:
            factors.append({
                'category': 'catalyst_alignment',
                'name': 'News Catalyst Alignment',
                'description': f"Catalyst support: {catalyst_support:.1%}",
                'direction': 'positive' if catalyst_support > 0.6 else 'negative' if catalyst_support < 0.4 else 'neutral',
                'weight': abs(catalyst_support - 0.5) * 2,  # Distance from neutral
                'alignment': 'supporting' if catalyst_support > 0.6 else 'conflicting' if catalyst_support < 0.4 else 'neutral'
            })

        # Sort by weight
        factors.sort(key=lambda f: f['weight'], reverse=True)

        return factors

    def _calculate_overall_confidence(
        self,
        ml_confidence: float,
        catalyst_support: float,
        num_catalysts: int
    ) -> float:
        """Calculate overall confidence combining ML and catalysts"""
        # Base confidence from ML
        confidence = ml_confidence * 0.6

        # Add catalyst support (weighted by number of catalysts)
        catalyst_weight = min(num_catalysts / 5.0, 0.3)  # Up to 30% weight
        confidence += catalyst_support * catalyst_weight

        # Add small bonus for having catalysts
        if num_catalysts > 0:
            confidence += 0.1

        return min(confidence, 1.0)


# Convenience function
def explain_prediction(
    ticker: str,
    ml_prediction: Dict[str, Any],
    news: List[NewsItem],
    horizon: str = '1month'
) -> Dict[str, Any]:
    """
    Convenience function to generate explanation

    Usage:
        from stockbot.ml.catalyst_detection import explain_prediction

        explanation = explain_prediction(
            ticker='AAPL',
            ml_prediction=predictions,
            news=news_items,
            horizon='1month'
        )

        print(explanation['reasoning'])
        for catalyst in explanation['catalysts']:
            print(f"- {catalyst['description']}")
    """
    generator = ExplanationGenerator()
    return generator.generate_explanation(ticker, ml_prediction, news, horizon)
