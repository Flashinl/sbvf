from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime, timedelta

try:
    # Only for type hints; avoid hard dependency at import time
    from .providers.price_yf import PriceSnapshot, Technicals
    from .providers.news_newsapi import NewsItem
except Exception:  # pragma: no cover
    PriceSnapshot = object  # type: ignore
    Technicals = object  # type: ignore
    NewsItem = object  # type: ignore


@dataclass
class Recommendation:
    label: str
    confidence: float  # 0..100
    rationale: str


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _avg(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def recommend(price: "PriceSnapshot", tech: "Technicals", news: List["NewsItem"], *, risk: str = "medium") -> Recommendation:
    """Combine fundamentals, technicals, and news into a simple Buy/Hold/Sell.

    Returns:
        Recommendation(label, confidence, rationale)
    """
    rationale_lines: List[str] = []

    # Technicals (primary)
    trend = getattr(tech, "trend_score", 0.0) or 0.0  # expected -1..1
    rsi = getattr(tech, "rsi14", None)
    tech_score = _clamp(float(trend), -1.0, 1.0)
    if rsi is not None:
        if rsi < 30:
            tech_score += 0.2; rationale_lines.append("RSI oversold (<30)")
        elif rsi < 45:
            tech_score += 0.05; rationale_lines.append("RSI supportive (30-45)")
        elif rsi > 70:
            tech_score -= 0.2; rationale_lines.append("RSI overbought (>70)")
        elif rsi > 60:
            tech_score -= 0.05; rationale_lines.append("RSI stretched (60-70)")
        else:
            rationale_lines.append("RSI neutral")

    # Fundamentals (lightweight)
    pe = getattr(price, "trailing_pe", None)
    dy = getattr(price, "dividend_yield", None)
    fund_score = 0.0
    if pe is not None:
        if pe < 12:
            fund_score += 0.15; rationale_lines.append("Attractive PE")
        elif pe < 35:
            fund_score += 0.05; rationale_lines.append("Reasonable PE")
        elif pe > 40:
            fund_score -= 0.1; rationale_lines.append("Rich PE")
    if dy and dy > 0.02:
        fund_score += 0.05; rationale_lines.append("Income support (dividend)")

    # News sentiment
    sentiments = []
    for n in news or []:
        s = getattr(n, "sentiment", 0.0)
        if s is not None:
            sentiments.append(float(s))
    news_score = _clamp(_avg(sentiments), -1.0, 1.0) * 0.5  # dampen a bit
    if sentiments:
        rationale_lines.append(f"News sentiment {news_score:+.2f}")

    # Earnings proximity risk adjustment
    ed: Optional[datetime] = getattr(price, "earnings_date", None)
    if ed:
        days = (ed - datetime.utcnow()).days
        if -1 <= days <= 3:  # within ~3 days ahead or very recent
            fund_score -= 0.05
            tech_score -= 0.05
            rationale_lines.append("Earnings proximity risk")

    # Combine with weights
    total = 0.5 * tech_score + 0.25 * fund_score + 0.25 * news_score

    # Risk appetite scaling
    risk = (risk or "medium").lower()
    risk_factor = {"low": 0.9, "medium": 1.0, "high": 1.1}.get(risk, 1.0)
    total *= risk_factor
    total = _clamp(total, -1.0, 1.0)

    # Map to label
    if total >= 0.25:
        label = "Buy"
    elif total <= -0.25:
        label = "Sell"
    else:
        label = "Hold"

    # Confidence as 50% base plus magnitude
    confidence = _clamp(50.0 + 45.0 * abs(total), 1.0, 99.0)

    # Build rationale
    if not rationale_lines:
        rationale_lines.append("Mixed signals")
    rationale = "\n".join(f"• {line}" for line in rationale_lines)

    return Recommendation(label=label, confidence=round(confidence, 2), rationale=rationale)
