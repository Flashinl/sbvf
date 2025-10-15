from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

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
    ai_analysis: str


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _avg(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def _qualitative_trend(score: float) -> str:
    if score >= 0.6:
        return "strong uptrend"
    if score >= 0.25:
        return "uptrend"
    if score <= -0.6:
        return "strong downtrend"
    if score <= -0.25:
        return "downtrend"
    return "sideways"


def recommend(price: "PriceSnapshot", tech: "Technicals", news: List["NewsItem"], *, risk: str = "medium") -> Recommendation:
    """Momentum-first recommendation that works even when fundamentals are missing.

    We purposely ignore fundamentals to support more tickers (ETFs, ADRs, IPOs).
    """
    rationale_lines: List[str] = []

    # Technicals (primary)
    trend = float(getattr(tech, "trend_score", 0.0) or 0.0)  # expected -1..1
    rsi = getattr(tech, "rsi14", None)
    sma20 = getattr(tech, "sma20", None)
    sma50 = getattr(tech, "sma50", None)
    sma200 = getattr(tech, "sma200", None)

    tech_score = _clamp(trend, -1.0, 1.0)
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

    # SMA alignment bonus/penalty
    if all(v is not None for v in [sma20, sma50, sma200]):
        if sma20 > sma50 > sma200:
            tech_score += 0.15; rationale_lines.append("Bullish SMA stack (20>50>200)")
        elif sma20 < sma50 < sma200:
            tech_score -= 0.15; rationale_lines.append("Bearish SMA stack (20<50<200)")

    # News sentiment (secondary)
    sentiments = []
    for n in news or []:
        s = getattr(n, "sentiment", 0.0)
        if s is not None:
            sentiments.append(float(s))
    news_score = _clamp(_avg(sentiments), -1.0, 1.0) * 0.5  # dampen a bit
    if sentiments:
        rationale_lines.append(f"News sentiment {news_score:+.2f}")

    # Ignore fundamentals entirely to broaden coverage

    # Combine with weights (momentum-first)
    total = 0.7 * tech_score + 0.3 * news_score

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

    # Build rationale bullets
    if not rationale_lines:
        rationale_lines.append("Mixed signals")
    rationale = "\n".join(f"• {line}" for line in rationale_lines)

    # AI-style narrative analysis
    t_qual = _qualitative_trend(trend)
    rsi_txt = "unknown RSI"
    if rsi is not None:
        if rsi < 30: rsi_txt = f"oversold RSI ({rsi:.0f})"
        elif rsi > 70: rsi_txt = f"overbought RSI ({rsi:.0f})"
        else: rsi_txt = f"neutral RSI ({rsi:.0f})"
    sma_txt = ""
    if all(v is not None for v in [sma20, sma50, sma200]):
        if sma20 > sma50 > sma200:
            sma_txt = " with a constructive 20>50>200-day moving-average stack"
        elif sma20 < sma50 < sma200:
            sma_txt = " with a deteriorating 20<50<200-day moving-average stack"
    ns_txt = " and headlines are balanced"
    if sentiments:
        if news_score > 0.05: ns_txt = " and headlines skew positive"
        elif news_score < -0.05: ns_txt = " and headlines skew negative"

    ai_analysis = (
        f"Momentum currently suggests {t_qual}{sma_txt}. The setup is supported by {rsi_txt}{ns_txt}. "
        f"Given a {risk}-risk profile, this points to a {label.lower()} with {confidence:.0f}% confidence."
    )

    return Recommendation(
        label=label,
        confidence=round(confidence, 2),
        rationale=rationale,
        ai_analysis=ai_analysis,
    )
