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
    predicted_price: float | None = None


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
    """Momentum-first with 'hidden gem' overlay.

    Keeps broad coverage (works without fundamentals) but overlays:
      - Small/mid-cap check (≈$200M–$5B)
      - Drawdown vs 52W high as proxy for asymmetric setup
      - Simple catalyst detection from headlines
      - Megatrend alignment via sector/industry keywords
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

    # Hidden gem overlay
    hidden_score = 0.0
    mc = getattr(price, "market_cap", None)
    price_now = getattr(price, "price", None)
    high_52w = getattr(price, "fifty_two_week_high", None)

    # Small/mid-cap window
    if mc is not None:
        if 2e8 <= mc <= 5e9:
            hidden_score += 0.05; rationale_lines.append("Small/mid-cap range (≈$200M–$5B)")
        elif mc > 2e10:
            hidden_score -= 0.02

    # Drawdown from 52W high as proxy for asymmetric setup
    dd_txt = None
    if price_now is not None and high_52w:
        dd = 1.0 - float(price_now) / float(high_52w)
        if dd > 0:
            dd_txt = f"{dd*100:.0f}% below 52W high"
            rationale_lines.append(dd_txt)
            if 0.7 <= dd <= 0.9:
                hidden_score += 0.10  # attractive asymmetry window
            elif dd > 0.9:
                hidden_score -= 0.05  # possibly distressed

    # Catalyst detection from headlines
    catalyst_keywords = [
        "award", "contract", "partnership", "deal", "grant", "funding", "order",
        "acquisition", "merger", "approval", "phase", "secures", "expansion",
        "agreement", "strategic", "pilot", "deployment", "doe", "dod", "nasa"
    ]
    hits = []
    for n in news or []:
        title = (getattr(n, "title", "") or "").lower()
        if any(k in title for k in catalyst_keywords):
            hits.append(title)
    if hits:
        hidden_score += min(0.12, 0.04 * len(hits))
        rationale_lines.append("Recent potential catalyst(s) in headlines")

    # Megatrend alignment via sector/industry
    sec = (getattr(price, "sector", None) or "").lower()
    ind = (getattr(price, "industry", None) or "").lower()
    megatrend_terms = [
        "semiconductor", "chip", "ev", "electric", "battery", "solar", "renewable",
        "robot", "automation", "ai", "cyber", "cloud", "infrastructure", "storage"
    ]
    if any(term in sec or term in ind for term in megatrend_terms):
        hidden_score += 0.05
        rationale_lines.append("Aligned with structural megatrend")

    # Bound overlay contribution
    hidden_score = _clamp(hidden_score, -0.2, 0.25)

    # Combine with weights (momentum-first + hidden overlay)
    total = 0.7 * tech_score + 0.3 * news_score + hidden_score

    # Risk appetite scaling (kept; default 'medium')
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

    # Predicted price (simple heuristic target)
    predicted = None
    if price_now and price_now > 0:
        if label == "Buy":
            # Aim halfway to 52W high if available; otherwise a default 15-25% gain scaled by trend/catalysts
            base_up = 0.15 + (0.1 if trend > 0.6 else 0.05 if trend > 0.25 else 0)
            base_up += 0.05 if hits else 0
            base_up += 0.03 if (mc is not None and 2e8 <= mc <= 5e9 and any(term in sec or term in ind for term in megatrend_terms)) else 0
            if high_52w and high_52w > price_now:
                target = price_now + 0.5 * (high_52w - price_now)
                predicted = max(target, price_now * (1 + base_up))
                predicted = min(predicted, high_52w * 0.98)
            else:
                predicted = price_now * (1 + base_up)
        elif label == "Hold":
            predicted = price_now * (1 + (0.05 if trend > 0 else -0.05))
        else:  # Sell
            predicted = price_now * (1 - (0.10 if trend < -0.25 else 0.05))
        if predicted is not None and predicted <= 0:
            predicted = None

    # AI-style narrative: multi-source, nuanced, and explicit conditions
    t_qual = _qualitative_trend(trend)
    ed = getattr(price, "earnings_date", None)

    # Collect sources and categorize stories
    srcs = []
    titles = []
    for n in news or []:
        s = (getattr(n, "source", None) or "").strip()
        if s:
            srcs.append(s)
        t = (getattr(n, "title", "") or "").lower()
        if t:
            titles.append(t)

    def _has_any(t: str, keys: list[str]) -> bool:
        return any(k in t for k in keys)

    earnings_keys_pos = ["beat", "beats", "raise", "raises", "record", "above", "strong"]
    earnings_keys_neg = ["miss", "misses", "below", "cut", "cuts", "weak", "trim"]
    product_keys = ["launch", "product", "platform", "integration", "contract", "partnership", "deal", "order"]
    regulatory_keys = ["fda", "approval", "sec", "ftc", "lawsuit", "probe", "investigation", "fine"]
    analyst_keys = ["upgrade", "downgrade", "initiates", "price target", "pt "]
    ma_keys = ["acquisition", "merger", "buyout", "takeover"]
    financing_keys = ["offering", "debt", "convertible", "raise", "funding"]

    bull_stories = []
    bear_stories = []
    for t in titles[:12]:  # cap parsing
        if _has_any(t, earnings_keys_pos) or _has_any(t, product_keys) or _has_any(t, ma_keys) or ("approval" in t):
            bull_stories.append(t)
        if _has_any(t, earnings_keys_neg) or _has_any(t, regulatory_keys) or _has_any(t, financing_keys):
            bear_stories.append(t)

    reasons_moving: list[str] = []
    reasons_continue: list[str] = []

    # Why it's moving
    if hits:
        reasons_moving.append("fresh catalysts (e.g., contract/partnership/funding) in headlines")
    if sentiments:
        if news_score > 0.05:
            reasons_moving.append("news tone skewing positive")
        elif news_score < -0.05:
            reasons_moving.append("news tone cautious")
    if t_qual in ("uptrend", "strong uptrend"):
        reasons_moving.append("accumulation — buyers in control")
    elif t_qual in ("downtrend", "strong downtrend"):
        reasons_moving.append("distribution — sellers pressing")
    else:
        reasons_moving.append("range/consolidation resolving")
    # Earnings proximity
    try:
        if ed is not None:
            from datetime import datetime as _dt
            dt_now = _dt.utcnow()
            delta_days = (ed - dt_now).days
            if -7 <= delta_days <= 14:
                reasons_moving.append("flow around earnings timing")
    except Exception:
        pass

    # Why it can continue
    if any(term in sec or term in ind for term in megatrend_terms):
        reasons_continue.append("exposure to a structural megatrend")
    if mc is not None and 2e8 <= mc <= 5e9:
        reasons_continue.append("small/mid-cap runway if execution continues")
    if price_now and high_52w and (1 - float(price_now)/float(high_52w)) >= 0.2:
        reasons_continue.append("room to recover toward prior highs if momentum sustains")
    if hits:
        reasons_continue.append("pipeline of potential follow-on catalysts")
    if t_qual in ("uptrend", "strong uptrend"):
        reasons_continue.append("constructive structure until it fails")

    # Conditions for upside and invalidation
    cond_up: list[str] = []
    invalidation: list[str] = []
    if price_now is not None:
        pval = float(price_now)
        if sma50 and pval < float(sma50):
            cond_up.append("reclaim the 50-day moving average")
        if sma200 and pval < float(sma200):
            cond_up.append("reclaim the 200-day and hold above it")
        if high_52w and pval < float(high_52w):
            cond_up.append("establish higher highs/lows and work toward prior 52W high")
        if t_qual not in ("uptrend", "strong uptrend"):
            cond_up.append("shift from distribution to accumulation (trend improvement)")
        # Invalidation
        if sma50 and pval > float(sma50):
            invalidation.append("decisive close back below the 50-day (trend failure)")
        if sentiments and news_score < -0.2:
            invalidation.append("deterioration in news tone (guide cut or adverse headline)")

    why_moving = ", ".join(dict.fromkeys([r for r in reasons_moving if r])) or "no strong driver detected"
    why_continue = ", ".join(dict.fromkeys([r for r in reasons_continue if r])) or "needs fresh catalysts or improvement"

    # Compose multi-part narrative
    src_list = ", ".join(sorted(set(srcs)))[:140]
    bull_summary = "; ".join(bull_stories[:3])
    bear_summary = "; ".join(bear_stories[:3])
    cond_up_txt = "; ".join(cond_up) or "continued constructive price action and supportive news"
    invalid_txt = "; ".join(invalidation) or "pattern failure and negative guidance"

    # Company context (what they do)
    name = getattr(price, "long_name", None) or getattr(price, "ticker", "")
    sec_txt = getattr(price, "sector", None) or ""
    ind_txt = getattr(price, "industry", None) or ""
    summary = getattr(price, "long_business_summary", None)
    about_txt = None
    try:
        if summary:
            first_sentence = summary.strip().split(". ")[0].strip()
            if len(first_sentence) > 220:
                first_sentence = first_sentence[:217].rstrip() + "..."
            about_txt = f"{name} — {sec_txt}/{ind_txt}. {first_sentence}"
        else:
            about_txt = f"{name} — {sec_txt}/{ind_txt}".strip().rstrip(" -/")
    except Exception:
        about_txt = f"{name} — {sec_txt}/{ind_txt}".strip().rstrip(" -/")

    ai_analysis = (
        (f"About the company: {about_txt}. " if about_txt else "")
        + f"Why it's moving: {why_moving}. "
        + f"Why it can continue: {why_continue}. "
        + f"Bull case themes: {bull_summary or 'no clear bullish story from headlines'}. "
        + f"Bear case/risks: {bear_summary or 'no clear bearish story from headlines'}. "
        + f"For upside to play out: {cond_up_txt}. "
        + f"What would invalidate: {invalid_txt}. "
        + f"Sources scanned: {src_list or 'multiple aggregators'}. "
        + f"Bottom line: {label} with {round(confidence,2)}% confidence."
        + (f" Predicted price (near-term): ${predicted:.2f}." if predicted else "")
    )

    return Recommendation(
        label=label,
        confidence=round(confidence, 2),
        rationale=rationale,
        ai_analysis=ai_analysis,
        predicted_price=(float(predicted) if predicted is not None else None),
    )
