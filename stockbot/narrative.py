from __future__ import annotations
import re
import hashlib
from typing import List, Optional

# Deterministic, template-based narrative composer.
# No external APIs. Varies phrasing using a seed derived from inputs.


def _scrub_numbers(text: str) -> str:
    if not text:
        return ""
    t = text
    # Remove currency and percentages and large standalone numbers
    t = re.sub(r"\$\s*\d[\d,\.]*", "", t)
    t = re.sub(r"\b\d+(?:\.\d+)?\s*%\b", "", t)
    # Drop common filing numbering like Item 1.01
    t = re.sub(r"Item\s+\d+\.\d+", "Item", t, flags=re.IGNORECASE)
    # Avoid day counts and explicit horizons
    t = re.sub(r"\b\d+\s*(days?|weeks?|months?|quarters?)\b", "near term", t, flags=re.IGNORECASE)
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _dedup(items: List[str], limit: int) -> List[str]:
    out: List[str] = []
    seen = set()
    for it in items or []:
        if not it:
            continue
        k = it.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(it.strip())
        if len(out) >= limit:
            break
    return out


def _choose(seed: int, options: List[str], offset: int = 0) -> str:
    if not options:
        return ""
    return options[(seed + offset) % len(options)]


def _rewrite_timing(msgs: List[str]) -> str:
    if not msgs:
        return ""
    joined = "; ".join(msgs)
    # Generalize timespans
    s = re.sub(r"\b\d+\s*[-–]\s*\d+\s*(days?|weeks?)\b", "the near term", joined, flags=re.IGNORECASE)
    s = re.sub(r"\b\d+\s*[-–]\s*\d+\s*months\b", "the next few quarters", s, flags=re.IGNORECASE)
    s = re.sub(r"\b\d+\s*months\b", "the next few quarters", s, flags=re.IGNORECASE)
    s = re.sub(r"\b\d+\s*weeks\b", "the near term", s, flags=re.IGNORECASE)
    s = re.sub(r"\b\d+\s*days\b", "the near term", s, flags=re.IGNORECASE)
    return _scrub_numbers(s)


def _conditions_text(cond_up: str, invalid: str) -> str:
    # Normalize mentions of specific moving averages
    txt = []
    cu = cond_up.lower() if cond_up else ""
    inv = invalid.lower() if invalid else ""

    cu_fragments = []
    if any(k in cu for k in ["50-day", "sma20", "sma50", "short-term"]):
        cu_fragments.append("reclaim short-term averages and hold above them")
    if any(k in cu for k in ["200-day", "sma200", "long-term"]):
        cu_fragments.append("maintain traction above long-term trend lines")
    if "higher highs" in cu or "higher lows" in cu:
        cu_fragments.append("keep forming higher highs and higher lows")
    if not cu_fragments and cu:
        cu_fragments.append(_scrub_numbers(cu))

    inv_fragments = []
    if any(k in inv for k in ["50-day", "sma20", "sma50", "short-term"]):
        inv_fragments.append("a decisive slip back below short-term support")
    if any(k in inv for k in ["200-day", "sma200", "long-term"]):
        inv_fragments.append("failure to hold key long-term levels")
    if "tone" in inv or "guidance" in inv or "headline" in inv:
        inv_fragments.append("a turn in tone around guidance or headlines")
    if not inv_fragments and inv:
        inv_fragments.append(_scrub_numbers(inv))

    cu_text = "; ".join(_dedup(cu_fragments, 3))
    inv_text = "; ".join(_dedup(inv_fragments, 3))

    base = []
    if cu_text:
        base.append(f"To go higher, the stock likely needs to {cu_text}.")
    if inv_text:
        base.append(f"What could go wrong includes {inv_text}.")
    return " ".join(base)


def compose_narrative(
    *,
    ticker: str,
    name: str,
    sector: str,
    industry: str,
    about: Optional[str],
    trend_label: str,
    why_moving: str,
    drivers_line: Optional[str],
    catalyst_details: List[str],
    nlp_drivers: List[str],
    facts: List[str],
    themes: List[str],
    timing_msgs: List[str],
    watch: List[str],
    bull_summary: str,
    bear_summary: str,
    cond_up_txt: str,
    invalid_txt: str,
    label: str,
) -> str:
    # Seed for deterministic variation
    seed_val = int(hashlib.sha256((ticker or name or "").encode("utf-8")).hexdigest(), 16)

    # Clean inputs
    about_clean = _scrub_numbers(about or "")
    why_clean = _scrub_numbers(why_moving)
    drivers_clean = _scrub_numbers(drivers_line or "")
    facts_clean = _dedup([_scrub_numbers(f) for f in facts], 4)
    themes_clean = _dedup([_scrub_numbers(t) for t in themes], 3)
    nlp_clean = _dedup([_scrub_numbers(d) for d in nlp_drivers], 3)
    catalysts = _dedup([drivers_clean] + nlp_clean, 3)

    # Paragraph 1: Company context
    p1_opts = [
        f"{name} operates in {sector}/{industry}. {about_clean}".strip(),
        f"{name} is positioned within {sector}/{industry}. {about_clean}".strip(),
        f"In the {sector}/{industry} space, {name} focuses on its core platform and execution. {about_clean}".strip(),
    ]
    p1 = _choose(seed_val, [s for s in p1_opts if s])

    # Paragraph 2: Catalysts synthesis
    cat_leads = [
        "Recent coverage points to concrete catalysts rather than chatter.",
        "Signals from multiple sources highlight tangible drivers, not just headlines.",
        "Momentum lately ties back to specific actions and company updates.",
    ]
    cat_body_parts = []
    if catalysts:
        cat_body_parts.append(f"Specific drivers include {', '.join(catalysts)}.")
    elif catalyst_details:
        cat_body_parts.append(f"Headlines flag {', '.join(_dedup(catalyst_details, 2))}.")
    if themes_clean:
        cat_body_parts.append(f"Common themes across sources center on {', '.join(themes_clean)}.")
    if facts_clean:
        cat_body_parts.append(f"Notably, filings and reports reference {', '.join(facts_clean)}.")
    p2 = " ".join([_choose(seed_val, cat_leads), " ".join(cat_body_parts)]).strip()

    # Paragraph 3: Timing and what the stock needs
    timing_text = _rewrite_timing(timing_msgs)
    watch_text = _scrub_numbers(", ".join(_dedup(watch, 4)))
    t_leads = [
        "Timing skews to the near term, with follow-through depending on execution.",
        "The setup is near-term sensitive but ultimately driven by delivery milestones.",
        "Expect the path to reflect both near-term tape action and medium-term proof points.",
    ]
    trend_line = _choose(seed_val, [
        f"Technically, the trend reads as {trend_label}.",
        f"From a technical lens, shares sit in a {trend_label}.",
        f"On the chart, the prevailing tone remains {trend_label}.",
    ], offset=1)
    needs_line = _conditions_text(cond_up_txt, invalid_txt)
    p3_parts = [
        _choose(seed_val, t_leads),
        (f"In practical terms: {timing_text}." if timing_text else ""),
        (f"Focus items include {watch_text}." if watch_text else ""),
        trend_line,
        needs_line,
    ]
    p3 = " ".join([s for s in p3_parts if s]).strip()

    # Paragraph 4: Positives and risks
    pos_leads = [
        "On the positive side, coverage emphasizes",
        "Constructively, recent commentary highlights",
        "Supportive elements include",
    ]
    risk_leads = [
        "Risks to monitor include",
        "Balanced against that are",
        "Counterpoints to watch are",
    ]
    positives = _scrub_numbers(bull_summary or "")
    risks = _scrub_numbers(bear_summary or "")
    if not positives:
        positives = "product progress, customer interest, or partner traction"
    if not risks:
        risks = "execution, integration, or regulatory attention"
    p4 = f"{_choose(seed_val, pos_leads)} {positives}. {_choose(seed_val, risk_leads, 2)} {risks}."

    # Bottom line (separate short paragraph)
    p5 = f"Bottom line: {label}."

    # Build 3–5 short paragraphs. If catalysts are thin, drop to 3–4.
    paras = [p for p in [p1, p2 if p2 else None, p3, p4, p5] if p]
    # Ensure 3–5
    if len(paras) < 3:
        paras = [p for p in [p1, p3, p5] if p]
    return "\n\n".join(paras)

