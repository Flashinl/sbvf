from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import requests

@dataclass
class NewsItem:
    title: str
    url: str
    published_at: str | None
    source: str | None
    sentiment: float  # -1..1 simple heuristic

_POSITIVE = {
    "beat", "beats", "surge", "soar", "soars", "upgrade", "upgrades",
    "raise", "raises", "record", "outperform", "strong", "growth",
}
_NEGATIVE = {
    "miss", "misses", "fall", "falls", "drop", "drops", "downgrade", "downgrades",
    "cut", "cuts", "weak", "slowdown", "slump", "lawsuit",
}


def _sentiment_score(text: str | None) -> float:
    if not text:
        return 0.0
    t = text.lower()
    pos = sum(1 for w in _POSITIVE if w in t)
    neg = sum(1 for w in _NEGATIVE if w in t)
    if pos == 0 and neg == 0:
        return 0.0
    return (pos - neg) / max(1, (pos + neg))  # normalize to [-1,1]


def fetch_news_newsapi(
    ticker: str,
    api_key: Optional[str],
    max_items: int = 10,
    company_name: Optional[str] = None,
    industry: Optional[str] = None,
) -> List[NewsItem]:
    """Fetch recent news via NewsAPI, tuned for finance relevance.
    If api_key is None/missing, returns an empty list.
    """
    if not api_key:
        return []

    # Build a finance-focused query to avoid "indie" noise for INDI, etc.
    name_terms = []
    if company_name:
        base = company_name.strip()
        if base:
            name_terms.append(base)
            # Remove common suffixes for an alternate form
            alt = base.replace(", Inc.", "").replace(" Inc", "").replace(", Ltd.", "").replace(" Ltd", "")
            if alt and alt != base:
                name_terms.append(alt)
    must_any = [ticker] + name_terms
    finance_terms = [
        "stock", "shares", "earnings", "guidance", "investor", "Nasdaq", "NYSE",
        "contract", "partnership", "order", "customer", "acquisition", "merger",
        "approval", "deal", "design win", "backlog", "revenue", "forecast",
    ]
    industry_hint = (industry or "").lower()
    if industry_hint:
        finance_terms.append(industry_hint)

    # Compose query: (name OR ticker) AND (finance terms)
    left = " OR ".join([f'"{t}"' for t in must_any if t])
    right = " OR ".join([f'"{t}"' for t in finance_terms])
    query = f"({left}) AND ({right})"

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "searchIn": "title",  # Title-only to reduce noise
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": max(1, min(int(max_items or 0), 25)),
        "apiKey": api_key,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "ok":
            return []

        # Post-filter out obviously non-financial content
        ban_tokens = {
            "film", "movie", "trailer", "festival", "oscars", "wrestle", "aew",
            "music", "album", "song", "concert", "gaming", "xbox", "playstation",
        }
        must_tokens = {ticker.lower()} | {t.lower() for t in name_terms}
        keep: List[NewsItem] = []
        for a in data.get("articles", [])[: params["pageSize"]]:
            title = (a.get("title") or "").strip()
            desc = a.get("description") or ""
            t_low = title.lower()
            if any(bt in t_low for bt in ban_tokens):
                continue
            if must_tokens and not any(mt in t_low for mt in must_tokens):
                # Allow if title contains industry hint like "semiconductor"
                if industry_hint and industry_hint not in t_low:
                    continue
            keep.append(
                NewsItem(
                    title=title,
                    url=a.get("url") or "",
                    published_at=a.get("publishedAt"),
                    source=(a.get("source") or {}).get("name"),
                    sentiment=float(_sentiment_score(title + " " + desc)),
                )
            )
        return keep
    except Exception:
        # Degrade gracefully on any error
        return []
