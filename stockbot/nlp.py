from __future__ import annotations
from typing import List, Dict, Any
import re
from urllib.parse import urlparse

import trafilatura
import spacy
import pytextrank

from .providers.news_newsapi import NewsItem
from .config import Settings
from .semantics import embed_texts, cluster_texts, summarize_themes

_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            # Fallback to blank pipeline if model is not available
            nlp = spacy.blank("en")
        try:
            nlp.add_pipe("textrank")
        except Exception:
            pass
        _nlp = nlp
    return _nlp


def _fetch_text(url: str) -> str:
    try:
        raw = trafilatura.fetch_url(url)
        return trafilatura.extract(raw) or ""
    except Exception:
        return ""


def _event_signals(text: str) -> Dict[str, bool]:
    t = text.lower()
    keys = {
        "lease": any(k in t for k in ["lease", "tenant", "tenancy", "leased", "lease expansion", "coreweave"]),
        "contract": any(k in t for k in ["contract", "order", "award", "deal", "agreement", "customer win", "partnership", "sponsorship"]),
        "capacity": any(k in t for k in ["capacity", "utilization", "buildout", "deployment", "ramp", "ramping", "expansion"]),
        "recognition": any(k in t for k in ["revenue recognition", "recognized", "recognition", "noi", "booked revenue"]),
        "guidance": any(k in t for k in ["guidance", "outlook", "raises", "cuts", "beat", "miss"]),
    }
    return keys


def analyze_articles(news: List[NewsItem], max_articles: int = 6) -> Dict[str, Any]:
    # Prefer non-aggregator domains first for diversity
    aggregator_domains = {"biztoc.com", "finance.yahoo.com", "markets.businessinsider.com", "investing.com"}
    selected_urls: List[str] = []
    seen_domains = set()
    # First pass: non-aggregators
    for n in (news or []):
        u = getattr(n, "url", None)
        if not u:
            continue
        try:
            d = urlparse(u).netloc.lower()
        except Exception:
            d = None
        if not d or d in seen_domains or any(ad in d for ad in aggregator_domains):
            continue
        seen_domains.add(d)
        selected_urls.append(u)
        if len(selected_urls) >= max_articles:
            break
    # Second pass: fill with remaining (aggregators allowed)
    if len(selected_urls) < max_articles:
        for n in (news or []):
            u = getattr(n, "url", None)
            if not u:
                continue
            try:
                d = urlparse(u).netloc.lower()
            except Exception:
                d = None
            if not d or d in seen_domains:
                continue
            seen_domains.add(d)
            selected_urls.append(u)
            if len(selected_urls) >= max_articles:
                break

    texts = []
    for u in selected_urls:
        tx = _fetch_text(u)
        if tx:
            texts.append((u, tx))
    if not texts:
        return {}

    nlp = _get_nlp()

    drivers: List[str] = []
    watch: List[str] = []
    timing: List[str] = []
    facts: List[str] = []
    found = {"lease": False, "contract": False, "capacity": False, "recognition": False, "guidance": False}

    action_words = {"sign", "signed", "announce", "announced", "award", "awarded", "secure", "secured", "partner", "partnership", "order", "deal", "agreement", "expand", "expanded", "launch", "launched"}

    for url, text in texts:
        doc = nlp(text)
        sig = _event_signals(text)
        for k, v in sig.items():
            if v:
                found[k] = True
        # Extract top-ranked sentences for drivers, skipping number-heavy lines
        try:
            if hasattr(doc._, "textrank_paragraphs"):
                for p in doc._.textrank.paragraphs[:2]:
                    sent = re.sub(r"\s+", " ", p.text.strip())
                    if len(sent) <= 40:
                        continue
                    digit_ratio = sum(ch.isdigit() for ch in sent) / max(1, len(sent))
                    if digit_ratio > 0.3 or sent.count('%') >= 2:
                        continue
                    drivers.append(sent)
        except Exception:
            pass
        # Extract simple facts: sentences with org + action word, trimmed and de-numerized
        try:
            for s in getattr(doc, 'sents', []):
                txt = re.sub(r"\s+", " ", s.text.strip())
                if len(txt) < 40:
                    continue
                low = txt.lower()
                if not any(w in low for w in action_words):
                    continue
                # require an ORG mention
                has_org = any(ent.label_ == 'ORG' for ent in getattr(s, 'ents', []) or [])
                if not has_org:
                    continue
                digit_ratio = sum(ch.isdigit() for ch in txt) / max(1, len(txt))
                if digit_ratio > 0.25 or txt.count('%') >= 2:
                    continue
                facts.append(txt)
        except Exception:
            pass

    if found["lease"]:
        watch.append("new tenant/lease announcements")
        timing.append("moves in days–weeks on lease news")
    if found["contract"]:
        watch.append("new customer/contract announcements")
        if "moves in days–weeks on lease news" not in timing:
            timing.append("moves in days–weeks on contract news")
    if found["capacity"] or found["recognition"]:
        watch.append("pace of capacity ramp and revenue recognition")
        timing.append("3–12 months as capacity is utilized and revenue recognized")
    if found["guidance"]:
        if "quarterly results" not in watch:
            watch.append("quarterly results and guidance updates")

    # Deduplicate and cap
    drivers = list(dict.fromkeys(drivers))[:3]
    facts = list(dict.fromkeys(facts))[:8]

    # Optional semantic themes via embeddings
    themes: List[str] = []
    try:
        st = Settings.load()
        if getattr(st, "embeddings_enabled", False) and getattr(st, "hf_api_token", None):
            candidates = drivers + facts
            if len(candidates) >= 3:
                embs = embed_texts(candidates, st)
                if embs:
                    k = 3 if len(candidates) >= 6 else 2
                    clusters = cluster_texts(candidates, embs, k)
                    themes = summarize_themes(candidates, clusters, embs, max_themes=3)
    except Exception:
        themes = []

    return {
        "drivers": drivers,
        "watch": list(dict.fromkeys(watch))[:4],
        "timing": list(dict.fromkeys(timing))[:3],
        "facts": facts,
        "themes": themes,
    }

