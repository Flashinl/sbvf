from __future__ import annotations
import json
import time
import re
from typing import List, Dict, Optional
from urllib import request, error

from .config import Settings

SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no}/{doc}"

_ticker_map: Dict[str, Dict] | None = None
_last_map_fetch = 0.0


def _ua(settings: Settings) -> str:
    ua = getattr(settings, "sec_user_agent", None) or "StockBotVF/1.0 (contact@example.com)"
    return ua


def _get(url: str, settings: Settings, timeout: int = 20) -> Optional[str]:
    req = request.Request(url, headers={"User-Agent": _ua(settings)})
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except error.HTTPError as e:
        if e.code in (429, 503):
            time.sleep(1.0)
            try:
                with request.urlopen(req, timeout=timeout) as resp:
                    return resp.read().decode("utf-8", errors="ignore")
            except Exception:
                return None
        return None
    except Exception:
        return None


def _ensure_ticker_map(settings: Settings) -> Dict[str, Dict]:
    global _ticker_map, _last_map_fetch
    now = time.time()
    if _ticker_map is not None and now - _last_map_fetch < 86400:
        return _ticker_map
    raw = _get(SEC_TICKER_MAP_URL, settings)
    if not raw:
        return _ticker_map or {}
    try:
        data = json.loads(raw)
        # data is list-like with {ticker, cik_str, title}
        mp = {}
        for entry in (data if isinstance(data, list) else data.values()):
            tkr = str(entry.get("ticker", "")).upper()
            if not tkr:
                continue
            mp[tkr] = {"cik": int(entry.get("cik_str")), "title": entry.get("title")}
        _ticker_map = mp
        _last_map_fetch = now
        return _ticker_map
    except Exception:
        return _ticker_map or {}


def get_cik_for_ticker(ticker: str, settings: Settings) -> Optional[int]:
    mp = _ensure_ticker_map(settings)
    return int(mp.get(ticker.upper(), {}).get("cik")) if ticker and ticker.upper() in mp else None


def fetch_recent_filings(ticker: str, settings: Settings, forms: List[str] = ["8-K", "10-Q"], limit: int = 3) -> List[Dict]:
    cik = get_cik_for_ticker(ticker, settings)
    if not cik:
        return []
    url = SEC_SUBMISSIONS_URL.format(cik=str(cik).zfill(10))
    raw = _get(url, settings)
    if not raw:
        return []
    try:
        data = json.loads(raw)
        rec = data.get("filings", {}).get("recent", {})
        out = []
        for i, form in enumerate(rec.get("form", [])):
            if form not in forms:
                continue
            acc = rec.get("accessionNumber", [None])[i]
            doc = rec.get("primaryDocument", [None])[i]
            if not acc or not doc:
                continue
            acc_no = str(acc).replace("-", "")
            out.append({
                "form": form,
                "url": SEC_ARCHIVES_BASE.format(cik=str(cik), acc_no=acc_no, doc=doc)
            })
            if len(out) >= limit:
                break
        return out
    except Exception:
        return []


def extract_key_items(text: str) -> List[str]:
    if not text:
        return []
    t = re.sub(r"\s+", " ", text)
    # Look for sections like "Item 1.01", "Item 2.02", etc., and capture sentences around them
    items = []
    for m in re.finditer(r"Item\s+(1\.01|1\.02|2\.01|2\.02|2\.03|8\.01)", t, flags=re.IGNORECASE):
        start = max(0, m.start() - 300)
        end = min(len(t), m.end() + 600)
        snippet = t[start:end]
        if len(snippet) > 60:
            items.append(snippet)
    # If nothing matched, fallback to first 2-3 long sentences
    if not items:
        parts = re.split(r"(?<=[\.!?])\s+", t)
        items = [p for p in parts if len(p) > 120][:3]
    # Clean up
    out = []
    for s in items:
        s = re.sub(r"\s+", " ", s).strip()
        # Strip excessive numbers
        s = re.sub(r"\b\d{2,}[\w\.%$]*", "", s)
        if len(s) > 60:
            out.append(s)
    return out[:5]


def fetch_sec_facts(ticker: str, settings: Settings) -> List[str]:
    if not getattr(settings, "sec_enabled", True):
        return []
    filings = fetch_recent_filings(ticker, settings)
    facts: List[str] = []
    for f in filings:
        raw = _get(f["url"], settings, timeout=25)
        if not raw:
            continue
        # The SEC doc may be HTML; try to strip tags very roughly
        text = re.sub(r"<[^>]+>", " ", raw)
        text = re.sub(r"\s+", " ", text)
        facts.extend(extract_key_items(text))
        if len(facts) >= 8:
            break
    # Deduplicate
    seen = set(); dedup = []
    for s in facts:
        k = s.lower()
        if k not in seen:
            seen.add(k); dedup.append(s)
    return dedup[:8]

