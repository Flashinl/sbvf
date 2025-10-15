from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from ta.momentum import RSIIndicator

@dataclass
class PriceSnapshot:
    ticker: str
    price: float | None
    currency: str | None
    market_cap: float | None
    trailing_pe: float | None
    eps_ttm: float | None
    dividend_yield: float | None
    fifty_two_week_high: float | None
    fifty_two_week_low: float | None
    earnings_date: datetime | None
    sector: str | None = None
    industry: str | None = None
    long_name: str | None = None

@dataclass
class Technicals:
    sma20: float | None
    sma50: float | None
    sma200: float | None
    rsi14: float | None
    trend_score: float | None  # -1 to 1 simple scale

def fetch_price_and_fundamentals(ticker: str) -> PriceSnapshot:
    t = yf.Ticker(ticker)
    info = t.info or {}
    cal = t.calendar
    earnings_date = None
    try:
        if cal is not None and not cal.empty:
            # yfinance may return earnings dates in the index
            ed = cal.loc["Earnings Date"][0]
            if isinstance(ed, (np.datetime64, pd.Timestamp)):
                earnings_date = pd.Timestamp(ed).to_pydatetime()
    except Exception:
        pass

    # Robust current price resolution
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    if not price:
        try:
            fi = getattr(t, "fast_info", None)
            # fast_info can be dict-like or object-like depending on yfinance version
            def _fi_get(k):
                if fi is None:
                    return None
                if isinstance(fi, dict):
                    return fi.get(k)
                return getattr(fi, k, None)
            for k in ("last_price", "lastPrice", "regularMarketPrice", "lastClose", "previousClose"):
                price = _fi_get(k)
                if price:
                    break
        except Exception:
            price = None
    if not price:
        try:
            hist = t.history(period="5d", interval="1d", auto_adjust=False)
            if hist is not None and not hist.empty:
                price = float(hist["Close"].iloc[-1])
        except Exception:
            price = None

    # Fill additional fields from fast_info if missing
    market_cap = info.get("marketCap")
    fifty_two_week_high = info.get("fiftyTwoWeekHigh")
    fifty_two_week_low = info.get("fiftyTwoWeekLow")
    try:
        fi = getattr(t, "fast_info", None)
        def _fi_get(k):
            if fi is None:
                return None
            if isinstance(fi, dict):
                return fi.get(k)
            return getattr(fi, k, None)
        if market_cap is None:
            market_cap = _fi_get("market_cap") or _fi_get("marketCap")
        if fifty_two_week_high is None:
            fifty_two_week_high = _fi_get("year_high") or _fi_get("fiftyTwoWeekHigh")
        if fifty_two_week_low is None:
            fifty_two_week_low = _fi_get("year_low") or _fi_get("fiftyTwoWeekLow")
    except Exception:
        pass

    return PriceSnapshot(
        ticker=ticker,
        price=float(price) if price is not None else None,
        currency=info.get("currency"),
        market_cap=float(market_cap) if market_cap is not None else None,
        trailing_pe=info.get("trailingPE"),
        eps_ttm=info.get("trailingEps"),
        dividend_yield=info.get("dividendYield"),
        fifty_two_week_high=float(fifty_two_week_high) if fifty_two_week_high is not None else None,
        fifty_two_week_low=float(fifty_two_week_low) if fifty_two_week_low is not None else None,
        earnings_date=earnings_date,
        sector=info.get("sector"),
        industry=info.get("industry"),
        long_name=info.get("longName"),
    )

def fetch_technicals(ticker: str, period: str = "1y", interval: str = "1d") -> Technicals:
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if df is None or df.empty:
        return Technicals(None, None, None, None, None)

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.squeeze().dropna()
    sma20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else np.nan
    sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else np.nan
    sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan
    rsi14 = RSIIndicator(close, window=14).rsi().iloc[-1] if len(close) >= 15 else np.nan

    # Simple trend score heuristic
    trend_score = 0.0
    p = close.iloc[-1]
    if not np.isnan(sma50) and not np.isnan(sma200):
        if sma50 > sma200:
            trend_score += 0.5
        else:
            trend_score -= 0.5
    if not np.isnan(sma20) and p > sma20:
        trend_score += 0.25
    if not np.isnan(sma50) and p > sma50:
        trend_score += 0.25

    return Technicals(
        float(sma20) if not np.isnan(sma20) else None,
        float(sma50) if not np.isnan(sma50) else None,
        float(sma200) if not np.isnan(sma200) else None,
        float(rsi14) if not np.isnan(rsi14) else None,
        float(trend_score),
    )

