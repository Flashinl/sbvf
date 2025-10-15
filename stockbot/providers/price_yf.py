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

    return PriceSnapshot(
        ticker=ticker,
        price=info.get("currentPrice") or info.get("regularMarketPrice"),
        currency=info.get("currency"),
        market_cap=info.get("marketCap"),
        trailing_pe=info.get("trailingPE"),
        eps_ttm=info.get("trailingEps"),
        dividend_yield=info.get("dividendYield"),
        fifty_two_week_high=info.get("fiftyTwoWeekHigh"),
        fifty_two_week_low=info.get("fiftyTwoWeekLow"),
        earnings_date=earnings_date,
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

