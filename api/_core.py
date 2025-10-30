import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange


# Supported universe mapped to Yahoo Finance tickers
UNIVERSE_MAP: Dict[str, str] = {
    # FX
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    # Crypto
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
    "SOLUSD": "SOL-USD",
    # Commodities
    "XAUUSD": "XAUUSD=X",  # Gold quoted vs USD
    "WTI": "CL=F",
    # Indices
    "SPX": "^GSPC",
    "NASDAQ": "^IXIC",
    "DAX": "^GDAXI",
}


def friendly_alias(yf_ticker: str) -> str:
    alias_map = {
        "^GSPC": "SPX",
        "^IXIC": "NASDAQ",
        "^GDAXI": "DAX",
        "BTC-USD": "BTCUSD",
        "ETH-USD": "ETHUSD",
        "SOL-USD": "SOLUSD",
        "CL=F": "WTI",
        "EURUSD=X": "EURUSD",
        "GBPUSD=X": "GBPUSD",
        "USDJPY=X": "USDJPY",
        "AUDUSD=X": "AUDUSD",
        "USDCAD=X": "USDCAD",
        "XAUUSD=X": "XAUUSD",
    }
    return alias_map.get(yf_ticker, yf_ticker)


def yf_symbol_for(symbol: str) -> str:
    s = symbol.upper().strip()
    # Accept already-YF symbols
    if s in UNIVERSE_MAP:
        return UNIVERSE_MAP[s]
    # Common synonyms
    synonyms = {
        "BTC-USD": "BTC-USD",
        "ETH-USD": "ETH-USD",
        "SOL-USD": "SOL-USD",
        "SPX": "^GSPC",
        "NAS100": "^IXIC",
        "NDQ": "^IXIC",
        "DAX": "^GDAXI",
    }
    if s in synonyms:
        return synonyms[s]
    # Fallback: try FX mapping for XXXUSD → XXXUSD=X
    if len(s) == 6 and s.endswith("USD"):
        return f"{s}=X"
    return s


def normalize_tf(tf: str) -> str:
    tf = tf.upper().strip()
    mapping = {
        "M15": "15m",
        "M30": "30m",
        "H1": "1h",
        "H4": "4h",
        "D1": "1d",
        "15M": "15m",
        "30M": "30m",
        "1H": "1h",
        "4H": "4h",
        "1D": "1d",
        "15M": "15m",
        "30M": "30m",
    }
    return mapping.get(tf, tf.lower())


def period_for_interval(interval: str, bars: int) -> str:
    # Conservative periods to keep function under 10s and within Yahoo constraints
    if interval in ("15m", "30m"):
        return "30d"
    if interval in ("1h", "4h"):
        return "90d"
    return "365d"


def fetch_ohlc(ticker: str, interval: str, bars: int) -> pd.DataFrame:
    period = period_for_interval(interval, bars)
    df = yf.download(tickers=ticker, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        # Single ticker sometimes returns multiindex; select first level
        df = df.xs(ticker, axis=1, level=1)
    df = df.dropna()
    if len(df) > bars:
        df = df.tail(bars)
    return df


def compute_indicators(df: pd.DataFrame) -> Dict[str, pd.Series]:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    ema9 = EMAIndicator(close=close, window=9).ema_indicator()
    ema21 = EMAIndicator(close=close, window=21).ema_indicator()
    rsi14 = RSIIndicator(close=close, window=14).rsi()
    macd = MACD(close=close, window_fast=12, window_slow=26, window_sign=9)
    macd_line = macd.macd()
    atr14 = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()

    return {
        "ema_fast": ema9,
        "ema_slow": ema21,
        "rsi": rsi14,
        "macd": macd_line,
        "atr": atr14,
    }


def zscore(series: pd.Series, window: int = 30) -> pd.Series:
    w = min(window, len(series))
    roll = series.tail(w)
    mu = float(roll.mean())
    sigma = float(roll.std()) or 1e-9
    return (roll - mu) / sigma


def fuse_signal(ind: Dict[str, pd.Series]) -> Tuple[str, float]:
    fast = ind["ema_fast"].iloc[-1]
    slow = ind["ema_slow"].iloc[-1]
    rsi_v = float(ind["rsi"].iloc[-1])
    macd_v = float(ind["macd"].iloc[-1])

    action = "NONE"
    if fast > slow and rsi_v < 70 and macd_v > 0:
        action = "BUY"
    elif fast < slow and rsi_v > 30 and macd_v < 0:
        action = "SELL"

    rsi_z = float(zscore(ind["rsi"]).iloc[-1])
    macd_z = float(zscore(ind["macd"]).iloc[-1])
    confidence = float(np.mean([abs(rsi_z), abs(macd_z)]))
    # Clamp to [0, 1.0] for user-friendly score
    confidence = float(max(0.0, min(1.0, confidence / 3.0)))
    return action, confidence


def compute_corr_bias(target_ticker: str, interval: str, bars: int) -> Dict[str, float]:
    tickers = list(set(UNIVERSE_MAP.values()))
    # Ensure target included
    if target_ticker not in tickers:
        tickers.append(target_ticker)

    period = period_for_interval(interval, bars)
    data = yf.download(tickers=tickers, period=period, interval=interval, progress=False)
    # Use Close prices for correlation and simple signals
    closes = data["Close"].dropna(how="all")
    closes = closes.tail(bars)
    returns = closes.pct_change().dropna()
    if target_ticker not in returns.columns or returns.empty:
        return {}

    corr_to_target = returns.corr()[target_ticker].dropna()
    # Compute simple signal sign for each asset using Close-only indicators
    bias: Dict[str, float] = {}
    for col in returns.columns:
        series = closes[col].dropna()
        if len(series) < 50:
            continue
        ema_fast = EMAIndicator(close=series, window=9).ema_indicator()
        ema_slow = EMAIndicator(close=series, window=21).ema_indicator()
        rsi14 = RSIIndicator(close=series, window=14).rsi()
        macd_line = MACD(close=series, window_fast=12, window_slow=26, window_sign=9).macd()
        sign = 0
        # Same fusion rules without ATR
        if ema_fast.iloc[-1] > ema_slow.iloc[-1] and float(rsi14.iloc[-1]) < 70 and float(macd_line.iloc[-1]) > 0:
            sign = 1
        elif ema_fast.iloc[-1] < ema_slow.iloc[-1] and float(rsi14.iloc[-1]) > 30 and float(macd_line.iloc[-1]) < 0:
            sign = -1
        corr = float(corr_to_target.get(col, np.nan))
        if not math.isnan(corr):
            bias[friendly_alias(col)] = float(corr * sign)

    # Keep top influences
    if bias:
        bias = dict(sorted(bias.items(), key=lambda kv: abs(kv[1]), reverse=True)[:6])
    return bias


def build_decision(symbol: str, tf: str, bars: int = 300) -> Dict:
    interval = normalize_tf(tf)
    yf_ticker = yf_symbol_for(symbol)
    df = fetch_ohlc(yf_ticker, interval, bars)
    if df.empty:
        return {
            "action": "NONE",
            "symbol": symbol,
            "volume": "0.00",
            "confidence": 0.0,
            "corr_bias": {},
            "meta": {"rsi": None, "macd": None, "atr": None},
        }

    ind = compute_indicators(df)
    action, confidence = fuse_signal(ind)

    # Simple ATR-based sizing (bounded)
    atr_v = float(ind["atr"].iloc[-1])
    price = float(df["Close"].iloc[-1])
    vol_scale = max(0.01, min(0.05, atr_v / (price + 1e-9)))
    volume = f"{vol_scale:.2f}"

    corr_bias = compute_corr_bias(yf_ticker, interval, bars)

    return {
        "action": action,
        "symbol": symbol.upper(),
        "volume": volume,
        "confidence": round(float(confidence), 4),
        "corr_bias": corr_bias,
        "meta": {
            "rsi": round(float(ind["rsi"].iloc[-1]), 4),
            "macd": round(float(ind["macd"].iloc[-1]), 8),
            "atr": round(atr_v, 6),
        },
    }