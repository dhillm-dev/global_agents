from typing import Optional, Dict, List, Tuple

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import math
import numpy as np
import pandas as pd
import yfinance as yf
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

UNIVERSE_MAP: Dict[str, str] = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
    "SOLUSD": "SOL-USD",
    "XAUUSD": "XAUUSD=X",
    "WTI": "CL=F",
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
    if s in UNIVERSE_MAP:
        return UNIVERSE_MAP[s]
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
    if interval in ("15m", "30m"):
        return "30d"
    if interval in ("1h", "4h"):
        return "90d"
    return "365d"

def fetch_ohlc(ticker: str, interval: str, bars: int) -> pd.DataFrame:
    period = period_for_interval(interval, bars)
    df = yf.download(tickers=ticker, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
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
    confidence = float(max(0.0, min(1.0, confidence / 3.0)))
    return action, confidence

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
    atr_v = float(ind["atr"].iloc[-1])
    price = float(df["Close"].iloc[-1])
    vol_scale = max(0.01, min(0.05, atr_v / (price + 1e-9)))
    volume = f"{vol_scale:.2f}"
    return {
        "action": action,
        "symbol": symbol.upper(),
        "volume": volume,
        "confidence": round(float(confidence), 4),
        "corr_bias": {},
        "meta": {
            "rsi": round(float(ind["rsi"].iloc[-1]), 4),
            "macd": round(float(ind["macd"].iloc[-1]), 8),
            "atr": round(atr_v, 6),
        },
    }


app = FastAPI(title="Global Agents Run Once", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    symbol: str = "ETHUSD"
    tf: str = "H1"
    bars: int = 300


@app.options("/")
def opts():
    return Response(status_code=204)


@app.post("/")
def run_once(req: RunRequest):
    try:
        decision = build_decision(req.symbol, req.tf, req.bars)
        return {"ok": True, "decision": decision}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200], "decision": None}


@app.get("/")
def run_once_get(symbol: Optional[str] = "ETHUSD", tf: Optional[str] = "H1", bars: Optional[int] = 300):
    try:
        decision = build_decision(symbol or "ETHUSD", tf or "H1", bars or 300)
        return {"ok": True, "decision": decision}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200], "decision": None}


@app.get("/healthz")
def healthz():
    return {"ok": True}