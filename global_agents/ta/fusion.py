from __future__ import annotations
import math, time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from ta.volatility import AverageTrueRange

# Map broker-ish symbols to Yahoo tickers
YF_MAP = {
    # FX
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "XAUUSD": "GC=F",   # Comex Gold futures (proxy)
    # Crypto
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
}

# MT5 tf -> yfinance interval+period (enough bars for SMA200/ATR14)
TF_MAP = {
    "M15": ("15m", "30d"),
    "M30": ("30m", "60d"),
    "H1":  ("60m", "90d"),
    "H4":  ("240m","720d"),
    "D1":  ("1d",  "4y"),
}

@dataclass
class Decision:
    action: str          # "BUY" or "SELL" or "NONE"
    symbol: str
    volume: float        # lot size for your EA (demo small)
    sl: Optional[float]  # absolute price
    tp: Optional[float]  # absolute price
    meta: dict

def _map_symbol(symbol: str) -> str:
    # strip common suffixes then map
    base = symbol.split(".")[0].upper()
    return YF_MAP.get(base, base)

def _yf_tf(tf: str) -> Tuple[str, str]:
    tf = tf.upper()
    return TF_MAP.get(tf, TF_MAP["H1"])

def _fetch_ohlc(symbol: str, tf: str) -> pd.DataFrame:
    yf_sym = _map_symbol(symbol)
    interval, period = _yf_tf(tf)
    df = yf.download(yf_sym, interval=interval, period=period, auto_adjust=False, progress=False)
    if df.empty or len(df) < 250:
        raise RuntimeError(f"Not enough data for {symbol}/{tf}: got {len(df)} bars")
    df = df.rename(columns=str.lower)
    return df

def _indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure 1-D Series for TA lib wrappers
    def _series_1d(x: pd.Series) -> pd.Series:
        arr = np.asarray(x).reshape(-1)
        return pd.Series(arr, index=df.index)

    close = _series_1d(df["close"])  # type: ignore
    high  = _series_1d(df["high"])   # type: ignore
    low   = _series_1d(df["low"])    # type: ignore

    df["sma50"]  = SMAIndicator(close, 50).sma_indicator()
    df["sma200"] = SMAIndicator(close, 200).sma_indicator()
    df["rsi14"]  = RSIIndicator(close, 14).rsi()
    atr = AverageTrueRange(high=high, low=low, close=close, window=14)
    df["atr14"]  = atr.average_true_range()
    return df

def _position_size(symbol: str, atr: float, price: float) -> float:
    """
    Very conservative size: target ATR value risk ≈ 0.2% notional.
    Your EA uses 'volume' as lot or units – keep tiny on demo.
    """
    if atr <= 0 or price <= 0:
        return 0.02
    risk_notional = 0.002 * price
    units = max(0.01, min(0.2, risk_notional / max(1e-9, atr)))
    return round(units, 2)

def _sl_tp_from_atr(side: str, price: float, atr: float, k_sl=1.5, k_tp=2.5) -> Tuple[float, float]:
    atr = max(atr, 1e-9)
    if side == "BUY":
        sl = price - k_sl * atr
        tp = price + k_tp * atr
    else:
        sl = price + k_sl * atr
        tp = price - k_tp * atr
    return (round(sl, 5), round(tp, 5))

def ta_decide(symbol: str, tf: str) -> Decision:
    df = _fetch_ohlc(symbol, tf)
    df = _indicators(df).dropna()
    last = df.iloc[-1]

    price = float(last["close"])
    sma50, sma200 = float(last["sma50"]), float(last["sma200"])
    rsi = float(last["rsi14"])
    atr = float(last["atr14"])

    # Trend filter
    trend = "UP" if sma50 > sma200 else "DOWN"
    # Simple regime strength
    slope = (df["sma50"].iloc[-1] - df["sma50"].iloc[-5]) / 5

    # Signals
    buy_cond  = (trend == "UP"   and rsi < 55) or (rsi < 35)   # buy dips in uptrend; oversold rescue
    sell_cond = (trend == "DOWN" and rsi > 45) or (rsi > 65)   # sell rips in downtrend; overbought fade

    if buy_cond and not sell_cond:
        side = "BUY"
    elif sell_cond and not buy_cond:
        side = "SELL"
    else:
        # ambiguous → no trade
        return Decision("NONE", symbol, 0.0, None, None, {
            "reason":"flat/ambiguous", "price":price, "rsi":rsi, "trend":trend, "slope":slope, "atr":atr
        })

    vol = _position_size(symbol, atr, price)
    sl, tp = _sl_tp_from_atr(side, price, atr)

    return Decision(side, symbol, vol, sl, tp, {
        "price":price, "rsi":rsi, "trend":trend, "slope":slope, "atr":atr
    })