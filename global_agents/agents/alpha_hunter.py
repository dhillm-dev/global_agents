import math
from typing import Dict, Any

import numpy as np
import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands


def _seed_for(symbol: str, tf: str) -> int:
    """Create a deterministic seed so signals are stable per symbol/timeframe."""
    return abs(hash(f"{symbol}:{tf}")) % (2**31)


def generate_ohlcv(symbol: str, tf: str, length: int = 300) -> pd.DataFrame:
    """Generate a synthetic but plausible OHLCV series for TA evaluation.

    This avoids external data dependencies while producing consistent signals.
    """
    rng = np.random.default_rng(_seed_for(symbol, tf))
    # Random walk around a base price inferred from symbol hash
    base = 50.0 + (abs(hash(symbol)) % 5000) / 100.0  # [50, 100]
    steps = rng.normal(loc=0.0, scale=0.4, size=length)
    close = base + np.cumsum(steps)
    # Derive OHLC from close with small ranges
    spread = rng.uniform(0.05, 0.6, size=length)
    high = close + spread
    low = close - spread
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = rng.integers(1_000, 20_000, size=length)

    idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=length, freq="1H")
    return pd.DataFrame({
        "timestamp": idx,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def compute_signal(symbol: str, tf: str) -> Dict[str, Any]:
    """Compute a directional signal and confidence using TA indicators.

    - Momentum via EMA(12/26) crossover
    - Mean-reversion via Bollinger Bands touches
    - RSI regime filter to avoid chasing extremes
    """
    df = generate_ohlcv(symbol, tf, length=300)
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Trend/Momentum
    ema_fast = EMAIndicator(close=close, window=12).ema_indicator()
    ema_slow = EMAIndicator(close=close, window=26).ema_indicator()
    ema_diff = ema_fast - ema_slow

    # RSI
    rsi = RSIIndicator(close=close, window=14).rsi()

    # Bollinger Bands
    bb = BollingerBands(close=close, window=20, window_dev=2.0)
    bb_up = bb.bollinger_hband()
    bb_mid = bb.bollinger_mavg()
    bb_low = bb.bollinger_lband()

    # Normalize momentum signal by local volatility proxy
    vol_proxy = (bb_up - bb_low).rolling(10).mean()
    vol_proxy = vol_proxy.replace(0, np.nan).bfill().ffill()
    mom_raw = ema_diff / vol_proxy

    # Squash to [-1, 1]
    mom_sig = np.tanh(mom_raw).iloc[-1]

    # Mean-reversion component
    price = close.iloc[-1]
    rsi_last = rsi.iloc[-1]
    bb_up_last = bb_up.iloc[-1]
    bb_low_last = bb_low.iloc[-1]

    mr_sig = 0.0
    rationale_parts = []
    if price < bb_low_last and rsi_last < 30:
        mr_sig = +0.6
        rationale_parts.append("MR: below lower band with RSI<30 → buy bias")
    elif price > bb_up_last and rsi_last > 70:
        mr_sig = -0.6
        rationale_parts.append("MR: above upper band with RSI>70 → sell bias")
    else:
        rationale_parts.append("MR: neutral (inside bands or RSI mid)")

    # Blend momentum and mean-reversion
    signal = 0.6 * float(mom_sig) + 0.4 * float(mr_sig)
    signal = max(-1.0, min(1.0, signal))

    # Confidence from trend strength and distance from band mid
    dist_mid = abs(price - float(bb_mid.iloc[-1]))
    band_width = float((bb_up_last - bb_low_last) or 1.0)
    pos_trend = 1.0 if mom_sig > 0 else 0.0
    conf = 0.5 * (dist_mid / max(band_width, 1e-6)) + 0.5 * pos_trend
    conf = max(0.1, min(1.0, conf))

    # Directional summary
    if signal >= 0.25:
        direction = "BUY"
    elif signal <= -0.25:
        direction = "SELL"
    else:
        direction = "FLAT"

    rationale_parts.append(
        f"Momentum tanh={mom_sig:.2f}; RSI={rsi_last:.1f}; bands width={band_width:.2f}"
    )

    return {
        "symbol": symbol,
        "tf": tf,
        "signal": signal,
        "confidence": conf,
        "direction": direction,
        "indicators": {
            "ema_fast": float(ema_fast.iloc[-1]),
            "ema_slow": float(ema_slow.iloc[-1]),
            "rsi": float(rsi_last),
            "bb_upper": float(bb_up_last),
            "bb_middle": float(bb_mid.iloc[-1]),
            "bb_lower": float(bb_low_last),
            "price": float(price),
        },
        "rationale": "; ".join(rationale_parts),
    }