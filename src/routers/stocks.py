from fastapi import APIRouter, Query
from typing import Optional
import math

router = APIRouter()


def _deterministic_01(key: str) -> float:
    h = abs(hash(key)) % 1000
    return h / 1000.0  # [0,1]


@router.get("/stocks/smartfinder")
def stocks_smartfinder(symbol: str = Query(...), timeframe: str = Query("1h"), price: Optional[float] = None):
    # Price filter 1 <= p <= 45; simulate price if not provided
    if price is None:
        price = (abs(hash(symbol)) % 10000) / 100.0  # [0,100)

    if not (1.0 <= float(price) <= 45.0):
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "signal": 0.0,
            "confidence": 0.2,
            "rationale": "price outside SSF range (1-45)",
        }

    # Placeholder inputs
    mom = _deterministic_01(symbol + timeframe + ":mom")
    fundamental = _deterministic_01(symbol + ":fund")
    sentiment = _deterministic_01(symbol + timeframe + ":sent")

    # Score s in [0,1]
    s = 0.5 * mom + 0.3 * fundamental + 0.2 * sentiment
    s = max(0.0, min(1.0, s))

    # Map s -> signal in [-1,1]
    signal = 2.0 * s - 1.0

    # Confidence = 0.7 + 0.3 * fundamental z-score clip [0,1]
    z = max(0.0, min(1.0, fundamental))
    confidence = max(0.0, min(1.0, 0.7 + 0.3 * z))

    rationale = "SSF placeholder: mom/fund/sent blended; price within range"
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "price": price,
        "signal": signal,
        "confidence": confidence,
        "rationale": rationale,
    }