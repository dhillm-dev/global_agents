from fastapi import APIRouter, Query
from typing import List

router = APIRouter()


@router.get("/correl/score")
def correl_score(symbol: str = Query(...), timeframe: str = Query("1h")):
    # Placeholder peers list per FX/Equity/Crypto regimes
    peers: List[str] = ["EURUSD", "GBPUSD", "DXY", "SPY", "BTCUSDT"]
    h = hash(symbol + timeframe)
    raw = ((h % 1000) - 500) / 500.0  # [-1, 1]
    score = max(-1.0, min(1.0, raw))
    signal = score  # In Trae: map correlation score → signal = (score × 2 − 1); here score is already [-1,1]
    confidence = 0.35
    return {
        "ts": None,
        "symbol": symbol,
        "timeframe": timeframe,
        "peers": peers,
        "score": score,
        "signal": signal,
        "confidence": confidence,
        "rationale": "correlation placeholder",
    }