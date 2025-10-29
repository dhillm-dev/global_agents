from fastapi import APIRouter, Query

router = APIRouter()


@router.get("/correl/score")
def correl_score(symbol: str = Query(...), timeframe: str = Query("1h")):
    # Placeholder correlation: deterministic pseudo-score
    h = hash(symbol + timeframe)
    raw = ((h % 1000) - 500) / 500.0  # [-1, 1]
    signal = max(-1.0, min(1.0, raw))
    confidence = 0.3
    return {
        "ts": None,
        "symbol": symbol,
        "timeframe": timeframe,
        "signal": signal,
        "confidence": confidence,
        "rationale": "correlation placeholder",
    }