from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

router = APIRouter()


class AlphaRequest(BaseModel):
    symbol: str
    timeframe: str
    context: Optional[str] = None


@router.post("/alpha/hunter")
def alpha_hunter(req: AlphaRequest):
    # Placeholder alpha logic: simple heuristic based on symbol hash
    h = hash(req.symbol + req.timeframe)
    raw = ((h % 2000) - 1000) / 1000.0  # [-1, 1]
    signal = max(-1.0, min(1.0, raw))
    confidence = 0.5
    return {
        "ts": None,
        "symbol": req.symbol,
        "timeframe": req.timeframe,
        "signal": signal,
        "confidence": confidence,
        "rationale": "alpha placeholder",
    }