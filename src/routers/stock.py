from fastapi import APIRouter, Query

router = APIRouter()


def _is_equity(sym: str) -> bool:
    return ("/" not in sym) and (not sym.endswith("USDT"))


@router.get("/stock/scanner")
def stock_scanner(symbol: str = Query(...), timeframe: str = Query("1h")):
    # If not equity, return neutral with low confidence
    if not _is_equity(symbol):
        return {
            "ts": None,
            "symbol": symbol,
            "timeframe": timeframe,
            "signal": 0.0,
            "confidence": 0.2,
            "rationale": "non-equity symbol; scanner neutral",
        }

    # Placeholder equity scanner: weak momentum proxy
    h = hash(symbol + timeframe)
    raw = ((h % 200) - 100) / 100.0  # [-1, 1] but smaller range
    signal = max(-1.0, min(1.0, raw * 0.5))
    confidence = 0.4
    return {
        "ts": None,
        "symbol": symbol,
        "timeframe": timeframe,
        "signal": signal,
        "confidence": confidence,
        "rationale": "stock scanner placeholder",
    }