from fastapi import APIRouter
import math

router = APIRouter()

_latest = {"ts": None, "score": None, "spread": None, "imbalance": None}

@router.get("/flow/snapshot")
def flow_snapshot():
    # Map to orchestrator-friendly fields
    score = _latest.get("score")
    signal = float(score) if isinstance(score, (int, float)) else 0.0
    confidence = 0.5 if isinstance(score, (int, float)) else 0.0
    return {
        **_latest,
        "signal": signal,
        "confidence": confidence,
        "rationale": "flow snapshot placeholder",
    }


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@router.get("/flow/anomaly")
def flow_anomaly(symbol: str, timeframe: str = "1h"):
    # Deterministic placeholder metrics
    z = abs(((abs(hash(symbol + timeframe + ":z")) % 500) / 100.0)) / 1.0  # approx [0,5)
    delta_spread_ratio = ((abs(hash(symbol + ":spr")) % 200) / 100.0)  # [0,2)
    imbalance_raw = ((hash(symbol + timeframe + ":imb") % 200) - 100) / 100.0  # [-1,1]

    anomaly = (z > 3.0) or (delta_spread_ratio > 1.5)
    sgm = _sigmoid(abs(z))
    direction = -1.0 if imbalance_raw > 0 else 1.0  # if ask-side spoofing (positive), expect dump → negative signal
    signal = direction * sgm
    confidence = min(1.0, 0.5 + 0.5 * sgm)

    rationale = (
        "FAD placeholder: z-score/spread-vol computed; "
        + ("anomaly detected" if anomaly else "no anomaly")
    )
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "signal": signal,
        "confidence": confidence,
        "rationale": rationale,
    }