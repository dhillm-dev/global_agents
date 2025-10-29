from fastapi import APIRouter

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