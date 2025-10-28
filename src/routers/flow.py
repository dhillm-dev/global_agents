from fastapi import APIRouter

router = APIRouter()

_latest = {"ts": None, "score": None, "spread": None, "imbalance": None}

@router.get("/flow/snapshot")
def flow_snapshot():
    return _latest