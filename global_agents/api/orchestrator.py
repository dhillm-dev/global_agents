from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator
from typing import Optional, Literal, Dict, Any
import time
import os
import json

router = APIRouter(prefix="/orchestrator", tags=["orchestrator"])

# -------- In-memory state (swap to Redis later if you want persistence) ----------
_STATE: Dict[str, Any] = {
    "last": None,        # stores the latest decision envelope (DecisionEnvelope)
    "last_updated": 0.0, # epoch seconds
}

# Optional Redis client, when REDIS_URL is present
_REDIS = None
try:
    REDIS_URL = os.getenv("REDIS_URL")
    if REDIS_URL:
        import redis
        _REDIS = redis.Redis.from_url(REDIS_URL)
except Exception:
    _REDIS = None

# ------------------------------ Models ------------------------------------------
Action = Literal["BUY", "SELL", "FLAT"]


class Decision(BaseModel):
    action: Action
    symbol: str
    volume: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    comment: Optional[str] = None
    tf: Optional[str] = None

    @field_validator("symbol")
    @classmethod
    def must_have_symbol(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("symbol required")
        return v.strip()


class DecisionEnvelope(BaseModel):
    ok: bool = True
    decision: Optional[Decision] = None


# ------------------------------ Helpers -----------------------------------------
def _set_last(envelope: DecisionEnvelope) -> None:
    # If Redis is configured, store there; otherwise store in memory
    if _REDIS is not None:
        try:
            _REDIS.set("orchestrator:last", json.dumps(envelope.model_dump()))
            _REDIS.set("orchestrator:last_updated", str(time.time()))
            return
        except Exception:
            # fall back to in-memory on error
            pass
    _STATE["last"] = envelope.model_dump()
    _STATE["last_updated"] = time.time()


def _get_last() -> Dict[str, Any]:
    if _REDIS is not None:
        try:
            raw = _REDIS.get("orchestrator:last")
            if raw:
                return {"last_decision": json.loads(raw)}
            else:
                return {"last_decision": None}
        except Exception:
            # fall back to in-memory on error
            pass
    return {"last_decision": _STATE["last"]}


# ------------------------------ Routes ------------------------------------------
@router.get("/last_decision")
def get_last_decision() -> Dict[str, Any]:
    """EA reads this. Returns {"last_decision": <DecisionEnvelope or null>}"""
    return _get_last()


@router.post("/last_decision")
def post_last_decision(envelope: DecisionEnvelope) -> Dict[str, Any]:
    """Let tools/workers push a decision that the EA will execute."""
    _set_last(envelope)
    return {"ok": True, "stored": _STATE["last"], "updated_at": _STATE["last_updated"]}


@router.post("/run_once")
def run_once(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Body example: {"symbol":"ETHUSD","tf":"H1"}
    Replace the stub 'fuse_agents_once' with your real orchestrator.
    """
    symbol = (payload.get("symbol") or "ETHUSD").upper()
    tf = payload.get("tf") or "H1"

    # --- TODO: plug in your real agent fusion ---
    # For now, make a harmless FLAT or small BUY for proof-of-pipe.
    decision = Decision(action="BUY", symbol=symbol, volume=0.02, tf=tf, comment="stub")

    env = DecisionEnvelope(ok=True, decision=decision)
    _set_last(env)
    return {"ok": True, "posted": _STATE["last"]}