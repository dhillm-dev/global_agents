from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict
from global_agents.ta.fusion import ta_decide, Decision as TADecision

app = FastAPI()

_LAST_DECISION: Optional[Dict] = None


class RunOnceReq(BaseModel):
    symbol: str
    tf: str = "H1"


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/orchestrator/last_decision")
def get_last_decision():
    return {"last_decision": _LAST_DECISION}


@app.post("/orchestrator/last_decision")
def set_last_decision(payload: Dict):
    global _LAST_DECISION
    _LAST_DECISION = payload.get("decision")
    return {"ok": True, "last_decision": _LAST_DECISION}


@app.post("/orchestrator/run_once")
def run_once(req: RunOnceReq):
    """
    Compute TA-based decision for (symbol, tf), store it, and return it.
    """
    global _LAST_DECISION
    d: TADecision = ta_decide(req.symbol, req.tf)

    if d.action == "NONE":
        _LAST_DECISION = None
        return {"ok": True, "last_decision": None, "note": "flat/ambiguous"}

    _LAST_DECISION = {
        "action": d.action,
        "symbol": d.symbol,
        "volume": d.volume,
        "sl": d.sl,
        "tp": d.tp,
        "meta": d.meta,
    }
    return {"ok": True, "last_decision": _LAST_DECISION}