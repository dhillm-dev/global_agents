from fastapi import APIRouter, Body
from global_agents.api.state import get_last_decision, set_last_decision
from global_agents.trae.core_agent import fuse_agents_once

router = APIRouter(prefix="/orchestrator", tags=["orchestrator"])


@router.get("/last_decision")
def read_last():
    return get_last_decision()


@router.post("/last_decision")
def write_last(payload: dict = Body(...)):
    # expects {"ok":true,"decision":{...}} or {"ok":true,"decision":null}
    set_last_decision(payload.get("decision"))
    return {"ok": True}


@router.post("/run_once")
def run_once(payload: dict = Body(...)):
    symbol = payload.get("symbol", "ETHUSD")
    tf = payload.get("tf", "H1")
    decision = fuse_agents_once(symbol=symbol, tf=tf)  # returns dict or None
    set_last_decision(decision)
    return {"ok": True, "decision": decision}