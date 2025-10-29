from fastapi import APIRouter, Request, Query
from global_agents.trae.core_agent import fuse_agents_once
from pathlib import Path
import json
from datetime import datetime, timezone
import asyncio
from typing import Optional

router = APIRouter()


MEM_PATH = Path("global_agents/trae/_memory.json")
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/orchestrator/last_decision")
def orchestrator_last_decision():
    path = MEM_PATH
    if not path.exists():
        return {"last_decision": None}
    try:
        data = json.loads(path.read_text())
        return {"last_decision": data.get("last_decision")}
    except Exception:
        return {"last_decision": None}


@router.post("/orchestrator/last_decision")
async def orchestrator_set_last_decision(request: Request):
    """
    Compatibility alias: accept a POST to /orchestrator/last_decision.
    Stores the provided payload as the latest decision and appends to history.

    Accepts either a raw decision object or an envelope like:
      {"ok": true, "decision": { ... }}
    """
    try:
        payload = await request.json()
    except Exception:
        return {"status": "error", "message": "invalid json"}

    # If the payload wraps the decision, unwrap it
    decision = payload.get("decision") if isinstance(payload, dict) else None
    if decision is None and isinstance(payload, dict):
        decision = payload  # treat entire payload as the decision

    if not isinstance(decision, dict):
        return {"status": "error", "message": "invalid decision payload"}

    # Load memory
    mem = {"weights": {}, "history": [], "metrics": {}, "last_decision": None}
    if MEM_PATH.exists():
        try:
            mem = json.loads(MEM_PATH.read_text())
        except Exception:
            pass

    # Append to history and set last_decision
    mem.setdefault("history", [])
    mem["history"].append(decision)
    mem["history"] = mem["history"][-5000:]
    mem["last_decision"] = decision

    # Persist safely
    tmp = MEM_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(mem, indent=2))
    tmp.replace(MEM_PATH)

    # Write audit log
    try:
        line = json.dumps({
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "event": "decision_posted_alias",
            "symbol": decision.get("symbol"),
            "direction": decision.get("direction") or decision.get("action"),
            "volume": decision.get("volume"),
        })
        (LOG_DIR / "decisions.jsonl").open("a", encoding="utf-8").write(line + "\n")
    except Exception:
        pass

    return {"status": "ok"}


@router.get("/orchestrator/history")
def orchestrator_history(limit: int = Query(50, ge=1, le=500)):
    path = MEM_PATH
    if not path.exists():
        return {"history": []}
    try:
        data = json.loads(path.read_text())
        hist = data.get("history", [])
        return {"history": hist[-limit:]}
    except Exception:
        return {"history": []}


@router.post("/orchestrator/decision")
async def orchestrator_decision(request: Request):
    payload = await request.json()
    # Basic validation
    if not isinstance(payload, dict):
        return {"status": "error", "message": "invalid payload"}

    # Load memory
    mem = {"weights": {}, "history": [], "metrics": {}, "last_decision": None}
    if MEM_PATH.exists():
        try:
            mem = json.loads(MEM_PATH.read_text())
        except Exception:
            pass

    # Append to history and set last_decision
    mem.setdefault("history", [])
    mem["history"].append(payload)
    mem["history"] = mem["history"][-5000:]
    mem["last_decision"] = payload

    # Persist
    tmp = MEM_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(mem, indent=2))
    tmp.replace(MEM_PATH)

    # Also write a lightweight audit log line
    try:
        line = json.dumps({
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "event": "decision_posted",
            "symbol": payload.get("symbol"),
            "direction": payload.get("direction"),
            "conviction": payload.get("conviction"),
            "confidence": payload.get("confidence"),
        })
        (LOG_DIR / "decisions.jsonl").open("a", encoding="utf-8").write(line + "\n")
    except Exception:
        pass

    return {"status": "ok"}


@router.post("/orchestrator/run_once")
async def orchestrator_run_once(request: Request):
    """
    Run a single orchestrator cycle and persist the fused decision.
    Body may include {"symbol": "ETHUSD", "tf": "H1"} to hint the target.
    """
    try:
        payload = await request.json()
    except Exception:
        payload = {}

    hint_symbol: Optional[str] = None
    hint_tf: Optional[str] = None
    if isinstance(payload, dict):
        hint_symbol = payload.get("symbol")
        hint_tf = payload.get("tf") or payload.get("timeframe")

    # Use demo fusion function to produce a decision (or None for FLAT)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    decision = fuse_agents_once(symbol=hint_symbol or "ETHUSD", tf=hint_tf or "H1")

    # Persist into memory like the decision endpoint
    mem = {"weights": {}, "history": [], "metrics": {}, "last_decision": None}
    if MEM_PATH.exists():
        try:
            mem = json.loads(MEM_PATH.read_text())
        except Exception:
            pass
    mem.setdefault("history", [])
    if isinstance(decision, dict):
        mem["history"].append(decision)
        mem["history"] = mem["history"][-5000:]
    mem["last_decision"] = decision
    tmp = MEM_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(mem, indent=2))
    tmp.replace(MEM_PATH)

    # Log
    try:
        info = {"ts": now, "event": "run_once_posted"}
        if isinstance(decision, dict):
            info.update({
                "symbol": decision.get("symbol"),
                "action": decision.get("action"),
            })
        (LOG_DIR / "decisions.jsonl").open("a", encoding="utf-8").write(json.dumps(info) + "\n")
    except Exception:
        pass

    return {"status": "ok", "decision": decision}