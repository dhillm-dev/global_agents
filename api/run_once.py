from typing import Optional

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ._core import build_decision


app = FastAPI(title="Global Agents Run Once", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    symbol: str = "ETHUSD"
    tf: str = "H1"
    bars: int = 300


@app.options("/")
def opts():
    return Response(status_code=204)


@app.post("/")
def run_once(req: RunRequest):
    try:
        decision = build_decision(req.symbol, req.tf, req.bars)
        return {"ok": True, "decision": decision}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200], "decision": None}


@app.get("/")
def run_once_get(symbol: Optional[str] = "ETHUSD", tf: Optional[str] = "H1", bars: Optional[int] = 300):
    try:
        decision = build_decision(symbol or "ETHUSD", tf or "H1", bars or 300)
        return {"ok": True, "decision": decision}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200], "decision": None}


@app.get("/healthz")
def healthz():
    return {"ok": True}