from typing import Optional

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ._core import build_decision


app = FastAPI(title="Global Agents TA Decision", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class DecisionRequest(BaseModel):
    symbol: str
    tf: str = "H1"
    bars: int = 300


@app.options("/")
def opts():
    return Response(status_code=204)


@app.post("/")
def ta_decision(req: DecisionRequest):
    try:
        decision = build_decision(req.symbol, req.tf, req.bars)
        return {"ok": True, "decision": decision}
    except Exception as e:
        # Keep errors terse to meet serverless constraints
        return {"ok": False, "error": str(e)[:200], "decision": None}


@app.get("/healthz")
def healthz():
    return {"ok": True}