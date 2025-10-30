from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Global Agents Last Decision", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/")
def last_decision():
    # Placeholder for KV-backed state; compatibility only
    return {"last_decision": None}


@app.get("/healthz")
def healthz():
    return {"ok": True}