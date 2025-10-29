from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from global_agents.api.orchestrator import router as orchestrator_router

app = FastAPI(title="Global Agents Hub")

# (optional) CORS – handy for dashboards or cross-origin tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz():
    return {"ok": True}

# orchestrator
app.include_router(orchestrator_router)