from fastapi import FastAPI
from global_agents.api.routers import orchestrator

app = FastAPI(title="global_agents")
app.include_router(orchestrator.router)


@app.get("/healthz")
def healthz():
    return {"ok": True}