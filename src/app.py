from fastapi import FastAPI
from src.routers import health, flow

app = FastAPI()
app.include_router(health.router)
app.include_router(flow.router)