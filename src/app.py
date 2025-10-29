from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from src.routers import health, flow
from src.routers import alpha, stock, correl, stocks, orchestrator

app = FastAPI()
app.include_router(health.router)
app.include_router(flow.router)
app.include_router(alpha.router)
app.include_router(stock.router)
app.include_router(correl.router)
app.include_router(stocks.router)
app.include_router(orchestrator.router)

# Serve static dashboard and assets
app.mount("/static", StaticFiles(directory="static"), name="static")