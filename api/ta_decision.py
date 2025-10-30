from fastapi import FastAPI
from fastapi.responses import JSONResponse
import yfinance as yf, ta, pandas as pd

app = FastAPI()

@app.get("/api/ta_decision")
def simple_decision():
    try:
        data = yf.download("ETH-USD", period="5d", interval="1h")
        data["rsi"] = ta.momentum.RSIIndicator(data["Close"]).rsi()
        rsi = float(data["rsi"].iloc[-1])
        signal = "BUY" if rsi < 30 else "SELL" if rsi > 70 else "HOLD"
        return JSONResponse({"ok": True, "rsi": rsi, "signal": signal})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)