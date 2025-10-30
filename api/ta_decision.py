from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

# Use root path so the function at /api/ta_decision serves this handler
@app.get("/")
def simple_decision():
    try:
        # Lazy-load heavy dependencies to avoid import-time crashes
        import yfinance as yf
        import pandas as pd
        import ta

        data = yf.download("ETH-USD", period="5d", interval="1h")
        if data.empty or "Close" not in data.columns:
            return JSONResponse({"ok": False, "error": "no data"}, status_code=502)

        data["rsi"] = ta.momentum.RSIIndicator(close=data["Close"]).rsi()
        rsi = float(data["rsi"].iloc[-1])
        signal = "BUY" if rsi < 30 else "SELL" if rsi > 70 else "HOLD"
        return JSONResponse({"ok": True, "symbol": "ETH-USD", "rsi": rsi, "signal": signal})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)