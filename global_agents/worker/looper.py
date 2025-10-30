import os
import time
import json
import requests

API_BASE = os.getenv("API_BASE", "https://global-agents-jciy.onrender.com")
UNIVERSE = [s.strip() for s in os.getenv("UNIVERSE", "ETHUSD,EURUSD,GBPUSD,XAUUSD").split(",") if s.strip()]
TF = os.getenv("TF", "H1")
COOLDOWN = int(os.getenv("COOLDOWN", "15"))
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "20"))


def run_once(symbol: str, tf: str) -> dict:
    """
    Calls orchestrator to compute 1 decision for (symbol, tf).
    Expects the API to store the decision and/or return it.
    """
    url = f"{API_BASE}/orchestrator/run_once"
    payload = {"symbol": symbol, "tf": tf}
    r = requests.post(url, json=payload, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()


if __name__ == "__main__":
    print(f"[worker] starting with API_BASE={API_BASE} UNIVERSE={UNIVERSE} TF={TF} COOLDOWN={COOLDOWN}s")
    while True:
        for sym in UNIVERSE:
            try:
                res = run_once(sym, TF)
                print(f"[worker] {sym}/{TF} → {json.dumps(res)}")
            except Exception as e:
                print(f"[worker][ERROR] {sym}/{TF}: {e}")
            time.sleep(1)
        time.sleep(COOLDOWN)