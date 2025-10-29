import os
import time
import requests

API = os.getenv("API_BASE", "https://global-agents-jciy.onrender.com")
UNIVERSE = os.getenv("UNIVERSE", "ETHUSD,EURUSD,GBPUSD,XAUUSD").split(",")
TF = os.getenv("TF", "H1")
COOLDOWN = int(os.getenv("COOLDOWN", "15"))
TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "20"))


def run_once(sym: str):
    r = requests.post(
        f"{API}/orchestrator/run_once",
        json={"symbol": sym, "tf": TF},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    print("posted decision for", sym, r.json())


if __name__ == "__main__":
    while True:
        for s in UNIVERSE:
            try:
                run_once(s.strip())
            except Exception as e:
                print("run_once error:", s, e)
        time.sleep(COOLDOWN)