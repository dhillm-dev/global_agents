import os
import asyncio
from typing import Optional

API_URL = os.getenv("API_URL", "http://localhost:8000")

def _ensure_mt5():
    try:
        import MetaTrader5 as mt5
        return mt5
    except Exception as e:
        raise RuntimeError("MetaTrader5 is not available on this host") from e

async def main():
    mt5 = _ensure_mt5()
    # Example initialization flow (fill in credentials via .env as needed)
    if not mt5.initialize():
        raise RuntimeError("Failed to initialize MetaTrader5")
    # TODO: connect to API_URL to fetch/subscribe to signals and execute orders
    await asyncio.sleep(0.1)
    mt5.shutdown()

if __name__ == "__main__":
    asyncio.run(main())