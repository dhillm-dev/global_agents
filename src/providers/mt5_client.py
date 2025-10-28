import os

ENABLE_MT5 = os.getenv("ENABLE_MT5", "0") == "1"
_mt5 = None

def try_import_mt5():
    global _mt5
    if _mt5 is not None:
        return _mt5
    if not ENABLE_MT5:
        return None
    try:
        import MetaTrader5 as _real
        _mt5 = _real
    except Exception:
        _mt5 = None
    return _mt5