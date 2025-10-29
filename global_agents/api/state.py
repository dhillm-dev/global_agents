from typing import Optional, Dict, Any

_last_decision: Optional[Dict[str, Any]] = None


def set_last_decision(decision: Optional[Dict[str, Any]]):
    global _last_decision
    _last_decision = decision


def get_last_decision() -> Dict[str, Any]:
    return {"last_decision": _last_decision}