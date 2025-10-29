from fastapi import APIRouter
from pathlib import Path
import json

router = APIRouter()


@router.get("/orchestrator/last_decision")
def orchestrator_last_decision():
    path = Path("global_agents/trae/_memory.json")
    if not path.exists():
        return {"last_decision": None}
    try:
        data = json.loads(path.read_text())
        return {"last_decision": data.get("last_decision")}
    except Exception:
        return {"last_decision": None}