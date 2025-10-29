import json
from fastapi.testclient import TestClient

from src.app import app


client = TestClient(app)


def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200
    data = r.json()
    assert data == {"ok": True}


def test_flow_snapshot():
    r = client.get("/flow/snapshot")
    assert r.status_code == 200
    data = r.json()
    assert "signal" in data
    assert "confidence" in data


def test_alpha_hunter():
    body = {"symbol": "EURUSD", "timeframe": "1h"}
    r = client.post("/alpha/hunter", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["symbol"] == "EURUSD"
    assert "signal" in data and isinstance(data["signal"], (int, float))
    assert "confidence" in data and isinstance(data["confidence"], (int, float))


def test_stock_scanner_equity_and_nonequity():
    # Equity case
    r1 = client.get("/stock/scanner", params={"symbol": "SPY", "timeframe": "1h"})
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["symbol"] == "SPY"
    assert "signal" in d1 and "confidence" in d1

    # Non-equity: FX should return neutral low confidence
    r2 = client.get("/stock/scanner", params={"symbol": "EURUSD", "timeframe": "1h"})
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["symbol"] == "EURUSD"
    assert "signal" in d2 and "confidence" in d2


def test_correl_score_peers_and_score():
    r = client.get("/correl/score", params={"symbol": "EURUSD", "timeframe": "1h"})
    assert r.status_code == 200
    data = r.json()
    assert data["symbol"] == "EURUSD"
    assert "peers" in data and isinstance(data["peers"], list) and len(data["peers"]) > 0
    assert "score" in data and isinstance(data["score"], (int, float))
    assert "signal" in data and isinstance(data["signal"], (int, float))
    assert "confidence" in data and isinstance(data["confidence"], (int, float))


def test_stocks_smartfinder():
    r = client.get("/stocks/smartfinder", params={"symbol": "PLTR", "timeframe": "1h", "price": 18.5})
    assert r.status_code == 200
    data = r.json()
    assert data["symbol"] == "PLTR"
    assert "signal" in data and isinstance(data["signal"], (int, float))
    assert "confidence" in data and isinstance(data["confidence"], (int, float))


def test_flow_anomaly():
    r = client.get("/flow/anomaly", params={"symbol": "EURUSD", "timeframe": "1h"})
    assert r.status_code == 200
    data = r.json()
    assert data["symbol"] == "EURUSD"
    assert "signal" in data and isinstance(data["signal"], (int, float))
    assert "confidence" in data and isinstance(data["confidence"], (int, float))


def test_orchestrator_last_decision():
    r = client.get("/orchestrator/last_decision")
    assert r.status_code == 200
    data = r.json()
    assert "last_decision" in data