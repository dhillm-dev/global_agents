import os
import json
import math
import asyncio
import aiohttp
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


def fuse_agents_once(symbol: str, tf: str) -> Optional[Dict[str, Any]]:
    """
    Demo fusion: return a small BUY decision so MT5 EA can trade.
    Replace with real Alpha/Correl/Flow/Sentiment fusion as needed.
    Return None to indicate FLAT (no trade).
    """
    return {
        "action": "BUY",
        "symbol": symbol,
        "volume": 0.02,
        "comment": f"demo-fuse {tf}",
        # Optional risk fields EA may consume:
        # "sl_points": 250,
        # "tp_points": 500,
    }


# ----------------------------
# Config
# ----------------------------
BASE_URL = os.getenv("GA_BASE_URL", "https://global-agents-jciy.onrender.com").rstrip("/")
SYMBOLS = [s.strip() for s in os.getenv("GA_SYMBOLS", "EURUSD,GBPUSD,SPY,BTCUSDT").split(",") if s.strip()]
TF = os.getenv("GA_TIMEFRAME", "1h")
CYCLE_S = int(os.getenv("GA_CYCLE_SECONDS", "600"))
MIN_CONF = float(os.getenv("GA_MIN_CONFIDENCE", "0.40"))

TMP_DIR = Path(os.getenv("TMPDIR", "/tmp"))
# Default memory path uses writable ephemeral /tmp on serverless platforms
MEM_PATH = Path(os.getenv("GA_MEMORY_PATH", str(TMP_DIR / "global_agents" / "_memory.json")))
try:
    MEM_PATH.parent.mkdir(parents=True, exist_ok=True)
except Exception:
    # On some environments, directory creation may fail; writes will be skipped gracefully
    pass

ENDPOINTS = {
    "AlphaHunter": "/alpha/hunter",
    "FlowWatcher": "/flow/snapshot",
    "StockScanner": "/stock/scanner",
    "CorrelationAI": "/correl/score",
    "SmartStockFinder": "/stocks/smartfinder",
    "FlowAnomalyDetector": "/flow/anomaly",
}

DEFAULT_WEIGHTS = {
    "AlphaHunter": 0.30,
    "FlowWatcher": 0.30,
    "StockScanner": 0.20,
    "CorrelationAI": 0.20,
    "SmartStockFinder": 0.15,
    "FlowAnomalyDetector": 0.15,
}

HALF_LIFE = int(os.getenv("GA_HALF_LIFE", "50"))
MAX_RPM = 30  # requests per minute throttle
BASE_RISK = float(os.getenv("GA_BASE_RISK", "0.01"))

# Execution / Paper trading config
BROKER_PROVIDER = os.getenv("BROKER_PROVIDER", "paper")
TRADE_MODE = os.getenv("TRADE_MODE", "paper").lower()
RISK_MAX_DAILY_DD = float(os.getenv("RISK_MAX_DAILY_DD", "0.05"))
RISK_MAX_PER_TRADE = float(os.getenv("RISK_MAX_PER_TRADE", "0.02"))
RISK_MAX_CONCURRENT = int(os.getenv("RISK_MAX_CONCURRENT", "3"))
SYMBOL_FILTER = [s.strip() for s in os.getenv("SYMBOL_FILTER", ",".join(SYMBOLS)).split(",") if s.strip()]

# Persist paper ledger and trades log in /tmp by default
LEDGER_PATH = Path(os.getenv("GA_LEDGER_PATH", str(TMP_DIR / "data" / "paper_ledger.json")))
TRADES_LOG = Path(os.getenv("GA_TRADES_LOG", str(TMP_DIR / "logs" / "trades.jsonl")))
for p in [LEDGER_PATH.parent, TRADES_LOG.parent]:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


# ----------------------------
# Utilities
# ----------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))


def to_minus1_plus1(signal: float) -> float:
    if -1.0 <= signal <= 1.0:
        return float(signal)
    return float(2.0 * signal - 1.0)


def is_equity(sym: str) -> bool:
    return ("/" not in sym) and (not sym.endswith("USDT"))


# ----------------------------
# Memory and Metrics
# ----------------------------
class Memory:
    def __init__(self, path: Path):
        self.path = path
        self.data: Dict[str, Any] = {
            "weights": DEFAULT_WEIGHTS.copy(),
            "history": [],
            "metrics": {
                "win_rate": 0.0,
                "avg_return": 0.0,
                "weight_entropy": 0.0,
                "conf_calibration": 0.0,
                "variance": 0.0,
                "samples": 0,
            },
        }
        self.load()

    def load(self):
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text())
            except Exception:
                pass

    def save(self):
        # Ensure parent directory exists before writing
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.replace(self.path)

    @property
    def weights(self) -> Dict[str, float]:
        for k, v in DEFAULT_WEIGHTS.items():
            self.data["weights"].setdefault(k, v)
        return self.data["weights"]

    def log_outcome(self, record: Dict[str, Any]):
        self.data["history"].append(record)
        self.data["history"] = self.data["history"][-5000:]

    def update_metrics(self, correct: bool, expected_ret: float, confidence: float, realized_ret: float):
        # Ensure metrics defaults exist
        m = self.data.setdefault("metrics", {
            "win_rate": 0.0,
            "avg_return": 0.0,
            "weight_entropy": 0.0,
            "conf_calibration": 0.0,
            "variance": 0.0,
            "samples": 0,
        })
        m.setdefault("win_rate", 0.0)
        m.setdefault("avg_return", 0.0)
        m.setdefault("weight_entropy", 0.0)
        m.setdefault("conf_calibration", 0.0)
        m.setdefault("variance", 0.0)
        m.setdefault("samples", 0)
        n = m["samples"] + 1
        # Running averages
        m["win_rate"] = ((m["win_rate"] * m["samples"]) + (1.0 if correct else 0.0)) / n
        m["avg_return"] = ((m["avg_return"] * m["samples"]) + realized_ret) / n
        # Confidence calibration (how close confidence is to correctness)
        calib = abs((1.0 if correct else 0.0) - confidence)
        m["conf_calibration"] = ((m["conf_calibration"] * m["samples"]) + (1.0 - calib)) / n
        # Variance of returns (running via Welford-ish simple update)
        # For simplicity, track mean absolute deviation proxy
        mad = abs(realized_ret - m["avg_return"])
        m["variance"] = ((m["variance"] * m["samples"]) + mad) / n
        m["samples"] = n
        # Weight entropy: -sum(w log w)
        w = self.weights
        entropy = 0.0
        for v in w.values():
            if v > 0:
                entropy += -v * math.log(v)
        self.data["metrics"]["weight_entropy"] = entropy

    def update_weights(self, contributions: Dict[str, float], decision_dir: int, realized_dir: int):
        # reinforcement per spec
        if realized_dir == 0:
            for k in self.weights:
                self.weights[k] *= 0.995
            self._renorm()
            return

        reward = 1.0 if realized_dir == decision_dir else -1.0
        for agent, contrib in contributions.items():
            delta = 0.05 * reward * abs(contrib)
            self.weights[agent] = clamp(self.weights.get(agent, 0.2) + delta, 0.05, 0.80)

        # decay toward defaults (half-life style simplified)
        for k in self.weights:
            self.weights[k] = 0.99 * self.weights[k] + 0.01 * DEFAULT_WEIGHTS[k]

        self._renorm()

    def _renorm(self):
        s = sum(self.weights.values())
        if s <= 0:
            self.weights.update(DEFAULT_WEIGHTS)
            s = sum(self.weights.values())
        for k in self.weights:
            self.weights[k] = self.weights[k] / s


# ----------------------------
# Orchestrator
# ----------------------------
class Orchestrator:
    def __init__(self, session: aiohttp.ClientSession, mem: Memory):
        self.http = session
        self.mem = mem
        self._recent_requests: List[float] = []

    async def _throttle(self):
        # Keep under MAX_RPM; clear timestamps older than 60s.
        now = datetime.now().timestamp()
        self._recent_requests = [t for t in self._recent_requests if now - t < 60]
        if len(self._recent_requests) >= MAX_RPM:
            await asyncio.sleep(1)
            return await self._throttle()

    async def _get(self, path: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        await self._throttle()
        url = f"{BASE_URL}{path}"
        try:
            async with self.http.get(url, params=params, timeout=25) as r:
                self._recent_requests.append(datetime.now().timestamp())
                if r.status == 200:
                    return await r.json()
        except Exception:
            return None
        return None

    async def _post(self, path: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        await self._throttle()
        url = f"{BASE_URL}{path}"
        try:
            async with self.http.post(url, json=payload, timeout=30) as r:
                self._recent_requests.append(datetime.now().timestamp())
                if r.status == 200:
                    return await r.json()
        except Exception:
            return None
        return None

    async def query_agents(self, symbol: str) -> Dict[str, Dict[str, Any]]:
        tasks: List[Tuple[str, asyncio.Task]] = []
        tasks.append(("AlphaHunter", asyncio.create_task(self._post(ENDPOINTS["AlphaHunter"], {"symbol": symbol, "timeframe": TF, "context": "Trae cycle"}))))
        tasks.append(("FlowWatcher", asyncio.create_task(self._get(ENDPOINTS["FlowWatcher"], {"symbol": symbol, "timeframe": TF}))))
        tasks.append(("StockScanner", asyncio.create_task(self._get(ENDPOINTS["StockScanner"], {"symbol": symbol, "timeframe": TF}))))
        tasks.append(("CorrelationAI", asyncio.create_task(self._get(ENDPOINTS["CorrelationAI"], {"symbol": symbol, "timeframe": TF}))))
        tasks.append(("SmartStockFinder", asyncio.create_task(self._get(ENDPOINTS["SmartStockFinder"], {"symbol": symbol, "timeframe": TF}))))
        tasks.append(("FlowAnomalyDetector", asyncio.create_task(self._get(ENDPOINTS["FlowAnomalyDetector"], {"symbol": symbol, "timeframe": TF}))))

        results: Dict[str, Dict[str, Any]] = {}
        for name, task in tasks:
            try:
                resp = await task
                if isinstance(resp, dict):
                    results[name] = resp
            except Exception:
                pass
        return results

    def fuse(self, symbol: str, agents: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        w = self.mem.weights
        numer = 0.0
        denom = 0.0
        contribs: Dict[str, float] = {}
        fad_conf = 0.0
        fad_sig = 0.0
        ssf_conf = 0.0
        for agent, data in agents.items():
            sig = to_minus1_plus1(float(data.get("signal", 0.0)))
            conf = float(data.get("confidence", 0.0))
            wa = float(w.get(agent, DEFAULT_WEIGHTS.get(agent, 0.2)))
            c = sig * wa * conf
            contribs[agent] = c
            numer += c
            denom += wa
            if agent == "FlowAnomalyDetector":
                fad_conf = conf
                fad_sig = sig
            if agent == "SmartStockFinder":
                ssf_conf = conf
        conviction = (numer / denom) if denom > 0 else 0.0
        confidence = clamp(0.55 + 0.45 * abs(conviction), 0.0, 1.0)
        direction = "HOLD"
        if confidence < MIN_CONF:
            direction = "HOLD"
        else:
            if conviction > 0.35:
                direction = "BUY"
            elif conviction < -0.35:
                direction = "SELL"

        # Dynamic risk sizing and hedge rule
        risk = BASE_RISK * confidence * abs(conviction)
        def _sgn(x: float) -> int:
            return 1 if x > 0 else (-1 if x < 0 else 0)
        if fad_conf > 0.8 and _sgn(conviction) != _sgn(fad_sig):
            risk *= 0.5

        # Expected return proxy leveraging SSF confidence
        expected_return = conviction * 0.02 * (1.0 + ssf_conf)

        result = {
            "timestamp": utc_now(),
            "symbol": symbol,
            "timeframe": TF,
            "direction": direction,
            "conviction": round(conviction, 5),
            "confidence": round(confidence, 5),
            "risk": round(risk, 5),
            "expected_return": round(expected_return, 5),
            "rationale": "Fusion of agents with adaptive weights",
            "agent_contributions": {k: round(v, 5) for k, v in contribs.items()},
        }
        return result

    def learn(self, fused: Dict[str, Any], realized_move: float):
        # realized_dir from realized move
        realized_dir = 1 if realized_move > 0 else -1 if realized_move < 0 else 0
        decision_dir = 1 if fused["direction"] == "BUY" else -1 if fused["direction"] == "SELL" else 0
        self.mem.update_weights(fused.get("agent_contributions", {}), decision_dir, realized_dir)
        correct = (decision_dir == realized_dir and decision_dir != 0)
        self.mem.update_metrics(correct, fused.get("expected_return", 0.0), fused.get("confidence", 0.0), realized_move)
        # Periodic decay every 50 samples
        if self.mem.data["metrics"]["samples"] % 50 == 0:
            for k in list(self.mem.weights.keys()):
                self.mem.weights[k] *= 0.98
            self.mem._renorm()

        # Persist full last decision record with realized move
        record = {
            **fused,
            "realized_move": realized_move,
        }
        # update memory fields and save
        self.mem.log_outcome(record)
        # also write last_decision for /orchestrator/last_decision
        self.mem.data["last_decision"] = record
        self.mem.save()

    # ----------------------------
    # Paper trading helpers
    # ----------------------------
    def _load_ledger(self) -> List[Dict[str, Any]]:
        try:
            if LEDGER_PATH.exists():
                return json.loads(LEDGER_PATH.read_text())
        except Exception:
            pass
        return []

    def _save_ledger(self, ledger: List[Dict[str, Any]]):
        try:
            tmp = LEDGER_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(ledger, indent=2))
            tmp.replace(LEDGER_PATH)
        except Exception:
            pass

    def _count_open_trades(self, ledger: List[Dict[str, Any]]) -> int:
        cnt = 0
        for t in ledger:
            if t.get("status") == "open":
                cnt += 1
        return cnt

    def _pre_trade_checks(self, fused: Dict[str, Any]) -> bool:
        if TRADE_MODE != "paper":
            return False
        if fused.get("direction") == "HOLD":
            return False
        if float(fused.get("confidence", 0.0)) < MIN_CONF:
            return False
        if float(fused.get("risk", 0.0)) > RISK_MAX_PER_TRADE:
            return False
        # Concurrency
        ledger = self._load_ledger()
        if self._count_open_trades(ledger) >= RISK_MAX_CONCURRENT:
            return False
        # Simple daily DD kill-switch using memory metrics avg_return as proxy
        # In a real system, compute true daily PnL; here we keep it conservative.
        dd_proxy = abs(self.mem.data.get("metrics", {}).get("avg_return", 0.0))
        if dd_proxy >= RISK_MAX_DAILY_DD:
            return False
        return True

    def _execute_paper_trade(self, fused: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            side = "buy" if fused.get("direction") == "BUY" else "sell"
            symbol = fused.get("symbol")
            order = {
                "symbol": symbol,
                "side": side,
                "qty": 10000,
                "type": "market",
                "sl": None,
                "tp": None,
                "ts": utc_now(),
                "status": "open",
            }
            # Example SL/TP for EURUSD
            if symbol == "EURUSD":
                order["sl"] = 1.0725
                order["tp"] = 1.0815
            # Persist to ledger and log
            ledger = self._load_ledger()
            ledger.append(order)
            self._save_ledger(ledger)
            try:
                TRADES_LOG.open("a", encoding="utf-8").write(json.dumps(order) + "\n")
            except Exception:
                pass
            return order
        except Exception:
            return None


async def run_once(symbols: List[str]) -> None:
    async with aiohttp.ClientSession() as session:
        mem = Memory(MEM_PATH)
        orch = Orchestrator(session, mem)
        for sym in symbols:
            agents = await orch.query_agents(sym)
            fused = orch.fuse(sym, agents)
            print(json.dumps(fused))
            # POST fused decision to backend for dashboard/history
            try:
                await orch._post("/orchestrator/decision", fused)
            except Exception:
                pass
            # Paper execution
            if sym in SYMBOL_FILTER and orch._pre_trade_checks(fused):
                orch._execute_paper_trade(fused)
            # dummy realized move proxy: small random drift around conviction
            realized_move = fused["conviction"] * 0.5
            orch.learn(fused, realized_move)


async def run_loop() -> None:
    while True:
        await run_once(SYMBOLS)
        await asyncio.sleep(CYCLE_S)


def main() -> None:
    global SYMBOLS, CYCLE_S
    import argparse
    p = argparse.ArgumentParser(description="Global Alpha Orchestrator (Trae)")
    p.add_argument("--once", action="store_true", help="Run one cycle then exit")
    p.add_argument("--symbols", type=str, default=",".join(SYMBOLS), help="Comma-separated symbols")
    p.add_argument("--cycle", type=int, default=CYCLE_S, help="Cycle seconds")
    args = p.parse_args()

    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    SYMBOLS = syms
    CYCLE_S = args.cycle

    if args.once:
        asyncio.run(run_once(SYMBOLS))
    else:
        asyncio.run(run_loop())


if __name__ == "__main__":
    main()