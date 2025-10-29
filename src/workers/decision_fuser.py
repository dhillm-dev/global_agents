import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, Any
from datetime import datetime

import requests


DEFAULT_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
STATE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "state")
MEMORY_PATH = os.path.join(STATE_DIR, "fuser_memory.json")


SIGNALS = [
    ("flow", "/flow/snapshot"),
    ("alpha", "/alpha/hunter"),
    ("stock", "/stock/scanner"),
    ("correl", "/correl/score"),
]


def _ensure_state_dir() -> None:
    if not os.path.isdir(STATE_DIR):
        os.makedirs(STATE_DIR, exist_ok=True)


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


@dataclass
class Memory:
    weights: Dict[str, float]
    last_decision_sign: Optional[int] = None  # -1, 0, 1
    last_fused_score: Optional[float] = None
    stats: Dict[str, Any] = None


class DecisionFuser:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, lr: float = 0.05, init_weight: float = 0.25):
        self.base_url = base_url.rstrip("/")
        self.lr = lr
        self.init_weight = init_weight
        _ensure_state_dir()
        self.memory: Dict[str, Memory] = self._load_memory()

    def _key(self, symbol: str, timeframe: str) -> str:
        return f"{symbol}|{timeframe}"

    def _load_memory(self) -> Dict[str, Memory]:
        if not os.path.exists(MEMORY_PATH):
            return {}
        try:
            with open(MEMORY_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            out: Dict[str, Memory] = {}
            for k, v in raw.items():
                out[k] = Memory(**v)
            return out
        except Exception:
            return {}

    def _save_memory(self) -> None:
        try:
            serializable = {k: asdict(v) for k, v in self.memory.items()}
            with open(MEMORY_PATH, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2)
        except Exception:
            pass

    def _init_weights_if_needed(self, key: str) -> None:
        if key not in self.memory:
            self.memory[key] = Memory(
                weights={s: self.init_weight for s, _ in SIGNALS},
                last_decision_sign=None,
                last_fused_score=None,
                stats={"cycles": 0, "wins": 0, "losses": 0},
            )

    def fetch_signal(self, endpoint: str, symbol: str, timeframe: str) -> Optional[float]:
        url = f"{self.base_url}{endpoint}"
        try:
            resp = requests.get(url, params={"symbol": symbol, "timeframe": timeframe}, timeout=10)
            if resp.status_code != 200:
                return None
            data = resp.json()
            # Prefer 'score'; fall back to common numeric fields
            for key in ("score", "alpha", "value", "signal"):
                v = data.get(key)
                if isinstance(v, (int, float)):
                    return float(v)
            # Try nested fields
            v = (
                data.get("flow", {}).get("score")
                if isinstance(data.get("flow"), dict)
                else None
            )
            return float(v) if isinstance(v, (int, float)) else None
        except Exception:
            return None

    def fuse(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        key = self._key(symbol, timeframe)
        self._init_weights_if_needed(key)
        mem = self.memory[key]

        # Fetch scores from endpoints
        scores: Dict[str, Optional[float]] = {}
        for sig_name, endpoint in SIGNALS:
            scores[sig_name] = self.fetch_signal(endpoint, symbol, timeframe)

        # Normalize weights only over available signals
        available = {k: v for k, v in scores.items() if isinstance(v, (int, float))}
        if not available:
            fused_score = 0.0
        else:
            total_w = sum(mem.weights[k] for k in available.keys())
            if total_w <= 0:
                # Reset to equal if degenerate
                for k in mem.weights.keys():
                    mem.weights[k] = self.init_weight
                total_w = sum(mem.weights[k] for k in available.keys())
            fused_score = sum((mem.weights[k] / total_w) * available[k] for k in available.keys())

        # Decision thresholds
        decision = "hold"
        decision_sign = 0
        if fused_score > 0.2:
            decision = "buy"
            decision_sign = 1
        elif fused_score < -0.2:
            decision = "sell"
            decision_sign = -1

        # Adaptive weight update using simple correctness proxy
        # Reward signals whose sign aligned with the subsequent fused score
        if mem.last_decision_sign is not None and mem.last_fused_score is not None:
            reward_sign = 1 if mem.last_decision_sign * fused_score > 0 else -1 if mem.last_decision_sign * fused_score < 0 else 0
            for k, v in available.items():
                sig_sign = 1 if v > 0 else -1 if v < 0 else 0
                aligned = 1 if sig_sign == (1 if fused_score > 0 else -1 if fused_score < 0 else 0) else -1 if sig_sign != 0 else 0
                # Blend reward with alignment to avoid overfitting to a single cycle
                delta = self.lr * (0.7 * reward_sign + 0.3 * aligned)
                mem.weights[k] = max(0.001, mem.weights[k] * (1 + delta))

            # Renormalize weights (not strictly necessary; handled during fusing)
            sum_w = sum(mem.weights.values())
            if sum_w > 0:
                for k in mem.weights.keys():
                    mem.weights[k] = mem.weights[k] / sum_w

            # Update stats
            mem.stats["cycles"] += 1
            if reward_sign > 0:
                mem.stats["wins"] += 1
            elif reward_sign < 0:
                mem.stats["losses"] += 1

        # Persist memory
        mem.last_decision_sign = decision_sign
        mem.last_fused_score = fused_score
        self._save_memory()

        result = {
            "ts": _now_iso(),
            "symbol": symbol,
            "timeframe": timeframe,
            "decision": decision,
            "confidence": round(abs(fused_score), 5),
            "fused_score": round(fused_score, 5),
            "components": {k: (round(v, 5) if isinstance(v, (int, float)) else None) for k, v in scores.items()},
            "weights": {k: round(w, 5) for k, w in mem.weights.items()},
            "memory_stats": mem.stats,
        }
        return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fuse signals from API and emit a JSON decision")
    p.add_argument("--symbols", type=str, default="EURUSD", help="Comma-separated symbols, e.g., EURUSD,BTCUSDT")
    p.add_argument("--timeframe", type=str, default="1h", help="Timeframe string, e.g., 1h, 5m")
    p.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL, help="Base URL of the API, defaults to API_BASE_URL env")
    p.add_argument("--interval", type=int, default=300, help="Seconds between cycles when not using --once")
    p.add_argument("--once", action="store_true", help="Run just one cycle")
    p.add_argument("--lr", type=float, default=0.05, help="Learning rate for adaptive weights")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fuser = DecisionFuser(base_url=args.base_url, lr=args.lr)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    def run_cycle() -> None:
        for sym in symbols:
            result = fuser.fuse(sym, args.timeframe)
            print(json.dumps(result))

    if args.once:
        run_cycle()
        return

    while True:
        start = time.time()
        run_cycle()
        elapsed = time.time() - start
        sleep_for = max(0, args.interval - elapsed)
        time.sleep(sleep_for)


if __name__ == "__main__":
    main()