#!/usr/bin/env python3
"""
Test script for orchestrator endpoints
"""
import requests
import json

BASE = "http://127.0.0.1:8001"

def test_orchestrator():
    print("=== Testing Orchestrator Endpoints ===")
    
    # 1. Get last_decision (before)
    print("\n1. GET /orchestrator/last_decision (before)")
    try:
        r1 = requests.get(f"{BASE}/orchestrator/last_decision", timeout=10)
        print(f"Status: {r1.status_code}")
        print(f"Response: {r1.text}")
    except Exception as e:
        print(f"Error: {e}")
    
    # 2. POST run_once to generate a new decision
    print("\n2. POST /orchestrator/run_once")
    try:
        payload = {"symbol": "ETHUSD", "tf": "H1"}
        r2 = requests.post(f"{BASE}/orchestrator/run_once", json=payload, timeout=10)
        print(f"Status: {r2.status_code}")
        print(f"Response: {r2.text}")
    except Exception as e:
        print(f"Error: {e}")
    
    # 3. Get last_decision (after)
    print("\n3. GET /orchestrator/last_decision (after)")
    try:
        r3 = requests.get(f"{BASE}/orchestrator/last_decision", timeout=10)
        print(f"Status: {r3.status_code}")
        print(f"Response: {r3.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_orchestrator()