#!/usr/bin/env python3
"""
Simple baseline capture script for data centralization refactoring
Captures current API responses for regression testing
"""

import requests
import json
import os
import time
from datetime import datetime

# Configuration
BASE_URL = "http://127.0.0.1:5050"  # Default webapp port
BASELINE_DIR = "tests/baseline_snapshots"

def ensure_directory():
    """Create baseline directory if it doesn't exist"""
    os.makedirs(BASELINE_DIR, exist_ok=True)

def capture_game_scenario(scenario_name, moves_sequence):
    """
    Capture API responses for a specific game scenario
    
    Args:
        scenario_name: Name for the scenario (e.g., "short_match_human_wins")
        moves_sequence: List of moves to play (e.g., ["paper", "stone", "scissor"])
    """
    print(f"\n� Capturing scenario: {scenario_name}")
    
    # Start fresh session
    session = requests.Session()
    
    # Play the sequence of moves
    game_responses = []
    for i, move in enumerate(moves_sequence):
        print(f"  Round {i+1}: Playing {move}")
        try:
            response = session.post(f"{BASE_URL}/play", json={"move": move})
            if response.status_code == 200:
                game_responses.append(response.json())
            else:
                print(f"    ❌ Play failed: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"    ❌ Connection error: {e}")
            return None
    
    # Capture AI Coach endpoints
    endpoints_data = {}
    
    ai_coach_endpoints = [
        ("status", "/ai_coach/status", "GET"),
        ("realtime", "/ai_coach/realtime", "POST"),
        ("comprehensive", "/ai_coach/comprehensive", "POST"),
        ("metrics", "/ai_coach/metrics", "GET")
    ]
    
    for endpoint_name, endpoint_path, method in ai_coach_endpoints:
        print(f"  Capturing {endpoint_name}...")
        try:
            if method == "GET":
                response = session.get(f"{BASE_URL}{endpoint_path}")
            else:
                response = session.post(f"{BASE_URL}{endpoint_path}", json={"type": endpoint_name})
            
            if response.status_code == 200:
                endpoints_data[endpoint_name] = response.json()
                print(f"    ✅ {endpoint_name} captured")
            else:
                print(f"    ⚠️  {endpoint_name} failed: {response.status_code}")
                endpoints_data[endpoint_name] = {"error": response.status_code, "text": response.text}
        except Exception as e:
            print(f"    ❌ {endpoint_name} error: {e}")
            endpoints_data[endpoint_name] = {"error": str(e)}
    
    # Compile snapshot
    snapshot = {
        "scenario_name": scenario_name,
        "timestamp": datetime.now().isoformat(),
        "moves_sequence": moves_sequence,
        "game_responses": game_responses,
        "ai_coach_endpoints": endpoints_data,
        "final_game_state": game_responses[-1] if game_responses else None
    }
    
    # Save to file
    filename = f"{BASELINE_DIR}/{scenario_name}_baseline.json"
    with open(filename, 'w') as f:
        json.dump(snapshot, f, indent=2)
    
    print(f"  ✅ Snapshot saved: {filename}")
    return snapshot

def main():
    """Capture baseline snapshots for representative scenarios"""
    ensure_directory()
    
    print("🚀 Starting baseline capture for data centralization refactoring")
    print(f"Target server: {BASE_URL}")
    
    # Test server connectivity
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        print(f"✅ Server accessible (status: {response.status_code})")
    except Exception as e:
        print(f"❌ Server not accessible: {e}")
        print("Please ensure webapp is running on the expected port")
        return
    
    # Define test scenarios
    scenarios = [
        ("short_match_human_wins", ["paper", "stone", "paper"]),
        ("short_match_ai_wins", ["stone", "paper", "stone"]),
        ("draw_heavy_match", ["paper", "paper", "stone", "stone", "scissor", "scissor"]),
        ("longer_match", ["paper", "stone", "scissor", "paper", "stone", "scissor", "paper", "stone"])
    ]
    
    results = {}
    for scenario_name, moves in scenarios:
        result = capture_game_scenario(scenario_name, moves)
        results[scenario_name] = result is not None
        
        # Brief pause between scenarios
        time.sleep(1)
    
    # Summary
    print(f"\n📊 Baseline Capture Summary:")
    for scenario, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {scenario}")
    
    successful_count = sum(results.values())
    print(f"\n🎯 Captured {successful_count}/{len(scenarios)} scenarios successfully")
    
    if successful_count > 0:
        print(f"� Snapshots saved in: {BASELINE_DIR}/")
        print("Ready for refactoring phase!")
    else:
        print("⚠️  No scenarios captured. Check server connectivity.")

if __name__ == "__main__":
    main()