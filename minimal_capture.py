#!/usr/bin/env python3
"""
Minimal baseline capture using only standard library + requests
"""

import subprocess
import sys
import os

def run_capture():
    """Run the baseline capture in a clean Python environment"""
    
    # Install requests if not available
    try:
        import requests
    except ImportError:
        print("Installing requests...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
        import requests
    
    import json
    import time
    from datetime import datetime
    
    # Configuration
    BASE_URL = "http://127.0.0.1:5050"
    BASELINE_DIR = "tests/baseline_snapshots"
    
    # Create directory
    os.makedirs(BASELINE_DIR, exist_ok=True)
    
    print("🚀 Starting minimal baseline capture")
    
    # Test server
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        print(f"✅ Server accessible (status: {response.status_code})")
    except Exception as e:
        print(f"❌ Server not accessible: {e}")
        return
    
    # Simple test scenario
    session = requests.Session()
    moves = ["paper", "stone", "paper"]
    
    print("🎯 Playing test scenario...")
    game_responses = []
    
    for i, move in enumerate(moves):
        print(f"  Round {i+1}: {move}")
        try:
            response = session.post(f"{BASE_URL}/play", json={"move": move})
            if response.status_code == 200:
                game_responses.append(response.json())
                print(f"    ✅ Success")
            else:
                print(f"    ❌ Failed: {response.status_code}")
        except Exception as e:
            print(f"    ❌ Error: {e}")
    
    # Capture one AI coach endpoint
    print("📊 Capturing AI coach comprehensive...")
    try:
        response = session.post(f"{BASE_URL}/ai_coach/comprehensive", json={"type": "comprehensive"})
        if response.status_code == 200:
            coach_data = response.json()
            print(f"    ✅ Captured comprehensive data")
            
            # Save snapshot
            snapshot = {
                "scenario": "test_capture",
                "timestamp": datetime.now().isoformat(),
                "moves": moves,
                "game_responses": game_responses,
                "comprehensive": coach_data
            }
            
            filename = f"{BASELINE_DIR}/test_baseline.json"
            with open(filename, 'w') as f:
                json.dump(snapshot, f, indent=2)
            
            print(f"✅ Snapshot saved: {filename}")
            
        else:
            print(f"    ❌ AI coach failed: {response.status_code}")
    except Exception as e:
        print(f"    ❌ AI coach error: {e}")

if __name__ == "__main__":
    run_capture()