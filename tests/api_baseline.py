#!/usr/bin/env python3
"""
Baseline API Test - Pure Client Script
Tests AI coach endpoints without importing Flask app components
"""

import requests
import json
import time
from datetime import datetime

def capture_baseline():
    """Capture baseline behavior by testing API endpoints directly"""
    
    print("🎮 Starting Baseline API Capture")
    print("This script tests AI coach endpoints as a pure client")
    print()
    
    # Configuration
    base_url = "http://127.0.0.1:5050"
    session = requests.Session()
    
    # Test basic connectivity first
    try:
        response = session.get(f"{base_url}/")
        print(f"✅ Server connectivity: {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("Please ensure webapp is running: python webapp/app.py")
        return
    
    # Scenario: Short game sequence
    moves = ['paper', 'stone', 'paper']
    
    print(f"\n🎲 Playing sequence: {moves}")
    
    # Play moves and collect responses
    game_responses = []
    for i, move in enumerate(moves):
        try:
            response = session.post(f"{base_url}/play", json={'move': move})
            if response.status_code == 200:
                data = response.json()
                game_responses.append(data)
                print(f"   Round {i+1} ({move}): ✅ Total moves = {data.get('total_moves', '?')}")
            else:
                print(f"   Round {i+1} ({move}): ❌ Status {response.status_code}")
                print(f"     Error: {response.text}")
        except Exception as e:
            print(f"   Round {i+1} ({move}): ❌ Exception: {e}")
    
    if not game_responses:
        print("❌ No successful game responses, cannot test AI coach endpoints")
        return
    
    print(f"\n🤖 Testing AI Coach Endpoints")
    
    # Test AI coach endpoints
    endpoints_to_test = [
        {
            'name': 'AI Coach Status',
            'method': 'GET',
            'path': '/ai_coach/status',
            'payload': None
        },
        {
            'name': 'AI Coach Realtime',
            'method': 'POST', 
            'path': '/ai_coach/realtime',
            'payload': {'type': 'realtime'}
        },
        {
            'name': 'AI Coach Comprehensive',
            'method': 'POST',
            'path': '/ai_coach/comprehensive', 
            'payload': {'type': 'comprehensive'}
        },
        {
            'name': 'AI Coach Metrics',
            'method': 'GET',
            'path': '/ai_coach/metrics',
            'payload': None
        }
    ]
    
    endpoint_results = {}
    
    for endpoint in endpoints_to_test:
        name = endpoint['name']
        method = endpoint['method']
        path = endpoint['path']
        payload = endpoint['payload']
        
        try:
            if method == 'GET':
                response = session.get(f"{base_url}{path}")
            else:
                response = session.post(f"{base_url}{path}", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                endpoint_results[path] = {
                    'status': 'success',
                    'data': data
                }
                
                # Show key information for comprehensive analysis
                if path == '/ai_coach/comprehensive' and 'metrics_summary' in data:
                    metrics = data['metrics_summary']
                    print(f"   {name}: ✅ Categories: {list(metrics.keys())}")
                    
                    # Check AI behavior specifically
                    ai_behavior = metrics.get('ai_behavior', {})
                    if ai_behavior:
                        print(f"     AI Behavior keys: {list(ai_behavior.keys())}")
                        model_accuracy = ai_behavior.get('model_accuracy', {})
                        if model_accuracy:
                            print(f"     Model accuracy models: {list(model_accuracy.keys())}")
                        else:
                            print(f"     ⚠️ Model accuracy is empty")
                    else:
                        print(f"     ⚠️ AI behavior section is empty")
                else:
                    print(f"   {name}: ✅")
                    
            else:
                endpoint_results[path] = {
                    'status': 'error',
                    'status_code': response.status_code,
                    'error': response.text
                }
                print(f"   {name}: ❌ Status {response.status_code}")
                
        except Exception as e:
            endpoint_results[path] = {
                'status': 'exception',
                'error': str(e)
            }
            print(f"   {name}: ❌ Exception: {e}")
    
    # Create baseline snapshot
    baseline_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'test_type': 'api_baseline_capture',
            'base_url': base_url,
            'moves_played': moves
        },
        'game_responses': game_responses,
        'ai_coach_endpoints': endpoint_results
    }
    
    # Save snapshot
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"tests/regression/baseline_api_{timestamp_str}.json"
    
    with open(filename, 'w') as f:
        json.dump(baseline_data, f, indent=2, default=str)
    
    print(f"\n💾 Baseline saved: {filename}")
    
    # Summary
    successful_endpoints = len([r for r in endpoint_results.values() if r['status'] == 'success'])
    total_endpoints = len(endpoint_results)
    
    print(f"\n📊 Baseline Capture Summary:")
    print(f"   Game rounds played: {len(game_responses)}")
    print(f"   AI coach endpoints tested: {successful_endpoints}/{total_endpoints}")
    print(f"   Snapshot file: {filename}")
    
    if successful_endpoints == total_endpoints:
        print("✅ All endpoints captured successfully")
    else:
        print("⚠️ Some endpoints failed - check logs above")

if __name__ == "__main__":
    capture_baseline()