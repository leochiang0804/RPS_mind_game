#!/usr/bin/env python3
"""
Test script to validate all centralized AI coach endpoints
"""

import requests
import json
import time
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_endpoint(endpoint_name, url, method='GET', data=None, expected_fields=None):
    """Test an endpoint and validate basic structure"""
    print(f"\n🧪 Testing {endpoint_name}...")
    
    try:
        if method == 'POST':
            response = requests.post(url, json=data, headers={'Content-Type': 'application/json'})
        else:
            response = requests.get(url)
            
        print(f"  Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"  ❌ Failed: {response.text}")
            return False
            
        result = response.json()
        
        # Check for success field
        if 'success' in result:
            print(f"  Success: {result['success']}")
            
        # Validate expected fields
        if expected_fields:
            for field in expected_fields:
                if field in result:
                    print(f"  ✅ {field}: present")
                else:
                    print(f"  ❌ {field}: missing")
                    
        # Show some key metrics if available
        if 'metrics' in result:
            metrics = result['metrics']
            print(f"  📊 Metrics categories: {len(metrics)}")
            
        if 'advice' in result:
            advice = result['advice']
            print(f"  💡 Advice type: {type(advice)}")
            
        if 'metrics_summary' in result:
            summary = result['metrics_summary']
            print(f"  📈 Summary fields: {len(summary)}")
            
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_with_game_data():
    """Test endpoints with some sample game data"""
    
    # Sample game data
    sample_data = {
        'human_moves': ['rock', 'paper', 'scissors'],
        'robot_moves': ['scissors', 'rock', 'paper'],
        'results': ['lose', 'lose', 'lose'],
        'llm_type': 'mock',
        'coaching_style': 'supportive'
    }
    
    base_url = "http://localhost:5000"
    
    print("🎯 Testing Centralized AI Coach Endpoints")
    print("=" * 50)
    
    # Test comprehensive endpoint
    success1 = test_endpoint(
        "Comprehensive Analysis",
        f"{base_url}/ai_coach/comprehensive",
        method='POST',
        data=sample_data,
        expected_fields=['success', 'metrics_summary', 'ai_behavior', 'coaching_advice']
    )
    
    # Test realtime endpoint  
    success2 = test_endpoint(
        "Real-time Coaching",
        f"{base_url}/ai_coach/realtime", 
        method='POST',
        data=sample_data,
        expected_fields=['success', 'advice', 'metrics_summary', 'llm_type']
    )
    
    # Test metrics endpoint
    success3 = test_endpoint(
        "Metrics Analysis",
        f"{base_url}/ai_coach/metrics",
        method='GET',
        expected_fields=['success', 'metrics', 'data_sources']
    )
    
    print("\n" + "=" * 50)
    if all([success1, success2, success3]):
        print("🎉 All centralized endpoints working!")
        return True
    else:
        print("❌ Some endpoints failed")
        return False

if __name__ == "__main__":
    print("🚀 Starting centralized endpoint tests...")
    
    # Wait a moment for server to be ready
    print("⏳ Waiting for server...")
    time.sleep(2)
    
    success = test_with_game_data()
    
    if success:
        print("\n✅ Centralization migration successful!")
        sys.exit(0) 
    else:
        print("\n❌ Centralization validation failed")
        sys.exit(1)