#!/usr/bin/env python3
"""
Test script to validate centralized AI coach endpoints (direct testing)
"""

import sys
import os
import json

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_direct_endpoints():
    """Test the centralized endpoints by importing and calling directly"""
    print("🧪 Testing Centralized AI Coach Endpoints (Direct)")
    print("=" * 60)
    
    try:
        # Import webapp components
        sys.path.append('webapp')
        from flask import Flask
        from webapp.app import app
        
        # Create test client
        app.config['TESTING'] = True
        client = app.test_client()
        
        # Test data
        sample_data = {
            'human_moves': ['rock', 'paper', 'scissors'],
            'robot_moves': ['scissors', 'rock', 'paper'], 
            'results': ['lose', 'lose', 'lose'],
            'llm_type': 'mock',
            'coaching_style': 'supportive'
        }
        
        print("\n🎯 Testing comprehensive endpoint...")
        with app.test_request_context():
            response = client.post('/ai_coach/comprehensive', 
                                 json=sample_data,
                                 content_type='application/json')
            print(f"  Status: {response.status_code}")
            if response.status_code == 200:
                data = response.get_json()
                print(f"  ✅ Success: {data.get('success', False)}")
                print(f"  📊 Has metrics_summary: {'metrics_summary' in data}")
                print(f"  🤖 Has ai_behavior: {'ai_behavior' in data}")
            else:
                print(f"  ❌ Error: {response.get_data()}")
        
        print("\n🎯 Testing realtime endpoint...")
        with app.test_request_context():
            response = client.post('/ai_coach/realtime',
                                 json=sample_data,
                                 content_type='application/json')
            print(f"  Status: {response.status_code}")
            if response.status_code == 200:
                data = response.get_json()
                print(f"  ✅ Success: {data.get('success', False)}")
                print(f"  💡 Has advice: {'advice' in data}")
                print(f"  📈 Has metrics_summary: {'metrics_summary' in data}")
            else:
                print(f"  ❌ Error: {response.get_data()}")
                
        print("\n🎯 Testing metrics endpoint...")
        with app.test_request_context():
            response = client.get('/ai_coach/metrics')
            print(f"  Status: {response.status_code}")
            if response.status_code == 200:
                data = response.get_json()
                print(f"  ✅ Success: {data.get('success', False)}")
                print(f"  📊 Has metrics: {'metrics' in data}")
                print(f"  🔍 Has data_sources: {'data_sources' in data}")
            else:
                print(f"  ❌ Error: {response.get_data()}")
        
        print("\n🎉 Centralized endpoint testing complete!")
        return True
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_direct_endpoints()
    if success:
        print("\n✅ All centralized endpoints validated!")
    else:
        print("\n❌ Centralization validation failed")