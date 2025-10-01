#!/usr/bin/env python3
"""Debug script to test real LLM response structure"""

import requests
import json

def test_real_llm_response():
    """Test what the real LLM endpoint actually returns"""
    
    base_url = "http://127.0.0.1:5050"
    
    # Test real-time coaching with Real LLM
    print("🔍 Testing Real LLM real-time response...")
    
    realtime_data = {
        "llm_type": "real",
        "coaching_style": "easy",
        "include_metrics": True,
        "include_metadata": True
    }
    
    try:
        response = requests.post(f"{base_url}/ai_coach/realtime", json=realtime_data)
        if response.status_code == 200:
            data = response.json()
            print("\n✅ Real LLM real-time response structure:")
            print(f"Success: {data.get('success')}")
            print(f"Raw response: {data.get('raw_response', 'NOT FOUND')[:200]}")
            print(f"Natural language full: {data.get('natural_language_full', 'NOT FOUND')[:200]}")
            print(f"Coaching personality: {data.get('coaching_personality', 'NOT FOUND')}")
            
            if 'advice' in data:
                advice = data['advice']
                print(f"\nAdvice structure:")
                print(f"- tips: {advice.get('tips', 'NOT FOUND')}")
                print(f"- raw_response in advice: {advice.get('raw_response', 'NOT FOUND')[:100]}")
                print(f"- natural_language_full in advice: {advice.get('natural_language_full', 'NOT FOUND')[:100]}")
            
            print(f"\nComplete response keys: {list(data.keys())}")
        else:
            print(f"❌ Request failed with status {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing real LLM: {e}")
    
    # Test comprehensive analysis
    print("\n" + "="*60)
    print("🔍 Testing Real LLM comprehensive analysis...")
    
    comprehensive_data = {
        "llm_type": "real",
        "coaching_style": "easy"
    }
    
    try:
        response = requests.post(f"{base_url}/ai_coach/comprehensive", json=comprehensive_data)
        if response.status_code == 200:
            data = response.json()
            print("\n✅ Real LLM comprehensive response structure:")
            print(f"Success: {data.get('success')}")
            print(f"Raw response: {data.get('raw_response', 'NOT FOUND')[:200]}")
            print(f"Natural language full: {data.get('natural_language_full', 'NOT FOUND')[:200]}")
            
            if 'analysis' in data:
                analysis = data['analysis']
                print(f"\nAnalysis structure:")
                print(f"- insights: {analysis.get('insights', 'NOT FOUND')}")
                print(f"- educational_content: {analysis.get('educational_content', 'NOT FOUND')}")
                print(f"- behavioral_analysis: {analysis.get('behavioral_analysis', 'NOT FOUND')}")
                print(f"- raw_response in analysis: {analysis.get('raw_response', 'NOT FOUND')[:100]}")
            
            print(f"\nComplete response keys: {list(data.keys())}")
        else:
            print(f"❌ Request failed with status {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing comprehensive analysis: {e}")

if __name__ == "__main__":
    test_real_llm_response()