#!/usr/bin/env python3
"""
Web Interface Test for Phase 3 Features
Tests the Flask app endpoints and functionality
"""

import sys
import os
import requests
import json
import time
import subprocess
import signal
from threading import Timer

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class FlaskAppTester:
    def __init__(self, base_url="http://127.0.0.1:5000"):
        self.base_url = base_url
        self.server_process = None
    
    def start_server(self, timeout=10):
        """Start the Flask server for testing"""
        try:
            # Start Flask app in background
            webapp_dir = os.path.join(os.path.dirname(__file__), 'webapp')
            cmd = [sys.executable, 'app.py']
            
            self.server_process = subprocess.Popen(
                cmd,
                cwd=webapp_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            
            # Wait for server to start
            for i in range(timeout):
                try:
                    response = requests.get(self.base_url, timeout=1)
                    if response.status_code == 200:
                        print(f"✅ Flask server started successfully on {self.base_url}")
                        return True
                except requests.exceptions.RequestException:
                    time.sleep(1)
            
            print("❌ Failed to start Flask server within timeout")
            return False
            
        except Exception as e:
            print(f"❌ Error starting Flask server: {e}")
            return False
    
    def stop_server(self):
        """Stop the Flask server"""
        if self.server_process:
            try:
                # Kill process group to ensure all children are terminated
                os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
                self.server_process.wait(timeout=5)
                print("✅ Flask server stopped")
            except Exception as e:
                print(f"⚠️  Error stopping Flask server: {e}")
    
    def test_main_page(self):
        """Test that the main page loads correctly"""
        try:
            response = requests.get(self.base_url, timeout=5)
            
            if response.status_code == 200:
                content = response.text
                required_elements = [
                    "Paper Scissor Stone",
                    "difficulty-select",
                    "strategy-select", 
                    "personality-select",
                    "game-stats"
                ]
                
                missing = [elem for elem in required_elements if elem not in content]
                
                if not missing:
                    print("✅ Main page loads with all required elements")
                    return True
                else:
                    print(f"❌ Main page missing elements: {missing}")
                    return False
            else:
                print(f"❌ Main page returned status code: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error testing main page: {e}")
            return False
    
    def test_play_endpoint(self):
        """Test the play endpoint with different strategies"""
        try:
            test_cases = [
                {
                    "name": "Enhanced + To Win + Aggressive",
                    "payload": {
                        "move": "paper",
                        "difficulty": "enhanced",
                        "strategy": "to_win",
                        "personality": "aggressive",
                        "multiplayer": False
                    }
                },
                {
                    "name": "Markov + Not to Lose + Defensive",
                    "payload": {
                        "move": "stone",
                        "difficulty": "markov",
                        "strategy": "not_to_lose",
                        "personality": "defensive",
                        "multiplayer": False
                    }
                },
                {
                    "name": "Frequency + Balanced + Neutral",
                    "payload": {
                        "move": "scissor",
                        "difficulty": "frequency",
                        "strategy": "balanced",
                        "personality": "neutral",
                        "multiplayer": False
                    }
                }
            ]
            
            all_passed = True
            
            for test_case in test_cases:
                response = requests.post(f"{self.base_url}/play", json=test_case["payload"], timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    required_fields = ["robot_move", "result", "stats", "difficulty", "strategy_preference", "personality"]
                    
                    missing_fields = [field for field in required_fields if field not in data]
                    
                    if not missing_fields:
                        print(f"✅ {test_case['name']}: Success")
                    else:
                        print(f"❌ {test_case['name']}: Missing fields {missing_fields}")
                        all_passed = False
                else:
                    print(f"❌ {test_case['name']}: Status code {response.status_code}")
                    all_passed = False
            
            return all_passed
            
        except Exception as e:
            print(f"❌ Error testing play endpoint: {e}")
            return False
    
    def test_coaching_endpoint(self):
        """Test the coaching endpoint"""
        try:
            response = requests.get(f"{self.base_url}/coaching", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                if "tips" in data and "patterns" in data:
                    print("✅ Coaching endpoint returns tips and patterns")
                    return True
                else:
                    print("❌ Coaching endpoint missing required fields")
                    return False
            else:
                print(f"❌ Coaching endpoint returned status code: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error testing coaching endpoint: {e}")
            return False
    
    def test_tournament_endpoints(self):
        """Test tournament system endpoints"""
        try:
            # Test tournament dashboard
            response = requests.get(f"{self.base_url}/tournament", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ["leaderboard", "total_players", "total_matches"]
                
                missing_fields = [field for field in required_fields if field not in data]
                
                if not missing_fields:
                    print("✅ Tournament dashboard loads correctly")
                    
                    # Test player creation (unique name)
                    test_player_name = f"TestPlayer_{int(time.time())}"
                    player_payload = {"name": test_player_name}
                    
                    player_response = requests.post(f"{self.base_url}/tournament/player", json=player_payload, timeout=5)
                    
                    if player_response.status_code in [200, 400]:  # 400 if player exists
                        print("✅ Tournament player creation endpoint working")
                        return True
                    else:
                        print(f"❌ Tournament player creation failed: {player_response.status_code}")
                        return False
                else:
                    print(f"❌ Tournament dashboard missing fields: {missing_fields}")
                    return False
            else:
                print(f"❌ Tournament dashboard returned status code: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error testing tournament endpoints: {e}")
            return False
    
    def test_analytics_export(self):
        """Test analytics export functionality"""
        try:
            response = requests.get(f"{self.base_url}/analytics/export?format=json", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                if "stats" in data:
                    print("✅ Analytics export working")
                    return True
                else:
                    print("❌ Analytics export missing stats field")
                    return False
            else:
                print(f"❌ Analytics export returned status code: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error testing analytics export: {e}")
            return False
    
    def run_all_tests(self):
        """Run all web interface tests"""
        print("🌐 PHASE 3 WEB INTERFACE TESTS")
        print("=" * 40)
        
        if not self.start_server():
            print("❌ Cannot start Flask server, skipping web tests")
            return {}
        
        try:
            tests = [
                ("Main Page Load", self.test_main_page),
                ("Play Endpoint", self.test_play_endpoint),
                ("Coaching Endpoint", self.test_coaching_endpoint),
                ("Tournament Endpoints", self.test_tournament_endpoints),
                ("Analytics Export", self.test_analytics_export),
            ]
            
            results = {}
            
            for test_name, test_func in tests:
                print(f"\n🔍 Testing {test_name}...")
                try:
                    result = test_func()
                    results[test_name] = result
                except Exception as e:
                    print(f"❌ {test_name} failed with exception: {e}")
                    results[test_name] = False
            
            return results
            
        finally:
            self.stop_server()

def main():
    """Main test function"""
    tester = FlaskAppTester()
    results = tester.run_all_tests()
    
    if results:
        print("\n" + "=" * 40)
        print("🎯 WEB TESTS SUMMARY")
        print("=" * 40)
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} {test_name}")
        
        success_rate = (passed / total * 100) if total > 0 else 0
        print(f"\n📊 Web Tests Success Rate: {passed}/{total} ({success_rate:.1f}%)")
        
        if success_rate >= 90:
            print("🎉 Web interface is working excellently!")
        elif success_rate >= 75:
            print("✅ Web interface is working well!")
        else:
            print("⚠️  Web interface has some issues.")

if __name__ == "__main__":
    main()