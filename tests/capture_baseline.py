#!/usr/bin/env python3
"""
Baseline Snapshot Capture Tool
Captures current behavior of AI coach endpoints and game insights for regression testing
"""

import sys
import os
# Add the parent directory to the path to avoid import conflicts
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import json
import time
from typing import Dict, List, Any
from datetime import datetime

class BaselineCapture:
    """Captures baseline behavior for regression testing"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:5050"):
        self.base_url = base_url
        self.session = requests.Session()
        self.snapshots = {}
        self.timestamp = datetime.now().isoformat()
        
    def reset_session(self):
        """Reset session state for clean scenario runs"""
        self.session = requests.Session()
        
    def play_sequence(self, moves: List[str], scenario_name: str, **game_settings) -> Dict[str, Any]:
        """
        Play a sequence of moves and capture the resulting state
        
        Args:
            moves: List of moves to play ['paper', 'stone', 'scissor']
            scenario_name: Name for this scenario
            **game_settings: Additional game settings (difficulty, personality, etc.)
        """
        print(f"🎮 Playing scenario: {scenario_name}")
        print(f"   Moves: {moves}")
        print(f"   Settings: {game_settings}")
        
        # Reset for clean state
        self.reset_session()
        
        # Play the sequence
        game_responses = []
        for i, move in enumerate(moves):
            play_data = {'move': move}
            play_data.update(game_settings)
            
            print(f"   Round {i+1}: {move}", end="")
            response = self.session.post(f"{self.base_url}/play", json=play_data)
            
            if response.status_code == 200:
                game_responses.append(response.json())
                print(" ✓")
            else:
                print(f" ✗ (Status: {response.status_code})")
                print(f"     Error: {response.text}")
                # Return empty dict instead of None for type safety
                return {
                    'scenario_name': scenario_name,
                    'moves': moves,
                    'game_settings': game_settings,
                    'error': f"Failed at round {i+1}: {response.text}",
                    'timestamp': self.timestamp
                }
                
        # Capture final game state from last response
        final_game_state = game_responses[-1] if game_responses else {}
        
        return {
            'scenario_name': scenario_name,
            'moves': moves,
            'game_settings': game_settings,
            'game_responses': game_responses,
            'final_state': final_game_state,
            'timestamp': self.timestamp
        }
    
    def capture_ai_coach_endpoints(self, scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        """Capture all AI coach endpoint responses for the current session state"""
        print(f"🤖 Capturing AI coach endpoints for {scenario_data['scenario_name']}")
        
        endpoints = {}
        
        # AI Coach Status
        try:
            response = self.session.get(f"{self.base_url}/ai_coach/status")
            endpoints['status'] = {
                'status_code': response.status_code,
                'data': response.json() if response.status_code == 200 else response.text
            }
            print("   ✓ /ai_coach/status")
        except Exception as e:
            endpoints['status'] = {'error': str(e)}
            print(f"   ✗ /ai_coach/status - {e}")
            
        # AI Coach Realtime
        try:
            response = self.session.post(f"{self.base_url}/ai_coach/realtime", 
                                       json={'type': 'realtime'})
            endpoints['realtime'] = {
                'status_code': response.status_code,
                'data': response.json() if response.status_code == 200 else response.text
            }
            print("   ✓ /ai_coach/realtime")
        except Exception as e:
            endpoints['realtime'] = {'error': str(e)}
            print(f"   ✗ /ai_coach/realtime - {e}")
            
        # AI Coach Comprehensive
        try:
            response = self.session.post(f"{self.base_url}/ai_coach/comprehensive", 
                                       json={'type': 'comprehensive'})
            endpoints['comprehensive'] = {
                'status_code': response.status_code,
                'data': response.json() if response.status_code == 200 else response.text
            }
            print("   ✓ /ai_coach/comprehensive")
        except Exception as e:
            endpoints['comprehensive'] = {'error': str(e)}
            print(f"   ✗ /ai_coach/comprehensive - {e}")
            
        # AI Coach Metrics
        try:
            response = self.session.get(f"{self.base_url}/ai_coach/metrics")
            endpoints['metrics'] = {
                'status_code': response.status_code,
                'data': response.json() if response.status_code == 200 else response.text
            }
            print("   ✓ /ai_coach/metrics")
        except Exception as e:
            endpoints['metrics'] = {'error': str(e)}
            print(f"   ✗ /ai_coach/metrics - {e}")
            
        return endpoints
    
    def save_snapshot(self, scenario_data: Dict[str, Any], ai_coach_data: Dict[str, Any]):
        """Save captured data as JSON snapshot"""
        snapshot = {
            'capture_metadata': {
                'timestamp': self.timestamp,
                'base_url': self.base_url,
                'capture_version': '1.0'
            },
            'scenario': scenario_data,
            'ai_coach_endpoints': ai_coach_data
        }
        
        scenario_name = scenario_data['scenario_name']
        filename = f"baseline_{scenario_name}_{self.timestamp.replace(':', '-').split('.')[0]}.json"
        filepath = f"tests/regression/{filename}"
        
        with open(filepath, 'w') as f:
            json.dump(snapshot, f, indent=2, default=str)
            
        print(f"💾 Saved snapshot: {filepath}")
        return filepath
    
    def run_all_scenarios(self) -> List[str]:
        """Run all predefined scenarios and capture baselines"""
        scenarios = [
            {
                'name': 'short_match_human_wins',
                'moves': ['paper', 'stone', 'paper', 'scissor'],
                'settings': {'difficulty': 'easy'}
            },
            {
                'name': 'full_match_human_leads',
                'moves': ['paper', 'stone', 'paper', 'stone', 'paper', 'stone', 'paper', 'stone', 'paper', 'stone'],
                'settings': {'difficulty': 'medium', 'personality': 'neutral'}
            },
            {
                'name': 'full_match_ai_leads',
                'moves': ['stone', 'stone', 'stone', 'paper', 'paper', 'paper', 'scissor', 'scissor'],
                'settings': {'difficulty': 'hard', 'personality': 'aggressive'}
            },
            {
                'name': 'draw_heavy_match',
                'moves': ['paper', 'paper', 'stone', 'stone', 'scissor', 'scissor'],
                'settings': {'difficulty': 'medium', 'personality': 'defensive'}
            },
            {
                'name': 'strategy_change_mid_game',
                'moves': ['paper', 'paper', 'stone', 'scissor', 'paper', 'stone'],
                'settings': {'difficulty': 'easy', 'strategy': 'balanced'}
            }
        ]
        
        snapshot_files = []
        
        print(f"🎯 Running {len(scenarios)} baseline scenarios...")
        print("=" * 60)
        
        for scenario in scenarios:
            try:
                # Play the scenario
                scenario_data = self.play_sequence(
                    moves=scenario['moves'],
                    scenario_name=scenario['name'],
                    **scenario['settings']
                )
                
                if scenario_data:
                    # Capture AI coach responses
                    ai_coach_data = self.capture_ai_coach_endpoints(scenario_data)
                    
                    # Save snapshot
                    snapshot_file = self.save_snapshot(scenario_data, ai_coach_data)
                    snapshot_files.append(snapshot_file)
                    
                print("=" * 60)
                time.sleep(1)  # Brief pause between scenarios
                
            except Exception as e:
                print(f"❌ Failed to capture scenario {scenario['name']}: {e}")
                
        print(f"✅ Captured {len(snapshot_files)} baseline snapshots")
        return snapshot_files


def main():
    """Run baseline capture"""
    print("🎯 Baseline Snapshot Capture Tool")
    print("This will capture current behavior for regression testing")
    print()
    
    # Check if server is running
    try:
        response = requests.get("http://127.0.0.1:5050/")
        if response.status_code not in [200, 403]:  # 403 is OK, means server is running
            print("❌ Server not responding properly. Please start the webapp first:")
            print("   python webapp/app.py")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Please start the webapp first:")
        print("   python webapp/app.py")
        return
        
    print("✅ Server is running, starting capture...")
    print()
    
    # Run capture
    capture = BaselineCapture()
    snapshot_files = capture.run_all_scenarios()
    
    print()
    print("📊 Baseline Capture Summary:")
    print(f"   Total snapshots: {len(snapshot_files)}")
    print(f"   Saved to: tests/regression/")
    print()
    print("Next steps:")
    print("1. Review the captured snapshots")
    print("2. Implement game_context.py")
    print("3. Run regression tests after refactoring")


if __name__ == "__main__":
    main()