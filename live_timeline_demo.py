#!/usr/bin/env python3
"""
Live test of strategy timeline with dramatic pattern changes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from webapp.app import app, game_state, change_detector

def live_timeline_demo():
    """Create dramatic strategy changes to test timeline visualization"""
    print("🎭 Live Strategy Timeline Demo")
    print("=" * 40)
    
    with app.test_client() as client:
        # Reset
        client.post('/reset')
        print("Starting fresh game...")
        
        # Phase 1: Pure stone repetition
        print("\n📍 Phase 1: Pure Repeater (stone only)")
        for i in range(8):
            response = client.post('/play', json={'move': 'stone', 'difficulty': 'enhanced'})
            data = response.get_json()
            print(f"  Move {i+1}: stone -> {data.get('current_strategy', 'unknown')}")
        
        print(f"  Change points so far: {len(data.get('change_points', []))}")
        
        # Phase 2: Switch to alternating pattern  
        print("\n📍 Phase 2: Switch to Alternating Pattern")
        alt_pattern = ['stone', 'paper'] * 6
        for i, move in enumerate(alt_pattern):
            response = client.post('/play', json={'move': move, 'difficulty': 'enhanced'})
            data = response.get_json()
            round_num = 8 + i + 1
            print(f"  Move {round_num}: {move} -> {data.get('current_strategy', 'unknown')}")
            
            if data.get('change_points'):
                for cp in data['change_points'][-1:]:  # Show only latest change
                    print(f"    🔄 CHANGE: {cp.get('description', 'Strategy changed')}")
        
        print(f"  Change points so far: {len(data.get('change_points', []))}")
        
        # Phase 3: Switch to pure cycle
        print("\n📍 Phase 3: Switch to Perfect Cycle")
        cycle_pattern = ['stone', 'paper', 'scissor'] * 4
        for i, move in enumerate(cycle_pattern):
            response = client.post('/play', json={'move': move, 'difficulty': 'enhanced'})
            data = response.get_json()
            round_num = 20 + i + 1
            print(f"  Move {round_num}: {move} -> {data.get('current_strategy', 'unknown')}")
            
            change_points = data.get('change_points', [])
            if len(change_points) > 0:
                latest_changes = change_points[-2:]  # Show recent changes
                for cp in latest_changes:
                    if cp.get('round', 0) >= round_num - 2:  # Only recent changes
                        print(f"    🔄 CHANGE: {cp.get('description', 'Strategy changed')}")
        
        # Final summary
        print(f"\n📊 Final Results:")
        print(f"   - Total rounds: {data['round']}")
        print(f"   - Final strategy: {data.get('current_strategy', 'unknown')}")
        print(f"   - Total change points: {len(data.get('change_points', []))}")
        print(f"   - Enhanced AI accuracy: {data.get('accuracy', {}).get('enhanced', 'N/A')}%")
        
        # Show all detected changes
        all_changes = data.get('change_points', [])
        if all_changes:
            print(f"\n📋 All Detected Strategy Changes:")
            for i, cp in enumerate(all_changes, 1):
                round_num = cp.get('round', cp.get('move_index', 'unknown'))
                description = cp.get('description', 'Strategy change')
                print(f"   {i}. Round {round_num}: {description}")
        
        print(f"\n🌐 View the live timeline at: http://127.0.0.1:5000")
        print(f"💡 The strategy timeline should show:")
        print(f"   - Color-coded strategy segments")
        print(f"   - Red triangle markers for change points")
        print(f"   - Real-time strategy display with colors")

if __name__ == "__main__":
    live_timeline_demo()