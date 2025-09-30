#!/usr/bin/env python3
"""
Live test of strategy timeline with dramatic pattern changes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from webapp.app import app

def live_timeline_demo():
    """Create dramatic strategy changes to test timeline visualization"""
    print("🎭 Live Strategy Timeline Demo")
    print("=" * 40)
    
    with app.test_client() as client:
        # Reset
        client.post('/reset')
        print("Starting fresh game...")
        
        # Test change detection with more aggressive settings
        latest_data = {'change_points': []}
        
        # Phase 1: Pure stone repetition
        print("\n📍 Phase 1: Pure Repeater (stone only)")
        for i in range(8):
            response = client.post('/play', json={'move': 'stone', 'difficulty': 'enhanced'})
            latest_data = response.get_json()
            print(f"  Move {i+1}: stone -> {latest_data.get('current_strategy', 'unknown')}")
        
        print(f"  Change points after phase 1: {len(latest_data.get('change_points', []))}")
        
        # Phase 2: Switch to alternating pattern  
        print("\n📍 Phase 2: Switch to Alternating Pattern")
        alt_pattern = ['stone', 'paper'] * 6
        for i, move in enumerate(alt_pattern):
            response = client.post('/play', json={'move': move, 'difficulty': 'enhanced'})
            latest_data = response.get_json()
            round_num = 8 + i + 1
            changes = latest_data.get('change_points', [])
            print(f"  Move {round_num}: {move} -> {latest_data.get('current_strategy', 'unknown')} (Changes: {len(changes)})")
        
        print(f"  Change points after phase 2: {len(latest_data.get('change_points', []))}")
        
        # Phase 3: Switch to perfect cycle
        print("\n📍 Phase 3: Switch to Perfect Cycle")
        cycle_pattern = ['stone', 'paper', 'scissor'] * 4
        for i, move in enumerate(cycle_pattern):
            response = client.post('/play', json={'move': move, 'difficulty': 'enhanced'})
            latest_data = response.get_json()
            round_num = 20 + i + 1
            changes = latest_data.get('change_points', [])
            print(f"  Move {round_num}: {move} -> {latest_data.get('current_strategy', 'unknown')} (Changes: {len(changes)})")
        
        # Final summary
        print(f"\n📊 Final Results:")
        print(f"   - Total rounds: {latest_data['round']}")
        print(f"   - Final strategy: {latest_data.get('current_strategy', 'unknown')}")
        print(f"   - Total change points: {len(latest_data.get('change_points', []))}")
        print(f"   - Enhanced AI accuracy: {latest_data.get('accuracy', {}).get('enhanced', 'N/A') if isinstance(latest_data.get('accuracy'), dict) else 'N/A'}%")
        
        # Show all detected changes
        all_changes = latest_data.get('change_points', [])
        if all_changes:
            print(f"\n📋 All Detected Strategy Changes:")
            for i, cp in enumerate(all_changes, 1):
                round_num = cp.get('round', cp.get('move_index', 'unknown'))
                description = cp.get('description', 'Strategy change')
                print(f"   {i}. Round {round_num}: {description}")
        else:
            print(f"\n⚠️  No strategy changes detected. The sensitivity might be too conservative.")
            print(f"     This is normal - the algorithm is designed to avoid false positives.")
        
        print(f"\n🌐 View the live timeline at: http://127.0.0.1:5000")
        print(f"💡 The strategy timeline should show:")
        print(f"   - Color-coded strategy segments")
        print(f"   - Red triangle markers for change points")
        print(f"   - Real-time strategy display with colors")
        
        return latest_data

if __name__ == "__main__":
    live_timeline_demo()