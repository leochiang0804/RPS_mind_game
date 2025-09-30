#!/usr/bin/env python3
"""
Test webapp integration with change-point detection
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import webapp module
from webapp.app import app, game_state, change_detector

def test_webapp_integration():
    """Test that webapp properly integrates change-point detection"""
    print("Testing webapp integration with change-point detection...")
    
    # Create Flask test client
    with app.test_client() as client:
        # Reset state
        response = client.post('/reset')
        if response.status_code != 200:
            print(f"Reset failed with status {response.status_code}")
            print(f"Response: {response.get_data(as_text=True)}")
            return
        print("✓ Reset successful")
        
        # Play a sequence of moves that should create patterns
        pattern_moves = ['stone', 'paper', 'scissor'] * 3  # Repeating cycle
        
        for i, move in enumerate(pattern_moves):
            print(f"Move {i+1}: {move}")
            response = client.post('/play', json={'move': move, 'difficulty': 'enhanced'})
            if response.status_code != 200:
                print(f"Play failed with status {response.status_code}")
                print(f"Response: {response.get_data(as_text=True)}")
                return
            
            data = response.get_json()
            print(f"  Robot move: {data['robot_move']}")
            print(f"  Result: {data['result']}")
            print(f"  Current strategy: {data.get('current_strategy', 'unknown')}")
            print(f"  Change points: {len(data.get('change_points', []))}")
            print(f"  Confidence: {data.get('confidence', 'N/A')}")
            
            # After 5 moves, we should have strategy analysis
            if i >= 4:
                assert 'current_strategy' in data
                assert 'change_points' in data
                print(f"  ✓ Strategy analysis available: {data['current_strategy']}")
            
            print()
        
        # Add a change in strategy - switch to anti-cycle
        anti_cycle_moves = ['scissor', 'stone', 'paper'] * 2
        
        print("Switching to anti-cycle pattern...")
        for i, move in enumerate(anti_cycle_moves):
            print(f"Anti-cycle move {i+1}: {move}")
            response = client.post('/play', json={'move': move, 'difficulty': 'enhanced'})
            if response.status_code != 200:
                print(f"Play failed with status {response.status_code}")
                return
            data = response.get_json()
            
            print(f"  Robot move: {data['robot_move']}")
            print(f"  Current strategy: {data.get('current_strategy', 'unknown')}")
            print(f"  Change points: {len(data.get('change_points', []))}")
            
            if data.get('change_points'):
                for cp in data['change_points']:
                    print(f"    Change at move {cp['move_index']}: {cp['description']}")
            print()
        
        print("✅ Webapp integration test completed successfully!")
        
        # Get final state
        final_response = client.get('/stats')
        final_data = final_response.get_json()
        
        print("\nFinal state:")
        print(f"Total moves: {final_data['round']}")
        print(f"Strategy: {game_state.get('current_strategy', 'unknown')}")
        print(f"Change points detected: {len(game_state.get('change_points', []))}")
        print(f"Accuracy: {final_data.get('accuracy', 'N/A')}")

if __name__ == "__main__":
    test_webapp_integration()