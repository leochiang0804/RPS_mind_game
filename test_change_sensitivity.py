#!/usr/bin/env python3
"""
Test chan            if change_points:
                for cp in change_points:
                    round_num = cp.get('round', cp.get('move_index', 'unknown'))
                    description = cp.get('description', 'Strategy change detected')
                    print(f"  🔄 Change detected at move {round_num}: {description}")point detection sensitivity in webapp
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from webapp.app import app, game_state, change_detector

def test_change_detection():
    """Test that change-point detection actually detects strategy changes"""
    print("Testing change-point detection sensitivity...")
    
    with app.test_client() as client:
        # Reset state
        client.post('/reset')
        
        # Phase 1: Pure repeater pattern (stone, stone, stone...)
        print("\nPhase 1: Repeater pattern (stone only)")
        for i in range(8):
            response = client.post('/play', json={'move': 'stone', 'difficulty': 'enhanced'})
            data = response.get_json()
            print(f"Move {i+1}: stone -> {data.get('current_strategy', 'unknown')}")
        
        # Phase 2: Switch to pure cycler (stone->paper->scissor->stone...)
        print("\nPhase 2: Switching to cycler pattern")
        cycle_moves = ['stone', 'paper', 'scissor'] * 4
        for i, move in enumerate(cycle_moves):
            response = client.post('/play', json={'move': move, 'difficulty': 'enhanced'})
            data = response.get_json()
            change_points = data.get('change_points', [])
            
            print(f"Move {9+i}: {move} -> {data.get('current_strategy', 'unknown')} (Changes: {len(change_points)})")
            
            if change_points:
                for cp in change_points:
                    round_num = cp.get('round', cp.get('move_index', 'unknown'))
                    description = cp.get('description', 'Strategy change detected')
                    print(f"  🔄 Change detected at move {round_num}: {description}")
        
        # Phase 3: Switch to anti-repeater (opposite of last move)
        print("\nPhase 3: Switching to anti-repeater pattern")
        last_move = 'scissor'
        anti_moves = []
        move_map = {'stone': 'paper', 'paper': 'scissor', 'scissor': 'stone'}
        
        for i in range(6):
            # Choose opposite of what we played last
            next_move = move_map[last_move]
            anti_moves.append(next_move)
            
            response = client.post('/play', json={'move': next_move, 'difficulty': 'enhanced'})
            data = response.get_json()
            change_points = data.get('change_points', [])
            
            print(f"Move {21+i}: {next_move} -> {data.get('current_strategy', 'unknown')} (Changes: {len(change_points)})")
            
            if change_points:
                for cp in change_points:
                    round_num = cp.get('round', cp.get('move_index', 'unknown'))
                    description = cp.get('description', 'Strategy change detected')
                    print(f"  🔄 Change detected at move {round_num}: {description}")
            
            last_move = next_move
        
        # Get final state
        final_response = client.get('/stats')
        final_data = final_response.get_json()
        
        print(f"\n✅ Test completed!")
        print(f"Final strategy: {game_state.get('current_strategy', 'unknown')}")
        print(f"Total change points: {len(game_state.get('change_points', []))}")
        
        # Print all detected changes
        all_changes = game_state.get('change_points', [])
        if all_changes:
            print("\n📊 All detected strategy changes:")
            for cp in all_changes:
                round_num = cp.get('round', cp.get('move_index', 'unknown'))
                description = cp.get('description', 'Strategy change detected')
                print(f"  Move {round_num}: {description}")
        else:
            print("\n⚠️  No strategy changes detected - consider adjusting sensitivity")

if __name__ == "__main__":
    test_change_detection()