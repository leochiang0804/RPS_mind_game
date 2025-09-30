#!/usr/bin/env python3
"""
Debug change-point detection to understand why it's not triggering
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from change_point_detector import ChangePointDetector

def debug_change_detection():
    """Debug the change detection algorithm directly"""
    print("Debugging change-point detection algorithm...")
    
    # Create detector with more sensitive settings
    detector = ChangePointDetector(window_size=5, chi2_threshold=2.0, min_segment_length=3)
    
    # Phase 1: Pure repeater
    print("\nPhase 1: Pure repeater (stone only)")
    moves_phase1 = ['stone'] * 6
    for i, move in enumerate(moves_phase1):
        result = detector.add_move(move)
        features = detector.get_recent_features()
        strategy = detector.get_current_strategy_label()
        
        print(f"Move {i+1}: {move}")
        print(f"  Strategy: {strategy}")
        print(f"  Features: repeat_prob={features.get('repeat_prob', 0):.3f}, "
              f"cycle_score={features.get('cycle_score', 0):.3f}, "
              f"entropy={features.get('entropy', 0):.3f}")
        
        if result:
            print(f"  🔄 CHANGE DETECTED: {result['description']}")
        else:
            print(f"  No change detected")
        print()
    
    # Phase 2: Switch to cycle
    print("Phase 2: Switch to cycle pattern")
    moves_phase2 = ['stone', 'paper', 'scissor'] * 4
    for i, move in enumerate(moves_phase2):
        result = detector.add_move(move)
        features = detector.get_recent_features()
        strategy = detector.get_current_strategy_label()
        
        move_num = len(moves_phase1) + i + 1
        print(f"Move {move_num}: {move}")
        print(f"  Strategy: {strategy}")
        print(f"  Features: repeat_prob={features.get('repeat_prob', 0):.3f}, "
              f"cycle_score={features.get('cycle_score', 0):.3f}, "
              f"entropy={features.get('entropy', 0):.3f}")
        
        if result:
            print(f"  🔄 CHANGE DETECTED: {result['description']}")
        else:
            print(f"  No change detected")
        print()
    
    # Show all change points
    all_changes = detector.get_all_change_points()
    print(f"\n📊 Summary: {len(all_changes)} change points detected")
    for i, cp in enumerate(all_changes):
        print(f"  Change {i+1}: {cp.get('description', 'No description')}")
        print(f"    Move index: {cp.get('move_index', 'Unknown')}")
        print(f"    Features: {cp.get('new_features', {})}")
    
    # Export full analysis
    analysis = detector.export_analysis()
    print(f"\nDetector analysis:")
    print(f"  Total moves: {len(analysis['move_history'])}")
    print(f"  Change points: {len(analysis['change_points'])}")
    print(f"  Current strategy: {analysis['current_strategy']}")

if __name__ == "__main__":
    debug_change_detection()