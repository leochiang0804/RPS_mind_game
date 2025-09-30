#!/usr/bin/env python3
"""
Debug change-point detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from change_point_detector import ChangePointDetector

def debug_change_point():
    print("=== Debug Change-Point Detection ===\n")
    
    detector = ChangePointDetector(window_size=6, chi2_threshold=1.0)
    
    # Start with repeater
    print("Phase 1: Repeater pattern")
    for i in range(10):
        detector.add_move('stone')
        if len(detector.move_history) >= 6:
            features = detector.get_recent_features()
            print(f"Move {i+1}: repeat_prob={features['repeat_prob']:.3f}, cycle_score={features['cycle_score']:.3f}")
    
    print(f"Strategy after repeater phase: {detector.get_current_strategy_label()}")
    
    # Shift to cycler
    print("\nPhase 2: Cycle pattern")
    cycle_pattern = ['stone', 'paper', 'scissor'] * 4
    
    for i, move in enumerate(cycle_pattern):
        change = detector.add_move(move)
        features = detector.get_recent_features()
        
        print(f"Move {i+11}: {move} -> repeat_prob={features['repeat_prob']:.3f}, cycle_score={features['cycle_score']:.3f}")
        
        if change:
            print(f"  *** CHANGE DETECTED! Chi2={change['chi2_statistic']:.3f}, Confidence={change['confidence']:.3f}")
            print(f"  Description: {change['description']}")
            break
        
        if i >= 8:  # Stop after a few cycles
            break
    
    print(f"Final strategy: {detector.get_current_strategy_label()}")
    
    print("\nAll change points:")
    for cp in detector.get_all_change_points():
        print(f"  Round {cp['round']}: {cp['description']} (chi2={cp['chi2_statistic']:.3f})")

if __name__ == "__main__":
    debug_change_point()