#!/usr/bin/env python3
"""
Confidence Debug Script

Analyzes why Challenger has unexpectedly low confidence compared to Rookie/Master.
"""

import sys
import os
import random
import statistics

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rps_ai_system import initialize_ai_system, get_ai_system
from game_context import set_opponent_parameters, get_ai_prediction, update_ai_with_result, reset_ai_system

def test_confidence_by_difficulty():
    """Test confidence levels for each difficulty on identical patterns"""
    
    print("🔍 CONFIDENCE DEBUG TEST")
    print("=" * 60)
    
    # Initialize AI system
    initialize_ai_system('challenger', 'to_win', 'neutral')
    
    difficulties = ['rookie', 'challenger', 'master']
    
    # Test pattern: simple alternating pattern that should be detectable
    test_pattern = ['rock', 'paper', 'rock', 'paper', 'rock', 'paper', 'rock', 'paper']
    
    print("Testing identical pattern against all difficulties:")
    print(f"Pattern: {test_pattern}")
    print()
    
    results = {}
    
    for difficulty in difficulties:
        print(f"🎯 Testing {difficulty.upper()} difficulty...")
        
        # Reset system
        reset_ai_system()
        
        # Set difficulty parameters
        success = set_opponent_parameters(difficulty, 'to_win', 'neutral')
        if not success:
            print(f"❌ Failed to set {difficulty} parameters")
            continue
        
        confidences = []
        predictions = []
        
        # Simulate the pattern
        for i, human_move in enumerate(test_pattern):
            # Get AI prediction before seeing human move
            session_data = {
                'human_moves': test_pattern[:i],  # Previous moves
                'results': [],
                'ai_difficulty': difficulty,
                'strategy_preference': 'to_win',
                'personality': 'neutral'
            }
            
            if i > 0:  # Only predict after seeing at least one move
                prediction_data = get_ai_prediction(session_data)
                ai_move = prediction_data.get('ai_move', 'rock')
                confidence = prediction_data.get('confidence', 0.33)
                
                confidences.append(confidence)
                predictions.append((ai_move, confidence, human_move))
                
                print(f"  Move {i}: Human={human_move}, AI={ai_move}, Confidence={confidence:.3f}")
                
                # Update AI with actual result
                result = 'tie' if human_move == ai_move else ('human' if 
                    (human_move == 'rock' and ai_move == 'scissors') or
                    (human_move == 'paper' and ai_move == 'rock') or  
                    (human_move == 'scissors' and ai_move == 'paper') else 'robot')
                
                update_ai_with_result(human_move, ai_move)
        
        if confidences:
            avg_confidence = statistics.mean(confidences)
            max_confidence = max(confidences)
            min_confidence = min(confidences)
            
            results[difficulty] = {
                'avg_confidence': avg_confidence,
                'max_confidence': max_confidence,
                'min_confidence': min_confidence,
                'all_confidences': confidences,
                'predictions': predictions
            }
            
            print(f"  📊 {difficulty.upper()} Results:")
            print(f"     Avg Confidence: {avg_confidence:.3f}")
            print(f"     Range: {min_confidence:.3f} - {max_confidence:.3f}")
            print(f"     Pattern Detection: {'✅ Working' if max_confidence > 0.5 else '❌ Poor'}")
        
        print()
    
    # Compare results
    print("🔍 COMPARISON ANALYSIS:")
    print("=" * 40)
    
    for difficulty in difficulties:
        if difficulty in results:
            r = results[difficulty]
            print(f"{difficulty.upper()}: {r['avg_confidence']:.3f} avg confidence")
    
    print("\n📊 EXPECTED vs ACTUAL:")
    print("Expected: Rookie < Challenger < Master")
    
    if 'rookie' in results and 'challenger' in results and 'master' in results:
        rookie_conf = results['rookie']['avg_confidence']
        challenger_conf = results['challenger']['avg_confidence']
        master_conf = results['master']['avg_confidence']
        
        print(f"Actual:   Rookie {rookie_conf:.3f} ? Challenger {challenger_conf:.3f} ? Master {master_conf:.3f}")
        
        if rookie_conf < challenger_conf < master_conf:
            print("✅ CORRECT: Confidence increases with difficulty")
        elif master_conf > max(rookie_conf, challenger_conf):
            print("⚠️ PARTIAL: Master highest, but Rookie/Challenger order wrong")
        else:
            print("❌ BROKEN: Master not highest")
            
        if challenger_conf < rookie_conf:
            print("🚨 CRITICAL: Challenger has LOWER confidence than Rookie!")
            print("   This suggests Challenger thresholds are too strict.")

def test_pattern_detection_thresholds():
    """Test if pattern detection thresholds are working correctly"""
    
    print("\n🔍 PATTERN DETECTION THRESHOLD TEST")
    print("=" * 60)
    
    # Test different strength patterns
    patterns = {
        'very_strong': ['rock'] * 8,  # Extremely predictable
        'strong': ['rock', 'paper'] * 4,  # Simple alternating
        'medium': ['rock', 'paper', 'scissors'] * 3,  # Cycle
        'weak': ['rock', 'paper', 'rock', 'scissors', 'paper', 'rock', 'scissors', 'paper']  # Irregular
    }
    
    difficulties = ['rookie', 'challenger', 'master']
    
    for pattern_name, pattern in patterns.items():
        print(f"\n🎯 Testing {pattern_name.upper()} pattern: {pattern[:6]}...")
        
        pattern_results = {}
        
        for difficulty in difficulties:
            reset_ai_system()
            set_opponent_parameters(difficulty, 'to_win', 'neutral')
            
            confidences = []
            
            for i in range(1, len(pattern)):
                session_data = {
                    'human_moves': pattern[:i],
                    'results': [],
                    'ai_difficulty': difficulty,
                    'strategy_preference': 'to_win',
                    'personality': 'neutral'
                }
                
                prediction_data = get_ai_prediction(session_data)
                confidence = prediction_data.get('confidence', 0.33)
                confidences.append(confidence)
                
                # Update AI
                human_move = pattern[i-1]
                ai_move = prediction_data.get('ai_move', 'rock')
                update_ai_with_result(human_move, ai_move)
            
            avg_conf = statistics.mean(confidences) if confidences else 0.0
            max_conf = max(confidences) if confidences else 0.0
            pattern_results[difficulty] = {'avg': avg_conf, 'max': max_conf}
            
            print(f"  {difficulty.upper()}: {avg_conf:.3f} avg, {max_conf:.3f} max")
        
        # Check if progression is correct for this pattern
        if all(d in pattern_results for d in difficulties):
            rookie_avg = pattern_results['rookie']['avg']
            challenger_avg = pattern_results['challenger']['avg']
            master_avg = pattern_results['master']['avg']
            
            if rookie_avg < challenger_avg < master_avg:
                print(f"  ✅ {pattern_name}: Correct progression")
            else:
                print(f"  ❌ {pattern_name}: Broken progression")

if __name__ == "__main__":
    test_confidence_by_difficulty()
    test_pattern_detection_thresholds()