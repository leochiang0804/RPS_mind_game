#!/usr/bin/env python3
"""
Detailed analysis of pattern detection effectiveness
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rps_ai_system import RPSAISystem
from game_context import set_opponent_parameters

def get_counter_move(move):
    """Get the move that beats the given move"""
    counters = {'rock': 'paper', 'paper': 'scissors', 'scissors': 'rock'}
    return counters[move]

def analyze_pattern_counter_effectiveness():
    print("🔍 Analyzing Pattern Counter Effectiveness")
    print("=" * 60)
    
    # Initialize AI system
    ai_system = RPSAISystem()
    set_opponent_parameters('rookie', 'to_win', 'neutral')
    ai_system.set_opponent('rookie', 'to_win', 'neutral')
    
    # Test the simple alternating pattern that was problematic
    pattern = ["rock", "paper"] * 10  # rock-paper-rock-paper...
    print(f"🎯 Testing pattern: {pattern[:8]}... (continues)")
    
    ai_system.move_history = []
    human_wins = 0
    ai_wins = 0
    ties = 0
    
    print(f"\n{'Move':<4} {'Human':<8} {'AI':<8} {'Expected':<10} {'Result':<6} {'Pattern?':<9} {'Predicted':<10}")
    print("-" * 70)
    
    for i, human_move in enumerate(pattern):
        # Get AI prediction
        probabilities, ai_move, metadata = ai_system.predict_next_move()
        
        # Determine the winner
        if human_move == ai_move:
            result = "TIE"
            ties += 1
        elif get_counter_move(human_move) == ai_move:
            result = "AI"
            ai_wins += 1
        else:
            result = "HUMAN"
            human_wins += 1
        
        # Check pattern detection info
        markov_metadata = metadata.get('markov_prediction', {}).get('metadata', {})
        pattern_detected = 'pattern_counter' in markov_metadata.get('method', '')
        predicted_move = markov_metadata.get('predicted_human_move', 'unknown')
        
        # What should the AI play to counter this human move?
        expected_ai_move = get_counter_move(human_move)
        
        print(f"{i+1:<4} {human_move:<8} {ai_move:<8} {expected_ai_move:<10} {result:<6} {'Yes' if pattern_detected else 'No':<9} {predicted_move:<10}")
        
        # Update history
        ai_system.move_history.append(human_move)
    
    # Summary
    total_moves = len(pattern)
    human_win_rate = human_wins / total_moves
    ai_win_rate = ai_wins / total_moves
    tie_rate = ties / total_moves
    
    print("\n" + "=" * 60)
    print("📊 RESULTS SUMMARY")
    print("=" * 60)
    print(f"Human wins: {human_wins}/{total_moves} ({human_win_rate:.1%})")
    print(f"AI wins:    {ai_wins}/{total_moves} ({ai_win_rate:.1%})")
    print(f"Ties:       {ties}/{total_moves} ({tie_rate:.1%})")
    
    # Effectiveness assessment
    if human_win_rate > 0.6:
        status = "🚨 HIGHLY VULNERABLE - Pattern detection not effective"
    elif human_win_rate > 0.4:
        status = "⚠️  MODERATELY VULNERABLE - Some improvement needed"
    else:
        status = "✅ WELL DEFENDED - Pattern detection working"
    
    print(f"\nAssessment: {status}")
    
    return human_win_rate

if __name__ == "__main__":
    analyze_pattern_counter_effectiveness()