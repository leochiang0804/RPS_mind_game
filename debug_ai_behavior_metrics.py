#!/usr/bin/env python3
"""Debug script to test AI behavior metrics"""

import json
from ai_coach_metrics import ComprehensiveMetricsAggregator

def test_ai_behavior_metrics():
    """Test the AI behavior metrics with sample data"""
    
    # Create sample game state with the data structure we expect
    sample_game_state = {
        'human_moves': ['rock', 'paper', 'scissors', 'rock', 'paper'],
        'robot_moves': ['paper', 'scissors', 'rock', 'paper', 'scissors'],
        'results': ['robot', 'robot', 'robot', 'robot', 'robot'],
        'round': 5,
        'current_strategy': 'enhanced',
        'model_predictions_history': {
            'random': ['rock', 'paper', 'scissors', 'rock', 'paper'],
            'frequency': ['rock', 'rock', 'paper', 'rock', 'rock'],
            'markov': ['paper', 'scissors', 'rock', 'paper', 'scissors'],
            'enhanced': ['paper', 'scissors', 'rock', 'paper', 'scissors'],
            'lstm': ['rock', 'paper', 'rock', 'rock', 'paper'],
            'to_win': ['paper', 'scissors', 'rock', 'paper', 'scissors'],
            'not_to_lose': ['scissors', 'rock', 'paper', 'scissors', 'rock']
        },
        'model_confidence_history': {
            'random': [0.33, 0.33, 0.33, 0.33, 0.33],
            'frequency': [0.30, 0.32, 0.34, 0.36, 0.38],
            'markov': [0.40, 0.415, 0.43, 0.445, 0.46],
            'enhanced': [0.45, 0.50, 0.55, 0.60, 0.65],
            'lstm': [0.50, 0.52, 0.48, 0.51, 0.53],
            'to_win': [0.35, 0.40, 0.45, 0.50, 0.55],
            'not_to_lose': [0.30, 0.35, 0.40, 0.45, 0.50]
        },
        'accuracy': {
            'random': 0.20,
            'frequency': 0.40,
            'markov': 0.60,
            'enhanced': 0.80,
            'lstm': 0.70,
            'to_win': 0.60,
            'not_to_lose': 0.50
        }
    }
    
    # Create metrics aggregator
    aggregator = ComprehensiveMetricsAggregator()
    
    # Get comprehensive metrics
    print("🔍 Testing AI behavior metrics...")
    metrics = aggregator.aggregate_comprehensive_metrics(sample_game_state)
    
    # Print AI behavior section
    ai_behavior = metrics.get('ai_behavior', {})
    print("\n📊 AI Behavior Metrics:")
    print(json.dumps(ai_behavior, indent=2))
    
    # Check for empty fields
    for key, value in ai_behavior.items():
        if not value or (isinstance(value, dict) and not value):
            print(f"⚠️ Empty field detected: {key}")
        else:
            print(f"✅ {key}: {type(value)} with {len(value) if isinstance(value, (dict, list)) else 'scalar'} items")

if __name__ == "__main__":
    test_ai_behavior_metrics()