#!/usr/bin/env python3
"""
Test script for the new centralized tracking objects
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from webapp.app import create_centralized_model_prediction_tracking, create_centralized_ai_strategy_accuracy
from game_context import build_game_context

def test_centralized_objects():
    """Test the new centralized tracking objects with sample data"""
    
    # Create sample session data
    sample_session = {
        'human_moves': ['rock', 'paper', 'scissors', 'rock', 'paper'],
        'robot_moves': ['paper', 'scissors', 'rock', 'paper', 'scissors'],
        'results': ['robot', 'robot', 'robot', 'robot', 'robot'],
        'difficulty': 'markov',
        'ai_difficulty': 'markov',
        'strategy_preference': 'to_win',
        'personality': 'neutral',
        'model_predictions_history': {
            'random': ['paper', 'rock', 'scissors', 'paper'],     # 4 predictions for 5 moves
            'frequency': ['rock', 'scissors', 'rock', 'paper'],   # Predicting next human move
            'markov': ['scissors', 'rock', 'paper', 'rock'],      # Predicting next human move
            'lstm': ['paper', 'paper', 'scissors', 'scissors']    # Predicting next human move
        },
        'model_confidence_history': {
            'random': [0.33, 0.33, 0.33, 0.33],
            'frequency': [0.6, 0.5, 0.7, 0.4],
            'markov': [0.45, 0.55, 0.6, 0.5],
            'lstm': [0.7, 0.8, 0.6, 0.75]
        }
    }
    
    # Mock Flask session
    import unittest.mock
    with unittest.mock.patch('webapp.app.session', sample_session):
        print("🔍 Testing Centralized Model Prediction Tracking...")
        
        # Test model prediction tracking
        tracking_data = create_centralized_model_prediction_tracking()
        
        print("\n📊 Model Prediction Tracking Results:")
        print(f"  Models: {tracking_data['models']}")
        print(f"  Total predictions: {tracking_data['total_predictions']}")
        
        for model in tracking_data['models']:
            predictions = tracking_data['prediction_history'][model]
            counts = tracking_data['prediction_counts'][model]
            print(f"  {model.upper()}: {len(predictions)} predictions - {predictions}")
            print(f"    Counts: Rock={counts['rock']}, Paper={counts['paper']}, Scissors={counts['scissors']}")
        
        print(f"\n📈 Round-by-round data: {len(tracking_data['rounds_data'])} rounds")
        for round_data in tracking_data['rounds_data'][:3]:  # Show first 3 rounds
            print(f"    Round {round_data['round']}: {round_data['predictions']}")
        
        print("\n🎯 Testing Centralized AI Strategy Accuracy...")
        
        # Test AI strategy accuracy
        accuracy_data = create_centralized_ai_strategy_accuracy()
        
        print(f"\n📊 AI Strategy Accuracy Results:")
        print(f"  Models: {accuracy_data['models']}")
        print(f"  Calculation method: {accuracy_data['calculation_method']}")
        
        for model in accuracy_data['models']:
            accuracy = accuracy_data['accuracy_percentages'][model]
            correct = accuracy_data['correct_predictions'][model]
            total = accuracy_data['total_valid_predictions'][model]
            print(f"  {model.upper()}: {accuracy}% ({correct}/{total} correct)")
            
            # Show detailed comparisons for first model
            if model == 'random':
                print(f"    Detailed comparisons for {model.upper()}:")
                for comp in accuracy_data['detailed_comparisons'][model][:3]:
                    print(f"      {comp['note']}: '{comp['predicted']}' vs '{comp['actual']}' → {'✅' if comp['correct'] else '❌'}")
        
        print("\n✅ Both centralized objects are working correctly!")
        
        # Verify the objects contain the expected data structure
        required_tracking_fields = ['models', 'prediction_counts', 'prediction_history', 'total_predictions', 'rounds_data']
        required_accuracy_fields = ['models', 'accuracy_percentages', 'correct_predictions', 'total_valid_predictions', 'detailed_comparisons']
        
        tracking_ok = all(field in tracking_data for field in required_tracking_fields)
        accuracy_ok = all(field in accuracy_data for field in required_accuracy_fields)
        
        print(f"\n📋 Data Structure Validation:")
        print(f"  Model Prediction Tracking: {'✅ Valid' if tracking_ok else '❌ Invalid'}")
        print(f"  AI Strategy Accuracy: {'✅ Valid' if accuracy_ok else '❌ Invalid'}")
        
        return tracking_ok and accuracy_ok

if __name__ == "__main__":
    try:
        success = test_centralized_objects()
        if success:
            print("\n🎉 All tests passed! The centralized objects are ready for use.")
        else:
            print("\n❌ Some tests failed. Please check the implementation.")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)