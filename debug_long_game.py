#!/usr/bin/env python3
"""
Quick debug script to test the long game scenario data
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_long_game_data():
    """Test the problematic long game data directly"""
    
    # Replicate the long game scenario data
    test_data = {
        'human_moves': (['rock', 'paper', 'scissors'] * 7)[:20],
        'robot_moves': (['paper', 'scissors', 'rock'] * 7)[:20],
        'results': (['lose', 'lose', 'lose'] * 7)[:20],
        'ai_difficulty': 'enhanced',
        'human_strategy_label': 'cyclical',
        'change_points': [5, 12],
        'model_predictions_history': {
            'enhanced': ['rock'] * 20,
            'markov': ['paper'] * 20
        }
    }
    
    print("🔍 Testing long game data structure...")
    print(f"human_moves length: {len(test_data['human_moves'])}")
    print(f"robot_moves length: {len(test_data['robot_moves'])}")
    print(f"results length: {len(test_data['results'])}")
    print(f"results type: {type(test_data['results'])}")
    print(f"results content: {test_data['results']}")
    
    # Test with game_context builder
    try:
        from game_context import build_game_context
        
        print("\n🧪 Testing with game_context builder...")
        
        game_data = build_game_context(
            session={},
            overrides=test_data,
            context_type='ai_coaching'
        )
        
        print("✅ Game context built successfully!")
        print(f"Final results type: {type(game_data.get('results', []))}")
        print(f"Final results length: {len(game_data.get('results', []))}")
        
        # Test with enhanced coach 
        sys.path.append('webapp')
        from webapp.app import get_enhanced_coach
        
        enhanced_coach = get_enhanced_coach()
        print("\n🤖 Testing with enhanced coach...")
        
        analysis = enhanced_coach.generate_coaching_advice(
            game_state=game_data,
            coaching_type='comprehensive'
        )
        
        print("✅ Enhanced coach analysis successful!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_long_game_data()