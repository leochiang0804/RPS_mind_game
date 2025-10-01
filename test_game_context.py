#!/usr/bin/env python3
"""
Test the game context builder to ensure it works correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from game_context import build_game_context, GameContextConfig


def test_basic_context():
    """Test basic context building"""
    
    # Mock session data
    session = {
        'human_moves': ['paper', 'stone', 'scissor'],
        'robot_moves': ['stone', 'paper', 'stone'],
        'results': ['win', 'lose', 'lose'],
        'human_strategy_label': 'adaptive',
        'ai_difficulty': 'hard',
        'accuracy': {'enhanced': 75.0, 'frequency': 60.0},
        'model_confidence_history': {'enhanced': [0.8, 0.7, 0.9]},
        'strategy_preference': 'aggressive',
        'personality': 'berserker'
    }
    
    print("🔧 Testing basic context building...")
    
    # Test full context
    context = build_game_context(session, context_type='full')
    
    # Validate basic fields
    assert context['human_moves'] == ['paper', 'stone', 'scissor']
    assert context['robot_moves'] == ['stone', 'paper', 'stone']
    assert context['results'] == ['win', 'lose', 'lose']
    assert context['round'] == 3
    assert context['human_strategy_label'] == 'adaptive'
    assert context['ai_difficulty'] == 'hard'
    assert context['accuracy'] == {'enhanced': 75.0, 'frequency': 60.0}
    
    print("✅ Basic context building works!")
    
    # Test with overrides
    overrides = {
        'human_strategy_label': 'overridden_strategy',
        'ai_difficulty': 'overridden_difficulty'
    }
    
    context_with_overrides = build_game_context(session, overrides=overrides)
    
    assert context_with_overrides['human_strategy_label'] == 'overridden_strategy'
    assert context_with_overrides['ai_difficulty'] == 'overridden_difficulty'
    # Original session data should still be there
    assert context_with_overrides['human_moves'] == ['paper', 'stone', 'scissor']
    
    print("✅ Context building with overrides works!")


def test_realtime_context():
    """Test optimized realtime context"""
    
    session = {
        'human_moves': ['paper'],
        'robot_moves': ['stone'],
        'results': ['win'],
        'accuracy': {'enhanced': 50.0}
    }
    
    print("🔧 Testing realtime context building...")
    
    context = build_game_context(session, context_type='realtime')
    
    assert context['human_moves'] == ['paper']
    assert context['round'] == 1
    assert context['accuracy'] == {'enhanced': 50.0}
    
    print("✅ Realtime context building works!")


def test_empty_session():
    """Test handling of empty/minimal session data"""
    
    print("🔧 Testing empty session handling...")
    
    session = {}
    
    context = build_game_context(session)
    
    # Should have defaults
    assert context['human_moves'] == []
    assert context['robot_moves'] == []
    assert context['results'] == []
    assert context['round'] == 0
    assert context['human_strategy_label'] == 'unknown'
    assert context['ai_difficulty'] == 'medium'
    
    print("✅ Empty session handling works!")


def test_fallback_logic():
    """Test strategy and difficulty fallback logic"""
    
    print("🔧 Testing fallback logic...")
    
    # Test strategy fallback
    session = {
        'current_strategy': 'fallback_strategy',
        'human_moves': ['paper'],
        'robot_moves': ['stone'],
        'results': ['win']
    }
    
    context = build_game_context(session)
    assert context['human_strategy_label'] == 'fallback_strategy'
    
    # Test difficulty fallback  
    session = {
        'difficulty': 'easy',
        'human_moves': ['paper'],
        'robot_moves': ['stone'],
        'results': ['win']
    }
    
    context = build_game_context(session)
    assert context['ai_difficulty'] == 'easy'
    
    print("✅ Fallback logic works!")


if __name__ == '__main__':
    print("🧪 Testing Game Context Builder\n")
    
    try:
        test_basic_context()
        test_realtime_context()
        test_empty_session()
        test_fallback_logic()
        
        print("\n🎉 All tests passed! Game context builder is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)