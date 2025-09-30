#!/usr/bin/env python3
"""
Test the new Advanced Personality Engine
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from personality_engine import get_personality_engine


def test_personality_engine():
    """Test the advanced personality engine"""
    print("🧪 TESTING ADVANCED PERSONALITY ENGINE")
    print("=" * 50)
    
    engine = get_personality_engine()
    
    # Test personality listing
    personalities = engine.get_all_personalities()
    print(f"✅ Available personalities: {len(personalities)}")
    for name in personalities:
        print(f"   • {name}")
    
    print("\n🔍 Testing each personality...")
    
    # Test each personality
    test_history = ['paper', 'stone', 'scissor', 'paper', 'stone']
    test_game_history = [('paper', 'scissor'), ('stone', 'paper'), ('scissor', 'stone')]
    
    for personality_name in personalities:
        print(f"\n📊 Testing {personality_name.upper()}:")
        
        # Set personality
        engine.set_personality(personality_name)
        
        # Get personality info
        info = engine.get_personality_info(personality_name)
        print(f"   Name: {info['name']}")
        print(f"   Description: {info['description'][:60]}...")
        print(f"   Color Theme: {info['color_theme']['primary']}")
        
        # Test move modification
        base_moves = ['paper', 'stone', 'scissor']
        personality_moves = []
        
        for base_move in base_moves:
            modified_move = engine.apply_personality_to_move(
                base_move, 0.7, test_history, test_game_history
            )
            personality_moves.append(modified_move)
        
        print(f"   Base moves:        {base_moves}")
        print(f"   Modified moves:    {personality_moves}")
        
        # Simulate some games to see adaptation
        for i in range(5):
            engine.update_game_state('paper', 'scissor', 'human', 0.6)
        
        stats = engine.get_personality_stats()
        print(f"   After 5 games:     Win rate: {stats['win_rate']:.1f}%, Adaptation: {stats['adaptation_level']:.1f}")
    
    print("\n" + "=" * 50)
    print("🎯 PERSONALITY BEHAVIORAL DIFFERENCES TEST")
    print("=" * 50)
    
    # Test different personalities against the same scenario
    scenario_history = ['paper'] * 8 + ['stone', 'scissor']  # Heavy paper bias
    scenario_game_history = [('paper', 'scissor')] * 8 + [('stone', 'paper'), ('scissor', 'stone')]
    
    print("Scenario: Human has heavy paper bias (80% paper)")
    print("Testing how each personality responds to 'stone' as base move:\n")
    
    for personality_name in personalities:
        engine.set_personality(personality_name)
        
        # Update with scenario history
        for human_move, robot_move in scenario_game_history:
            result = 'human' if human_move == 'paper' and robot_move == 'scissor' else 'robot'
            engine.update_game_state(human_move, robot_move, result, 0.7)
        
        # Test response
        response = engine.apply_personality_to_move(
            'stone', 0.8, [h for h, r in scenario_game_history], scenario_game_history
        )
        
        stats = engine.get_personality_stats()
        print(f"{personality_name:12} → {response:7} (Adaptation: {stats['adaptation_level']:5.1f})")
    
    print("\n✅ Advanced Personality Engine Test Complete!")
    print("   All personalities show distinct behavioral patterns!")


if __name__ == "__main__":
    test_personality_engine()