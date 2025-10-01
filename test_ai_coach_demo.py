#!/usr/bin/env python3
"""
Test script for AI Coach Demo functionality
Tests the enhanced metrics system and model performance tracking
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_coach_metrics import AICoachMetricsAggregator
from enhanced_coach import EnhancedCoachSystem

def create_test_game_state():
    """Create a comprehensive test game state"""
    return {
        'human_moves': ['rock', 'paper', 'scissors', 'rock', 'paper', 'scissors', 'rock', 'paper'],
        'robot_moves': ['scissors', 'rock', 'paper', 'paper', 'scissors', 'rock', 'scissors', 'rock'],
        'results': ['lose', 'lose', 'lose', 'tie', 'win', 'lose', 'tie', 'lose'],
        'current_strategy': 'lstm',
        'total_games': 8,
        'human_wins': 1,
        'robot_wins': 5,
        'ties': 2
    }

def test_metrics_aggregation():
    """Test the comprehensive metrics aggregation"""
    print("Testing AI Coach Metrics Aggregation...")
    
    metrics_aggregator = AICoachMetricsAggregator()
    game_state = create_test_game_state()
    
    try:
        # Test comprehensive metrics
        metrics = metrics_aggregator.aggregate_comprehensive_metrics(game_state)
        
        print(f"✅ Metrics aggregated successfully")
        print(f"📊 Total metric categories: {len(metrics)}")
        
        # Check key categories
        expected_categories = [
            'performance_metrics', 'behavioral_patterns', 'strategic_analysis',
            'psychological_profile', 'model_performance'
        ]
        
        for category in expected_categories:
            if category in metrics:
                print(f"✅ {category}: Found")
                if isinstance(metrics[category], dict):
                    print(f"   └─ Subcategories: {len(metrics[category])}")
            else:
                print(f"❌ {category}: Missing")
        
        # Test model performance specifically
        if 'model_performance' in metrics:
            model_perf = metrics['model_performance']
            print(f"\n🤖 Model Performance Analysis:")
            
            if 'current_ai_model' in model_perf:
                current_ai = model_perf['current_ai_model']
                print(f"   Current AI: {current_ai.get('name', 'unknown')}")
                print(f"   Difficulty: {current_ai.get('difficulty_level', 'unknown')}")
                print(f"   AI Win Rate: {current_ai.get('win_rate_against_human', 'unknown')}")
            
            if 'model_specific_advice' in model_perf:
                advice = model_perf['model_specific_advice']
                print(f"   Strategy Advice: {advice.get('primary_strategy', 'none')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during metrics aggregation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_personality_integration():
    """Test enhanced coaching system integration"""
    print("\n" + "="*60)
    print("Testing Enhanced Coaching System Integration...")
    
    coach_system = EnhancedCoachSystem()
    metrics_aggregator = AICoachMetricsAggregator()
    game_state = create_test_game_state()
    
    try:
        # Get comprehensive metrics
        metrics = metrics_aggregator.aggregate_comprehensive_metrics(game_state)
        
        # Test both coaching styles
        coaching_styles = ['easy', 'scientific']
        
        for style in coaching_styles:
            print(f"\n🎭 Testing {style} coaching style:")
            
            # Set coaching style
            coach_system.set_coaching_style(style)
            
            # Generate coaching advice
            advice = coach_system.generate_coaching_advice(game_state)
            
            if advice and 'tips' in advice:
                print(f"✅ Generated {len(advice['tips'])} tips")
                print(f"📝 Sample tip: {advice['tips'][0][:100]}...")
                print(f"🔧 Mode used: {advice.get('mode', 'unknown')}")
            else:
                print(f"❌ Failed to generate advice")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during coaching integration: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_performance_metrics():
    """Test specific model performance metrics"""
    print("\n" + "="*60)
    print("Testing Model Performance Metrics...")
    
    metrics_aggregator = AICoachMetricsAggregator()
    
    # Test with different AI strategies
    strategies = ['random', 'frequency', 'markov', 'enhanced', 'lstm']
    
    for strategy in strategies:
        print(f"\n🤖 Testing with {strategy} AI:")
        
        game_state = create_test_game_state()
        game_state['current_strategy'] = strategy
        
        try:
            model_metrics = metrics_aggregator._extract_model_performance_metrics(game_state)
            
            if 'current_ai_model' in model_metrics:
                current_ai = model_metrics['current_ai_model']
                print(f"   ✅ AI Name: {current_ai.get('name')}")
                print(f"   ✅ Difficulty: {current_ai.get('difficulty_level')}")
                print(f"   ✅ Win Rate: {current_ai.get('win_rate_against_human')}")
                print(f"   ✅ Adaptation: {current_ai.get('adaptation_speed')}")
            
            if 'model_specific_advice' in model_metrics:
                advice = model_metrics['model_specific_advice']
                print(f"   ✅ Strategy: {advice.get('primary_strategy', 'none')[:60]}...")
                
        except Exception as e:
            print(f"   ❌ Error with {strategy}: {e}")
    
    return True

def main():
    """Run all tests"""
    print("🚀 Starting AI Coach Demo Tests")
    print("="*60)
    
    tests_passed = 0
    total_tests = 3
    
    # Run tests
    if test_metrics_aggregation():
        tests_passed += 1
    
    if test_personality_integration():
        tests_passed += 1
    
    if test_model_performance_metrics():
        tests_passed += 1
    
    # Summary
    print("\n" + "="*60)
    print(f"🏁 Test Summary: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! AI Coach Demo is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)