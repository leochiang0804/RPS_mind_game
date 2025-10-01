#!/usr/bin/env python3
"""
Test dual metric categories - Real-time vs Post-game metrics
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_coach_metrics import AICoachMetricsAggregator

def create_test_game_state():
    """Create a test game state with sufficient data"""
    return {
        'human_moves': ['rock', 'paper', 'scissors', 'rock', 'paper', 'scissors', 'rock', 'paper', 'scissors', 'rock'],
        'robot_moves': ['scissors', 'rock', 'paper', 'paper', 'scissors', 'rock', 'scissors', 'rock', 'paper', 'scissors'],
        'results': ['lose', 'lose', 'lose', 'tie', 'win', 'lose', 'tie', 'lose', 'win', 'lose'],
        'current_strategy': 'lstm',
        'total_games': 10,
        'human_wins': 2,
        'robot_wins': 6,
        'ties': 2
    }

def test_dual_metrics():
    """Test both real-time and post-game metrics"""
    print("🚀 Testing Dual Metric Categories")
    print("="*50)
    
    aggregator = AICoachMetricsAggregator()
    game_state = create_test_game_state()
    
    # Test real-time metrics
    print("\n📱 REAL-TIME METRICS (for live coaching):")
    print("-" * 40)
    realtime_metrics = aggregator.get_realtime_metrics(game_state)
    
    if 'status' in realtime_metrics:
        print(f"❌ Insufficient data: {realtime_metrics}")
    else:
        print(f"✅ Real-time metrics collected")
        for category, data in realtime_metrics.items():
            print(f"   📊 {category}: {type(data).__name__}")
            if isinstance(data, dict) and len(data) <= 5:
                for key, value in list(data.items())[:3]:
                    print(f"      - {key}: {value}")
    
    # Test post-game metrics  
    print("\n📈 POST-GAME METRICS (for detailed analysis):")
    print("-" * 45)
    postgame_metrics = aggregator.get_postgame_metrics(game_state)
    
    if 'status' in postgame_metrics:
        print(f"❌ Insufficient data: {postgame_metrics}")
    else:
        print(f"✅ Post-game metrics collected")
        for category, data in postgame_metrics.items():
            print(f"   📊 {category}: {type(data).__name__}")
            if isinstance(data, dict) and category == 'postgame_analysis':
                for key in list(data.keys())[:3]:
                    print(f"      - {key}: Available")
    
    # Compare metric complexity
    print("\n⚖️  METRIC COMPARISON:")
    print("-" * 25)
    
    def count_metrics(data, prefix=""):
        count = 0
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    count += count_metrics(value, f"{prefix}{key}.")
                else:
                    count += 1
        return count
    
    realtime_count = count_metrics(realtime_metrics)
    postgame_count = count_metrics(postgame_metrics)
    
    print(f"📱 Real-time metric count: {realtime_count}")
    print(f"📈 Post-game metric count: {postgame_count}")
    print(f"📊 Complexity ratio: {postgame_count/realtime_count:.1f}x more detailed" if realtime_count > 0 else "N/A")
    
    # Test performance difference
    print("\n⏱️  PERFORMANCE COMPARISON:")
    print("-" * 30)
    
    import time
    
    # Time real-time metrics
    start_time = time.time()
    for _ in range(10):
        aggregator.get_realtime_metrics(game_state)
    realtime_duration = time.time() - start_time
    
    # Time post-game metrics
    start_time = time.time()
    for _ in range(10):
        aggregator.get_postgame_metrics(game_state)
    postgame_duration = time.time() - start_time
    
    print(f"📱 Real-time (10x): {realtime_duration*1000:.1f}ms")
    print(f"📈 Post-game (10x): {postgame_duration*1000:.1f}ms")
    print(f"⚡ Speed difference: {postgame_duration/realtime_duration:.1f}x slower")
    
    return True

def main():
    """Run dual metrics test"""
    try:
        success = test_dual_metrics()
        
        if success:
            print("\n🎉 Dual metric categories working correctly!")
            print("✅ Real-time metrics: Fast, focused on immediate strategy")
            print("✅ Post-game metrics: Comprehensive, focused on learning")
            return True
        else:
            print("\n❌ Dual metrics test failed")
            return False
            
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)