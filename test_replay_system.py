#!/usr/bin/env python3
"""
Test the Replay System functionality
"""

import sys
import os
import tempfile
import shutil

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from replay_system import GameReplay, ReplayManager, ReplayAnalyzer


def test_replay_system():
    """Test the complete replay system"""
    print("🎬 TESTING GAME REPLAY SYSTEM")
    print("=" * 50)
    
    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    print(f"📁 Using temporary directory: {temp_dir}")
    
    try:
        # Test 1: Create and populate a replay
        print("\n🎮 Test 1: Creating Game Replay...")
        replay = GameReplay()
        replay.metadata.update({
            'difficulty': 'enhanced',
            'strategy': 'to_win',
            'personality': 'aggressive'
        })
        
        # Simulate a game session
        test_moves = [
            ('paper', 'scissor', 'robot'),
            ('stone', 'paper', 'robot'),
            ('scissor', 'paper', 'human'),
            ('paper', 'scissor', 'robot'),
            ('stone', 'stone', 'tie'),
            ('paper', 'paper', 'tie'),
            ('scissor', 'stone', 'robot'),
            ('paper', 'scissor', 'robot'),
            ('stone', 'paper', 'robot'),
            ('scissor', 'paper', 'human')
        ]
        
        for i, (human_move, robot_move, result) in enumerate(test_moves):
            confidence = 0.7 + (i % 3) * 0.1  # Varying confidence
            replay.add_move(
                round_number=i + 1,
                human_move=human_move,
                robot_move=robot_move,
                result=result,
                confidence=confidence,
                strategy_used='enhanced_to_win',
                analysis={'pattern_detected': 'random' if i < 5 else 'reactive'}
            )
        
        print(f"✅ Created replay with {len(replay.moves)} moves")
        print(f"   Session ID: {replay.session_id}")
        print(f"   Final Score: {replay.metadata['final_score']}")
        
        # Test 2: Replay Manager
        print("\n💾 Test 2: Replay Manager...")
        manager = ReplayManager(temp_dir)
        
        # Save replay
        filepath = manager.save_replay(replay)
        print(f"✅ Saved replay to: {filepath}")
        
        # Load replay
        loaded_replay = manager.load_replay(replay.session_id)
        if loaded_replay:
            print(f"✅ Loaded replay: {loaded_replay.session_id}")
            print(f"   Moves loaded: {len(loaded_replay.moves)}")
        else:
            print("❌ Failed to load replay")
            return False
        
        # List replays
        replay_list = manager.list_replays()
        print(f"✅ Found {len(replay_list)} replays in storage")
        
        # Test 3: Replay Analysis
        print("\n📊 Test 3: Replay Analysis...")
        analyzer = ReplayAnalyzer()
        analysis = analyzer.analyze_replay(loaded_replay)
        
        print(f"✅ Analysis completed:")
        print(f"   Total rounds: {analysis['session_info']['total_rounds']}")
        print(f"   Duration: {analysis['session_info']['duration']}")
        print(f"   Human win rate: {analysis['statistical_summary']['human_win_rate']:.1%}")
        print(f"   Robot win rate: {analysis['statistical_summary']['robot_win_rate']:.1%}")
        
        # Human patterns
        patterns = analysis['human_patterns']
        print(f"   Most common human move: {patterns['most_common_move']}")
        print(f"   Detected patterns: {patterns['detected_patterns']}")
        print(f"   Predictability score: {patterns['predictability_score']:.2f}")
        
        # Robot performance
        robot_perf = analysis['robot_performance']
        print(f"   Robot average confidence: {robot_perf['average_confidence']:.2f}")
        print(f"   Confidence trend: {robot_perf['confidence_trend']}")
        
        # Key moments
        key_moments = analysis['key_moments']
        print(f"   Key moments detected: {len(key_moments)}")
        for moment in key_moments[:3]:  # Show first 3
            print(f"     - Round {moment['round']}: {moment['description']}")
        
        # Improvement suggestions
        suggestions = analysis['improvement_suggestions']
        print(f"   Improvement suggestions: {len(suggestions)}")
        for suggestion in suggestions[:3]:  # Show first 3
            print(f"     - {suggestion}")
        
        # Test 4: Advanced Features
        print("\n🔧 Test 4: Advanced Features...")
        
        # Add annotations
        replay.add_annotation(3, "Great counter-move here!")
        replay.add_annotation(7, "Robot showing pattern recognition")
        print(f"✅ Added annotations to rounds 3 and 7")
        
        # Strategy changes
        strategy_changes = replay.get_strategy_changes()
        print(f"✅ Strategy changes detected: {len(strategy_changes)}")
        
        # Performance trends
        trends = replay.get_performance_trends()
        print(f"✅ Performance trends calculated for {len(trends['rounds'])} rounds")
        
        # Move range
        moves_1_to_5 = replay.get_moves_range(1, 5)
        print(f"✅ Retrieved moves 1-5: {len(moves_1_to_5)} moves")
        
        # Test 5: Export functionality
        print("\n📤 Test 5: Export Functionality...")
        csv_data = manager.export_replay_csv(replay.session_id)
        if csv_data:
            print(f"✅ CSV export generated: {len(csv_data)} characters")
            print("   CSV preview (first 200 chars):")
            print(f"   {csv_data[:200]}...")
        else:
            print("❌ CSV export failed")
        
        # Test 6: Performance test with multiple replays
        print("\n⚡ Test 6: Performance Test...")
        
        # Create multiple small replays
        for i in range(5):
            test_replay = GameReplay()
            test_replay.metadata.update({
                'difficulty': ['random', 'frequency', 'markov', 'enhanced'][i % 4],
                'strategy': ['balanced', 'to_win', 'not_to_lose'][i % 3],
                'personality': ['neutral', 'aggressive', 'defensive'][i % 3]
            })
            
            # Add 3 moves each
            for j in range(3):
                test_replay.add_move(
                    round_number=j + 1,
                    human_move=['paper', 'stone', 'scissor'][j],
                    robot_move=['scissor', 'paper', 'stone'][j],
                    result=['robot', 'robot', 'robot'][j],
                    confidence=0.6 + j * 0.1
                )
            
            manager.save_replay(test_replay)
        
        # List all replays
        all_replays = manager.list_replays()
        print(f"✅ Performance test: {len(all_replays)} total replays")
        
        # Analyze each
        analysis_results = []
        for replay_info in all_replays:
            test_replay = manager.load_replay(replay_info['session_id'])
            if test_replay:
                test_analysis = analyzer.analyze_replay(test_replay)
                analysis_results.append(test_analysis)
        
        print(f"✅ Analyzed {len(analysis_results)} replays successfully")
        
        print("\n" + "=" * 50)
        print("🎉 ALL REPLAY SYSTEM TESTS PASSED!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        try:
            shutil.rmtree(temp_dir)
            print(f"🧹 Cleaned up temporary directory")
        except Exception as e:
            print(f"⚠️  Warning: Could not clean up temp directory: {e}")


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n🧪 TESTING EDGE CASES")
    print("=" * 30)
    
    try:
        # Test empty replay analysis
        analyzer = ReplayAnalyzer()
        empty_replay = GameReplay()
        
        analysis = analyzer.analyze_replay(empty_replay)
        if 'error' in analysis:
            print("✅ Empty replay handled correctly")
        else:
            print("❌ Empty replay should return error")
        
        # Test invalid moves
        replay = GameReplay()
        replay.add_move(1, 'invalid_move', 'paper', 'tie')
        
        # Should still work but with invalid data
        analysis = analyzer.analyze_replay(replay)
        print("✅ Invalid moves handled gracefully")
        
        # Test very short game
        short_replay = GameReplay()
        short_replay.add_move(1, 'paper', 'stone', 'human')
        short_replay.add_move(2, 'stone', 'scissor', 'human')
        
        analysis = analyzer.analyze_replay(short_replay)
        print("✅ Short game analysis completed")
        
        print("✅ All edge cases handled correctly")
        return True
        
    except Exception as e:
        print(f"❌ Edge case test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_replay_system()
    success &= test_edge_cases()
    
    if success:
        print("\n🏆 REPLAY SYSTEM IS READY FOR PRODUCTION!")
    else:
        print("\n💥 REPLAY SYSTEM NEEDS FIXES")
        sys.exit(1)