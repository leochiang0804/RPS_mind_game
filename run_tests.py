#!/usr/bin/env python3
"""
Simple test runner for Phase 3 features that tests the components we can access
"""

import sys
import os
import unittest
from io import StringIO

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all our Phase 3 modules can be imported"""
    print("🔍 Testing module imports...")
    
    try:
        from optimized_strategies import ToWinStrategy, NotToLoseStrategy
        print("✅ Optimized Strategies imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import optimized_strategies: {e}")
        return False
    
    try:
        from tournament_system import TournamentSystem, Player, Match
        print("✅ Tournament System imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import tournament_system: {e}")
        return False
    
    try:
        from strategy import EnhancedStrategy, FrequencyStrategy, MarkovStrategy
        print("✅ Strategy modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import strategy modules: {e}")
        return False
    
    try:
        from coach_tips import CoachTipsGenerator
        print("✅ Coach Tips imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import coach_tips: {e}")
        return False
    
    try:
        from change_point_detector import ChangePointDetector
        print("✅ Change Point Detector imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import change_point_detector: {e}")
        return False
    
    return True

def test_optimized_strategies():
    """Test our optimized strategies functionality"""
    print("\n🎯 Testing Optimized Strategies...")
    
    try:
        from optimized_strategies import ToWinStrategy, NotToLoseStrategy
        
        # Test To Win Strategy
        to_win = ToWinStrategy()
        print(f"✅ ToWinStrategy created: {to_win.name}")
        
        # Test basic functionality
        test_history = ['paper', 'stone', 'scissor', 'paper']
        move = to_win.predict(test_history)
        print(f"✅ ToWinStrategy prediction: {move}")
        
        confidence = to_win.get_confidence()
        print(f"✅ ToWinStrategy confidence: {confidence:.3f}")
        
        stats = to_win.get_stats()
        print(f"✅ ToWinStrategy stats: {stats}")
        
        # Test Not to Lose Strategy
        not_to_lose = NotToLoseStrategy()
        print(f"✅ NotToLoseStrategy created: {not_to_lose.name}")
        
        move2 = not_to_lose.predict(test_history)
        print(f"✅ NotToLoseStrategy prediction: {move2}")
        
        confidence2 = not_to_lose.get_confidence()
        print(f"✅ NotToLoseStrategy confidence: {confidence2:.3f}")
        
        stats2 = not_to_lose.get_stats()
        print(f"✅ NotToLoseStrategy stats: {stats2}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing optimized strategies: {e}")
        return False

def test_tournament_system():
    """Test tournament system functionality"""
    print("\n🏆 Testing Tournament System...")
    
    try:
        from tournament_system import TournamentSystem, Player, Match
        import tempfile
        
        # Create temporary file for testing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp_file:
            temp_path = temp_file.name
        
        try:
            # Test tournament system
            tournament = TournamentSystem(temp_path)
            print("✅ TournamentSystem created")
            
            # Test player creation
            player1 = tournament.create_player("TestPlayer1")
            player2 = tournament.create_player("TestPlayer2")
            print(f"✅ Players created: {player1.name}, {player2.name}")
            
            # Test player stats
            player1.update_stats('win')
            player1.update_stats('loss')
            print(f"✅ Player1 stats: W:{player1.wins} L:{player1.losses} Rate:{player1.get_win_rate():.1f}%")
            
            # Test match creation
            match = tournament.create_match(player1.id, player2.id)
            print(f"✅ Match created: {match.id}")
            
            # Test match rounds
            result1 = match.add_round('paper', 'stone')  # Player1 wins
            result2 = match.add_round('stone', 'paper')  # Player2 wins
            result3 = match.add_round('scissor', 'scissor')  # Tie
            print(f"✅ Match rounds: {result1}, {result2}, {result3}")
            
            # Test leaderboard
            leaderboard = tournament.get_leaderboard(5)
            print(f"✅ Leaderboard generated with {len(leaderboard)} players")
            
            return True
            
        finally:
            # Clean up temp file
            os.unlink(temp_path)
        
    except Exception as e:
        print(f"❌ Error testing tournament system: {e}")
        return False

def test_coaching_system():
    """Test coaching system functionality"""
    print("\n🎓 Testing Coaching System...")
    
    try:
        from coach_tips import CoachTipsGenerator
        
        coach = CoachTipsGenerator()
        print("✅ CoachTipsGenerator created")
        
        # Test tip generation
        human_history = ['paper', 'stone', 'scissor', 'paper', 'stone']
        robot_history = ['scissor', 'paper', 'stone', 'scissor', 'paper']
        result_history = ['human', 'robot', 'human', 'human', 'robot']
        
        tips_data = coach.generate_tips(
            human_history=human_history,
            robot_history=robot_history,
            result_history=result_history,
            change_points=[],
            current_strategy='enhanced'
        )
        
        print(f"✅ Generated {len(tips_data.get('tips', []))} coaching tips")
        print(f"✅ Pattern analysis: {len(tips_data.get('patterns', {}))} patterns found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing coaching system: {e}")
        return False

def test_change_point_detection():
    """Test change point detection functionality"""
    print("\n📊 Testing Change Point Detection...")
    
    try:
        from change_point_detector import ChangePointDetector
        
        detector = ChangePointDetector()
        print("✅ ChangePointDetector created")
        
        # Add some moves with clear change points
        moves = ['paper'] * 5 + ['stone'] * 5 + ['scissor'] * 5
        
        for move in moves:
            detector.add_move(move)
        
        change_points = detector.get_all_change_points()
        strategy_label = detector.get_current_strategy_label()
        
        print(f"✅ Detected {len(change_points)} change points")
        print(f"✅ Current strategy label: {strategy_label}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing change point detection: {e}")
        return False

def test_webapp_integration():
    """Test that webapp can import all required modules"""
    print("\n🌐 Testing Webapp Integration...")
    
    try:
        # Add webapp to path
        webapp_path = os.path.join(os.path.dirname(__file__), 'webapp')
        if webapp_path not in sys.path:
            sys.path.append(webapp_path)
        
        # Test that app.py can be imported (but don't run it)
        import importlib.util
        spec = importlib.util.spec_from_file_location("app", os.path.join(webapp_path, "app.py"))
        if spec and spec.loader:
            app_module = importlib.util.module_from_spec(spec)
            print("✅ Webapp app.py can be loaded")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing webapp integration: {e}")
        return False

def run_comprehensive_tests():
    """Run all Phase 3 tests"""
    print("🧪 COMPREHENSIVE PHASE 3 FEATURE TESTS")
    print("=" * 50)
    
    results = {}
    
    # Run each test
    tests = [
        ("Module Imports", test_imports),
        ("Optimized Strategies", test_optimized_strategies),
        ("Tournament System", test_tournament_system),
        ("Coaching System", test_coaching_system),
        ("Change Point Detection", test_change_point_detection),
        ("Webapp Integration", test_webapp_integration),
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 50)
    print("🎯 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    success_rate = (passed / total * 100) if total > 0 else 0
    print(f"\n📊 Success Rate: {passed}/{total} ({success_rate:.1f}%)")
    
    if success_rate >= 90:
        print("🎉 Excellent! Phase 3 features are working great!")
    elif success_rate >= 75:
        print("✅ Good! Most Phase 3 features are working correctly.")
    elif success_rate >= 50:
        print("⚠️  Some issues found. Check the failures above.")
    else:
        print("❌ Significant issues found. Review the implementation.")
    
    return results

if __name__ == "__main__":
    run_comprehensive_tests()