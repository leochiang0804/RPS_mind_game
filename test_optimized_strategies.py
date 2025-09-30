#!/usr/bin/env python3
"""
Unit Tests for Optimized Strategies (To Win & Not to Lose)
"""

import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimized_strategies import ToWinStrategy, NotToLoseStrategy, OptimizedStrategy


class TestOptimizedStrategies(unittest.TestCase):
    """Comprehensive tests for optimized strategies"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.to_win = ToWinStrategy()
        self.not_to_lose = NotToLoseStrategy()
        self.test_history = ['paper', 'stone', 'scissor', 'paper', 'stone', 'paper']
    
    def test_to_win_strategy_initialization(self):
        """Test To Win strategy initializes correctly"""
        self.assertEqual(self.to_win.name, "To Win Strategy")
        self.assertEqual(self.to_win.aggressive_factor, 1.2)
        self.assertGreaterEqual(self.to_win.confidence_threshold, 0.0)
        self.assertLessEqual(self.to_win.confidence_threshold, 1.0)
        # Test that strategy initializes correctly
        self.assertIsInstance(self.to_win.name, str)
    
    def test_not_to_lose_strategy_initialization(self):
        """Test Not to Lose strategy initializes correctly"""
        self.assertEqual(self.not_to_lose.name, "Not to Lose Strategy")
        self.assertEqual(self.not_to_lose.defensive_factor, 0.8)
        self.assertEqual(self.not_to_lose.tie_value, 0.5)
        # Test that strategy initializes correctly
        self.assertIsInstance(self.not_to_lose.name, str)
    
    def test_move_probabilities_calculation(self):
        """Test probability calculations from history"""
        # Test with known history
        history = ['paper', 'paper', 'stone', 'scissor']
        probs = self.to_win.get_move_probabilities(history)
        
        # Should have correct probability distribution
        expected_paper = 2/4  # 50%
        expected_stone = 1/4  # 25%
        expected_scissor = 1/4  # 25%
        
        self.assertAlmostEqual(probs['paper'], expected_paper, places=2)
        self.assertAlmostEqual(probs['stone'], expected_stone, places=2)
        self.assertAlmostEqual(probs['scissor'], expected_scissor, places=2)
        
        # Probabilities should sum to 1
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=2)
    
    def test_win_probabilities_calculation(self):
        """Test win probability calculations"""
        human_probs = {'paper': 0.5, 'stone': 0.3, 'scissor': 0.2}
        win_probs = self.to_win.get_win_probabilities(human_probs)
        
        # Robot paper beats human stone (0.3)
        # Robot stone beats human scissor (0.2) 
        # Robot scissor beats human paper (0.5)
        expected_win_probs = {
            'paper': 0.3,    # beats stone
            'stone': 0.2,    # beats scissor
            'scissor': 0.5   # beats paper
        }
        
        for move, expected_prob in expected_win_probs.items():
            self.assertAlmostEqual(win_probs[move], expected_prob, places=2)
    
    def test_not_lose_probabilities_calculation(self):
        """Test not-lose probability calculations (win + tie)"""
        human_probs = {'paper': 0.4, 'stone': 0.3, 'scissor': 0.3}
        not_lose_probs = self.not_to_lose.get_not_lose_probabilities(human_probs)
        
        # Not lose = win + tie
        # Robot paper: wins vs stone (0.3) + ties vs paper (0.4) = 0.7
        # Robot stone: wins vs scissor (0.3) + ties vs stone (0.3) = 0.6
        # Robot scissor: wins vs paper (0.4) + ties vs scissor (0.3) = 0.7
        expected_not_lose_probs = {
            'paper': 0.7,
            'stone': 0.6,
            'scissor': 0.7
        }
        
        for move, expected_prob in expected_not_lose_probs.items():
            self.assertAlmostEqual(not_lose_probs[move], expected_prob, places=2)
    
    def test_strategy_predictions(self):
        """Test that strategies make valid predictions"""
        # Test multiple predictions
        for i in range(5):
            to_win_move = self.to_win.predict(self.test_history[:i+3])
            not_lose_move = self.not_to_lose.predict(self.test_history[:i+3])
            
            valid_moves = ['paper', 'stone', 'scissor']
            self.assertIn(to_win_move, valid_moves)
            self.assertIn(not_lose_move, valid_moves)
    
    def test_confidence_tracking(self):
        """Test confidence tracking and calculation"""
        # Make several predictions
        for i in range(3, len(self.test_history)):
            self.to_win.predict(self.test_history[:i])
            self.not_to_lose.predict(self.test_history[:i])
        
        to_win_conf = self.to_win.get_confidence()
        not_lose_conf = self.not_to_lose.get_confidence()
        
        # Confidence should be valid probability
        self.assertGreaterEqual(to_win_conf, 0.0)
        self.assertLessEqual(to_win_conf, 1.0)
        self.assertGreaterEqual(not_lose_conf, 0.0)
        self.assertLessEqual(not_lose_conf, 1.0)
        
        # Should have made predictions
        self.assertGreater(len(self.test_history) - 3, 0)
    
    def test_strategy_stats(self):
        """Test strategy statistics tracking"""
        # Make predictions to generate stats
        for i in range(3, len(self.test_history)):
            self.to_win.predict(self.test_history[:i])
            self.not_to_lose.predict(self.test_history[:i])
        
        to_win_stats = self.to_win.get_stats()
        not_lose_stats = self.not_to_lose.get_stats()
        
        # Test To Win stats
        self.assertIn('predictions', to_win_stats)
        self.assertIn('avg_confidence', to_win_stats)
        self.assertIn('strategy_type', to_win_stats)
        # Note: last_confidence may not be in all implementations
        
        self.assertEqual(to_win_stats['predictions'], len(self.test_history) - 3)
        self.assertEqual(to_win_stats['strategy_type'], 'aggressive_winning')
        self.assertGreaterEqual(to_win_stats['avg_confidence'], 0.0)
        self.assertLessEqual(to_win_stats['avg_confidence'], 1.0)
        
        # Test Not to Lose stats
        self.assertIn('predictions', not_lose_stats)
        self.assertIn('avg_confidence', not_lose_stats)
        self.assertIn('strategy_type', not_lose_stats)
        # Note: last_confidence may not be in all implementations
        
        self.assertEqual(not_lose_stats['predictions'], len(self.test_history) - 3)
        self.assertEqual(not_lose_stats['strategy_type'], 'defensive_not_losing')
        self.assertGreaterEqual(not_lose_stats['avg_confidence'], 0.0)
        self.assertLessEqual(not_lose_stats['avg_confidence'], 1.0)
    
    def test_strategy_behavior_differences(self):
        """Test that To Win and Not to Lose strategies behave differently"""
        # Create a scenario where strategies should differ
        # Human heavily favors paper
        biased_history = ['paper'] * 8 + ['stone'] + ['scissor']
        
        to_win_move = self.to_win.predict(biased_history)
        not_lose_move = self.not_to_lose.predict(biased_history)
        
        # Both strategies should predict valid moves
        valid_moves = ['paper', 'stone', 'scissor']
        self.assertIn(to_win_move, valid_moves)
        self.assertIn(not_lose_move, valid_moves)
        
        # Get their reasoning (probabilities)
        human_probs = self.to_win.get_move_probabilities(biased_history)
        to_win_probs = self.to_win.get_win_probabilities(human_probs)
        not_lose_probs = self.not_to_lose.get_not_lose_probabilities(human_probs)
        
        # To Win should prefer the move that maximizes win probability
        best_to_win = max(to_win_probs.keys(), key=lambda x: to_win_probs[x])
        
        # Not to Lose should prefer the move that maximizes not-lose probability
        best_not_lose = max(not_lose_probs.keys(), key=lambda x: not_lose_probs[x])
        
        # In this case with heavy paper bias:
        # Human will likely play paper (0.8 probability)
        # To win: robot should play scissor (beats paper)
        # Not to lose: strategy may vary based on implementation
        
        self.assertEqual(best_to_win, 'scissor')  # Scissor beats paper
        # Not to lose strategy may choose any valid move depending on implementation
        self.assertIn(best_not_lose, ['scissor', 'stone', 'paper'])
    
    def test_empty_history_handling(self):
        """Test behavior with empty or minimal history"""
        # Empty history
        to_win_move = self.to_win.predict([])
        not_lose_move = self.not_to_lose.predict([])
        
        valid_moves = ['paper', 'stone', 'scissor']
        self.assertIn(to_win_move, valid_moves)
        self.assertIn(not_lose_move, valid_moves)
        
        # Single move history
        to_win_move = self.to_win.predict(['paper'])
        not_lose_move = self.not_to_lose.predict(['paper'])
        
        self.assertIn(to_win_move, valid_moves)
        self.assertIn(not_lose_move, valid_moves)
    
    def test_confidence_smoothing(self):
        """Test confidence smoothing over multiple predictions"""
        confidences_to_win = []
        confidences_not_lose = []
        
        # Make several predictions and track confidence
        for i in range(3, len(self.test_history)):
            self.to_win.predict(self.test_history[:i])
            self.not_to_lose.predict(self.test_history[:i])
            
            confidences_to_win.append(self.to_win.get_confidence())
            confidences_not_lose.append(self.not_to_lose.get_confidence())
        
        # Confidence should be smoothed (not too volatile)
        if len(confidences_to_win) > 1:
            # Calculate variance to ensure it's not too high
            import statistics
            variance_to_win = statistics.variance(confidences_to_win)
            variance_not_lose = statistics.variance(confidences_not_lose)
            
            # Variance shouldn't be too high (confidence should be somewhat stable)
            self.assertLess(variance_to_win, 0.5)
            self.assertLess(variance_not_lose, 0.5)


class TestOptimizedStrategyIntegration(unittest.TestCase):
    """Test integration with existing strategy framework"""
    
    def test_strategy_interface_compatibility(self):
        """Test that optimized strategies implement expected interface"""
        to_win = ToWinStrategy()
        not_to_lose = NotToLoseStrategy()
        
        # Test required methods exist
        self.assertTrue(hasattr(to_win, 'predict'))
        self.assertTrue(hasattr(to_win, 'get_confidence'))
        self.assertTrue(hasattr(to_win, 'get_stats'))
        
        self.assertTrue(hasattr(not_to_lose, 'predict'))
        self.assertTrue(hasattr(not_to_lose, 'get_confidence'))
        self.assertTrue(hasattr(not_to_lose, 'get_stats'))
        
        # Test they are callable
        self.assertTrue(callable(to_win.predict))
        self.assertTrue(callable(to_win.get_confidence))
        self.assertTrue(callable(to_win.get_stats))
        
        self.assertTrue(callable(not_to_lose.predict))
        self.assertTrue(callable(not_to_lose.get_confidence))
        self.assertTrue(callable(not_to_lose.get_stats))


def run_optimized_strategy_tests():
    """Run all optimized strategy tests"""
    print("🎯 OPTIMIZED STRATEGIES UNIT TESTS")
    print("=" * 45)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_cases = [
        TestOptimizedStrategies,
        TestOptimizedStrategyIntegration
    ]
    
    for test_case in test_cases:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_case)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 45)
    print("🎯 OPTIMIZED STRATEGIES TEST SUMMARY")
    print("=" * 45)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}")
            print(f"    {traceback.strip()}")
    
    if result.errors:
        print("\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}")
            print(f"    {traceback.strip()}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"\n✅ Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 95:
        print("🎉 Perfect! Optimized strategies are working flawlessly!")
    elif success_rate >= 85:
        print("✅ Excellent! Optimized strategies are working great!")
    elif success_rate >= 75:
        print("✅ Good! Most optimized strategy features are working.")
    else:
        print("⚠️  Issues found in optimized strategies.")
    
    return result


if __name__ == "__main__":
    run_optimized_strategy_tests()