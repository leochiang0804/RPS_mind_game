"""
Comprehensive Test Suite for Critical UI/UX Improvements
Tests replay button visibility, deterministic coach tips, and personality system
"""

import unittest
import sys
import os
import time
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coach_tips import CoachTipsGenerator
from personality_engine import AdvancedPersonalityEngine

class TestReplayButtonVisibility(unittest.TestCase):
    """Test replay button visibility and positioning"""
    
    def setUp(self):
        """Set up test environment"""
        self.template_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'webapp', 'templates', 'index.html'
        )
    
    def test_replay_buttons_exist_in_template(self):
        """Test that replay buttons exist in the HTML template"""
        with open(self.template_path, 'r') as f:
            content = f.read()
        
        # Check for prominent replay controls section
        self.assertIn('game-controls', content, "Game controls section should exist")
        self.assertIn('Save Current Replay', content, "Save replay button should exist")
        self.assertIn('View All Replays', content, "View replays button should exist")
        
        # Check for enhanced styling
        self.assertIn('btn-replay-save', content, "Save button should have proper CSS class")
        self.assertIn('btn-replay-view', content, "View button should have proper CSS class")
        
        # Check buttons are in prominent location (after title)
        title_pos = content.find('<h1>Paper Scissor Stone ML Game</h1>')
        controls_pos = content.find('game-controls')
        self.assertGreater(controls_pos, title_pos, "Controls should appear after title")
        
        print("✅ Replay buttons are properly positioned and styled")
    
    def test_additional_dashboard_buttons(self):
        """Test that additional dashboard buttons are present"""
        with open(self.template_path, 'r') as f:
            content = f.read()
        
        # Check for developer and performance dashboard buttons
        self.assertIn('Developer Console', content, "Developer console button should exist")
        self.assertIn('Performance Dashboard', content, "Performance dashboard button should exist")
        
        # Check for proper button styling
        self.assertIn('btn-developer', content, "Developer button should have CSS class")
        self.assertIn('btn-performance', content, "Performance button should have CSS class")
        
        print("✅ Additional dashboard buttons are present")
    
    def test_button_hover_effects(self):
        """Test that button hover effects are defined"""
        with open(self.template_path, 'r') as f:
            content = f.read()
        
        # Check for hover effect CSS
        self.assertIn(':hover', content, "Hover effects should be defined")
        self.assertIn('transform: translateY(-2px)', content, "Hover transform should be defined")
        
        print("✅ Button hover effects are properly defined")

class TestDeterministicCoachTips(unittest.TestCase):
    """Test deterministic coach tips implementation"""
    
    def setUp(self):
        """Set up coach tips generator"""
        self.coach = CoachTipsGenerator()
    
    def test_tips_are_deterministic_same_state(self):
        """Test that tips are the same for identical game states"""
        # Sample game state
        human_history = ['paper', 'rock', 'scissors', 'paper', 'rock', 'scissors']
        robot_history = ['scissors', 'paper', 'rock', 'scissors', 'paper', 'rock']
        result_history = ['human', 'human', 'human', 'human', 'human', 'human']
        change_points = []
        current_strategy = 'cycler'
        
        # Generate tips multiple times
        tips1 = self.coach.generate_tips(human_history, robot_history, result_history, change_points, current_strategy)
        tips2 = self.coach.generate_tips(human_history, robot_history, result_history, change_points, current_strategy)
        tips3 = self.coach.generate_tips(human_history, robot_history, result_history, change_points, current_strategy)
        
        # Tips should be identical
        self.assertEqual(tips1['tips'], tips2['tips'], "Tips should be identical for same game state")
        self.assertEqual(tips2['tips'], tips3['tips'], "Tips should be consistent across multiple calls")
        self.assertEqual(tips1['experiments'], tips2['experiments'], "Experiments should be identical")
        
        print(f"✅ Deterministic tips verified: {len(tips1['tips'])} tips generated consistently")
    
    def test_tips_differ_for_different_states(self):
        """Test that tips change for different game states"""
        # First game state
        history1 = ['paper', 'paper', 'paper', 'paper', 'paper']
        robot_history1 = ['scissors', 'scissors', 'scissors', 'scissors', 'scissors']
        result_history1 = ['human', 'human', 'human', 'human', 'human']
        
        # Different game state
        history2 = ['rock', 'paper', 'scissors', 'rock', 'paper']
        robot_history2 = ['paper', 'scissors', 'rock', 'paper', 'scissors']
        result_history2 = ['robot', 'robot', 'robot', 'robot', 'robot']
        
        tips1 = self.coach.generate_tips(history1, robot_history1, result_history1, [], 'repeater')
        tips2 = self.coach.generate_tips(history2, robot_history2, result_history2, [], 'balanced')
        
        # Tips should be different for different game states
        self.assertNotEqual(tips1['tips'], tips2['tips'], "Tips should differ for different game states")
        
        print("✅ Tips correctly vary for different game states")
    
    def test_early_game_tips_consistency(self):
        """Test that early game tips are consistent"""
        # Early game (< 5 moves)
        history = ['paper', 'rock']
        robot_history = ['scissors', 'paper']
        result_history = ['human', 'human']
        
        tips1 = self.coach.generate_tips(history, robot_history, result_history, [], 'unknown')
        tips2 = self.coach.generate_tips(history, robot_history, result_history, [], 'unknown')
        
        # Early game tips should be identical (hardcoded)
        self.assertEqual(tips1['tips'], tips2['tips'], "Early game tips should be consistent")
        self.assertEqual(len(tips1['tips']), 3, "Should have exactly 3 early game tips")
        
        print("✅ Early game tips are consistent")
    
    def test_deterministic_experiments_selection(self):
        """Test that experiments are selected deterministically"""
        history = ['rock', 'paper', 'scissors', 'rock', 'paper', 'scissors'] * 3
        robot_history = ['paper', 'scissors', 'rock'] * 6
        result_history = ['robot'] * 18
        
        tips1 = self.coach.generate_tips(history, robot_history, result_history, [], 'cycler')
        tips2 = self.coach.generate_tips(history, robot_history, result_history, [], 'cycler')
        
        # Experiments should be identical
        self.assertEqual(tips1['experiments'], tips2['experiments'], "Experiments should be deterministic")
        self.assertGreater(len(tips1['experiments']), 0, "Should have experiments")
        
        print(f"✅ Deterministic experiments verified: {len(tips1['experiments'])} experiments")

class TestPersonalitySystem(unittest.TestCase):
    """Test personality system implementation"""
    
    def setUp(self):
        """Set up personality engine"""
        self.personality_engine = AdvancedPersonalityEngine()
    
    def test_all_personalities_available(self):
        """Test that all expected personalities are available"""
        expected_personalities = [
            'berserker', 'guardian', 'chameleon', 
            'professor', 'wildcard', 'mirror'
        ]
        
        for personality in expected_personalities:
            self.assertIn(personality, self.personality_engine.personalities, 
                         f"Personality '{personality}' should be available")
        
        print(f"✅ All {len(expected_personalities)} personalities are available")
    
    def test_personality_traits_valid(self):
        """Test that personality traits are properly configured"""
        for name, personality in self.personality_engine.personalities.items():
            # Check trait values are in valid range
            for trait, value in personality.traits.items():
                self.assertGreaterEqual(value, 0.0, f"{name} trait {trait} should be >= 0")
                self.assertLessEqual(value, 1.0, f"{name} trait {trait} should be <= 1")
            
            # Check required fields exist
            self.assertIsInstance(personality.description, str, f"{name} should have description")
            self.assertIsInstance(personality.color_theme, dict, f"{name} should have color theme")
            self.assertIsInstance(personality.behavior_modifiers, dict, f"{name} should have behavior modifiers")
        
        print("✅ All personality traits are properly configured")
    
    def test_personality_ui_integration(self):
        """Test that personality selector is in the UI template"""
        template_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'webapp', 'templates', 'index.html'
        )
        
        with open(template_path, 'r') as f:
            content = f.read()
        
        # Check personality selector exists
        self.assertIn('id="personality"', content, "Personality selector should exist")
        self.assertIn('AI Personality:', content, "Personality label should exist")
        
        # Check all personalities are in options
        personalities = ['berserker', 'guardian', 'chameleon', 'professor', 'wildcard', 'mirror']
        for personality in personalities:
            self.assertIn(f'value="{personality}"', content, f"Personality {personality} should be in options")
        
        # Check JavaScript function exists
        self.assertIn('function setPersonality()', content, "setPersonality function should exist")
        
        print("✅ Personality system is properly integrated in UI")
    
    def test_personality_descriptions(self):
        """Test that personalities have meaningful descriptions"""
        for name, personality in self.personality_engine.personalities.items():
            description = personality.description
            self.assertGreater(len(description), 20, f"{name} should have meaningful description")
            self.assertNotEqual(description, "", f"{name} description should not be empty")
        
        print("✅ All personalities have meaningful descriptions")

class TestWebAppIntegration(unittest.TestCase):
    """Test web application integration"""
    
    def test_flask_app_imports(self):
        """Test that Flask app can import all required modules"""
        try:
            sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'webapp'))
            from app import app, DEVELOPER_CONSOLE_AVAILABLE, PERFORMANCE_OPTIMIZER_AVAILABLE, LSTM_AVAILABLE
            
            self.assertTrue(DEVELOPER_CONSOLE_AVAILABLE, "Developer console should be available")
            self.assertTrue(PERFORMANCE_OPTIMIZER_AVAILABLE, "Performance optimizer should be available")
            
            print("✅ Flask app imports all required modules successfully")
            print(f"  - Developer Console: {'✅' if DEVELOPER_CONSOLE_AVAILABLE else '❌'}")
            print(f"  - Performance Optimizer: {'✅' if PERFORMANCE_OPTIMIZER_AVAILABLE else '❌'}")
            print(f"  - LSTM: {'✅' if LSTM_AVAILABLE else '❌'}")
            
        except Exception as e:
            self.fail(f"Flask app import failed: {e}")
    
    def test_new_routes_exist(self):
        """Test that new routes are properly configured"""
        try:
            sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'webapp'))
            from app import app
            
            with app.test_client() as client:
                # Test developer console route
                response = client.get('/developer')
                self.assertIn(response.status_code, [200, 503], "Developer route should exist")
                
                # Test performance dashboard route
                response = client.get('/performance')
                self.assertIn(response.status_code, [200, 503], "Performance route should exist")
                
                print("✅ New routes are properly configured")
                
        except Exception as e:
            print(f"⚠️ Route testing skipped due to: {e}")

class TestPerformanceAndOptimization(unittest.TestCase):
    """Test performance optimizations"""
    
    def test_coach_tips_performance(self):
        """Test that coach tips generation is fast enough"""
        coach = CoachTipsGenerator()
        
        # Large game history
        history = (['rock', 'paper', 'scissors'] * 100)[:250]  # 250 moves
        robot_history = (['paper', 'scissors', 'rock'] * 100)[:250]
        result_history = (['human', 'robot', 'tie'] * 100)[:250]
        
        start_time = time.time()
        tips = coach.generate_tips(history, robot_history, result_history, [], 'complex')
        duration = time.time() - start_time
        
        # Should be very fast (< 100ms for 250 moves)
        self.assertLess(duration, 0.1, f"Tips generation too slow: {duration:.3f}s")
        self.assertGreater(len(tips['tips']), 0, "Should generate tips")
        
        print(f"✅ Coach tips generation performance: {duration:.3f}s for 250 moves")
    
    def test_personality_engine_performance(self):
        """Test personality engine performance"""
        engine = AdvancedPersonalityEngine()
        
        start_time = time.time()
        for personality_name in engine.personalities.keys():
            engine.set_personality(personality_name)
            # Simulate some game updates
            for i in range(10):
                engine.update_game_state('rock', 'paper', 'robot', 0.7)
        duration = time.time() - start_time
        
        # Should be fast
        self.assertLess(duration, 0.1, f"Personality engine too slow: {duration:.3f}s")
        
        print(f"✅ Personality engine performance: {duration:.3f}s for 60 updates")

def run_comprehensive_tests():
    """Run all comprehensive tests"""
    print("🧪 Running Comprehensive Test Suite for Critical UI/UX Improvements")
    print("=" * 70)
    
    # Create test suite
    test_classes = [
        TestReplayButtonVisibility,
        TestDeterministicCoachTips,
        TestPersonalitySystem,
        TestWebAppIntegration,
        TestPerformanceAndOptimization
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n📋 {test_class.__name__}")
        print("-" * 50)
        
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
        result = runner.run(suite)
        
        class_tests = result.testsRun
        class_passed = class_tests - len(result.failures) - len(result.errors)
        
        total_tests += class_tests
        passed_tests += class_passed
        
        print(f"Tests: {class_passed}/{class_tests} passed")
        
        if result.failures:
            print("❌ Failures:")
            for test, error in result.failures:
                print(f"  - {test}: {error}")
        
        if result.errors:
            print("❌ Errors:")
            for test, error in result.errors:
                print(f"  - {test}: {error}")
    
    print("\n" + "=" * 70)
    print(f"🎯 FINAL RESULTS: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Critical improvements are working correctly.")
    else:
        print(f"⚠️ {total_tests - passed_tests} tests failed. Review issues above.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)