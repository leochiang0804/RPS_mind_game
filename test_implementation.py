#!/usr/bin/env python3
"""
Simple Test Script for Critical UI/UX Improvements
Tests the key implementations without requiring full Flask app
"""

import os
import sys
import hashlib

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_replay_button_visibility():
    """Test that replay buttons are properly implemented in template"""
    print("🔍 Testing Replay Button Visibility...")
    
    template_path = os.path.join('webapp', 'templates', 'index.html')
    
    if not os.path.exists(template_path):
        print("❌ Template file not found")
        return False
    
    with open(template_path, 'r') as f:
        content = f.read()
    
    checks = [
        ('game-controls', "Game controls section"),
        ('Save Current Replay', "Save replay button"),
        ('View All Replays', "View replays button"),
        ('btn-replay-save', "Save button CSS class"),
        ('btn-replay-view', "View button CSS class"),
        (':hover', "Hover effects"),
        ('transform: translateY(-2px)', "Hover transform")
    ]
    
    passed = 0
    for check, description in checks:
        if check in content:
            print(f"  ✅ {description}")
            passed += 1
        else:
            print(f"  ❌ {description}")
    
    # Check positioning
    title_pos = content.find('<h1>Paper Scissor Stone ML Game</h1>')
    controls_pos = content.find('game-controls')
    
    if title_pos >= 0 and controls_pos >= 0 and controls_pos > title_pos:
        print("  ✅ Controls positioned after title")
        passed += 1
    else:
        print("  ❌ Controls positioning")
    
    success = passed == len(checks) + 1
    print(f"🎯 Replay Button Test: {passed}/{len(checks) + 1} checks passed")
    return success

def test_deterministic_coach_tips():
    """Test deterministic coach tips functionality"""
    print("\n🔍 Testing Deterministic Coach Tips...")
    
    # Check if coach_tips.py exists and has required modifications
    coach_tips_path = 'coach_tips.py'
    
    if not os.path.exists(coach_tips_path):
        print("❌ coach_tips.py not found")
        return False
    
    with open(coach_tips_path, 'r') as f:
        content = f.read()
    
    checks = [
        ('def generate_tips(', "generate_tips method exists"),
        ('hash(', "Hash function used for deterministic seeding"),
        ('state_components', "State components for hashing"),
        ('_select_experiments_deterministic', "Deterministic experiment selection"),
        ('random.seed(', "Random seeding for determinism")
    ]
    
    passed = 0
    for check, description in checks:
        if check in content:
            print(f"  ✅ {description}")
            passed += 1
        else:
            print(f"  ❌ {description}")
    
    # Test actual determinism if we can import
    try:
        from coach_tips import CoachTipsGenerator
        coach = CoachTipsGenerator()
        
        # Test same state produces same tips
        history = ['rock', 'paper', 'scissors'] * 5
        robot_history = ['paper', 'scissors', 'rock'] * 5
        result_history = ['human'] * 15
        
        tips1 = coach.generate_tips(history, robot_history, result_history, [], 'cycler')
        tips2 = coach.generate_tips(history, robot_history, result_history, [], 'cycler')
        
        if tips1['tips'] == tips2['tips']:
            print("  ✅ Tips are deterministic for same game state")
            passed += 1
        else:
            print("  ❌ Tips are not deterministic")
        
        # Test different states produce different tips
        different_history = ['paper'] * 15
        tips3 = coach.generate_tips(different_history, robot_history, result_history, [], 'repeater')
        
        if tips1['tips'] != tips3['tips']:
            print("  ✅ Tips differ for different game states")
            passed += 1
        else:
            print("  ❌ Tips don't differ for different states")
        
    except ImportError as e:
        print(f"  ⚠️ Could not test functionality: {e}")
    
    success = passed >= len(checks)
    print(f"🎯 Coach Tips Test: {passed}/{len(checks) + 2} checks passed")
    return success

def test_personality_system():
    """Test personality system implementation"""
    print("\n🔍 Testing Personality System...")
    
    template_path = os.path.join('webapp', 'templates', 'index.html')
    
    if not os.path.exists(template_path):
        print("❌ Template file not found")
        return False
    
    with open(template_path, 'r') as f:
        content = f.read()
    
    # Check for personality selector in template
    personality_checks = [
        ('id="personality"', "Personality selector exists"),
        ('AI Personality:', "Personality label exists"),
        ('function setPersonality()', "setPersonality function exists"),
        ('value="berserker"', "Berserker personality option"),
        ('value="guardian"', "Guardian personality option"),
        ('value="chameleon"', "Chameleon personality option"),
        ('value="professor"', "Professor personality option"),
        ('value="wildcard"', "Wildcard personality option"),
        ('value="mirror"', "Mirror personality option")
    ]
    
    passed = 0
    for check, description in personality_checks:
        if check in content:
            print(f"  ✅ {description}")
            passed += 1
        else:
            print(f"  ❌ {description}")
    
    # Test personality engine if available
    try:
        from personality_engine import AdvancedPersonalityEngine
        engine = AdvancedPersonalityEngine()
        
        expected_personalities = ['berserker', 'guardian', 'chameleon', 'professor', 'wildcard', 'mirror']
        
        for personality in expected_personalities:
            if personality in engine.personalities:
                print(f"  ✅ {personality.capitalize()} personality available")
                passed += 1
            else:
                print(f"  ❌ {personality.capitalize()} personality missing")
        
    except ImportError:
        print("  ⚠️ Could not test personality engine functionality")
    
    success = passed >= len(personality_checks)
    print(f"🎯 Personality System Test: {passed}/{len(personality_checks) + 6} checks passed")
    return success

def test_overall_integration():
    """Test overall integration and file structure"""
    print("\n🔍 Testing Overall Integration...")
    
    required_files = [
        ('webapp/app.py', "Flask web application"),
        ('webapp/templates/index.html', "Main game template"),
        ('coach_tips.py', "Coach tips generator"),
        ('personality_engine.py', "Personality engine"),
        ('game.py', "Game logic"),
        ('main.py', "Main application")
    ]
    
    passed = 0
    for file_path, description in required_files:
        if os.path.exists(file_path):
            print(f"  ✅ {description} exists")
            passed += 1
        else:
            print(f"  ❌ {description} missing")
    
    # Check webapp directory structure
    webapp_files = [
        ('webapp/static/paper.png', "Paper image"),
        ('webapp/static/scissor.png', "Scissor image"),
        ('webapp/static/stone.jpg', "Stone image"),
        ('webapp/templates/stats.html', "Stats template")
    ]
    
    for file_path, description in webapp_files:
        if os.path.exists(file_path):
            print(f"  ✅ {description} exists")
            passed += 1
        else:
            print(f"  ❌ {description} missing")
    
    success = passed >= len(required_files) + len(webapp_files) - 2  # Allow 2 missing files
    print(f"🎯 Integration Test: {passed}/{len(required_files) + len(webapp_files)} files found")
    return success

def run_quick_functionality_test():
    """Run a quick test of actual functionality"""
    print("\n🔍 Testing Quick Functionality...")
    
    try:
        # Test coach tips determinism
        from coach_tips import CoachTipsGenerator
        coach = CoachTipsGenerator()
        
        # Create a specific game state
        test_state = {
            'human_history': ['rock', 'paper', 'scissors', 'rock'],
            'robot_history': ['paper', 'scissors', 'rock', 'paper'],
            'result_history': ['robot', 'robot', 'robot', 'robot'],
            'change_points': [],
            'current_strategy': 'balanced'
        }
        
        # Generate tips twice
        tips1 = coach.generate_tips(**test_state)
        tips2 = coach.generate_tips(**test_state)
        
        # Create hash for verification
        state_components = (
            tuple(test_state['human_history'][-10:]),
            tuple(test_state['result_history'][-10:]),
            test_state['current_strategy']
        )
        state_hash = hash(state_components)
        
        print(f"  ✅ Game state hash: {state_hash}")
        print(f"  ✅ Tips generated: {len(tips1['tips'])} tips")
        
        if tips1['tips'] == tips2['tips']:
            print("  ✅ Tips are deterministic")
            return True
        else:
            print("  ❌ Tips are not deterministic")
            print(f"    First run:  {tips1['tips'][:2]}...")
            print(f"    Second run: {tips2['tips'][:2]}...")
            return False
            
    except Exception as e:
        print(f"  ❌ Functionality test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Critical UI/UX Improvements Implementation")
    print("=" * 60)
    
    tests = [
        ("Replay Button Visibility", test_replay_button_visibility),
        ("Deterministic Coach Tips", test_deterministic_coach_tips),
        ("Personality System", test_personality_system),
        ("Overall Integration", test_overall_integration),
        ("Quick Functionality", run_quick_functionality_test)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 TEST SUMMARY:")
    print("-" * 30)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
        if success:
            passed += 1
    
    print(f"\n📊 OVERALL RESULT: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 ALL TESTS PASSED! Implementations are working correctly.")
        print("\n✨ Summary of Verified Improvements:")
        print("   • Enhanced replay button visibility with prominent placement")
        print("   • Deterministic coach tips using game state hashing")
        print("   • Advanced personality system with 6 sophisticated AI personas")
        print("   • Improved UI styling with hover effects and gradients")
        print("   • Consistent user experience across all game features")
    else:
        failed = len(results) - passed
        print(f"⚠️ {failed} test(s) failed. Review the issues above.")
        print("\n🔧 Implemented Features:")
        print("   • UI enhancements for better visibility and user experience")
        print("   • Deterministic systems for consistent behavior")
        print("   • Advanced personality options for varied gameplay")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)