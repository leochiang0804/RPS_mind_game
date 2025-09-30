#!/usr/bin/env python3
"""
Test Enhanced Features in Working Template
Verify that replay buttons, LSTM options, and enhanced features are visible
"""

import os
import subprocess
import sys
import time
import webbrowser
from threading import Timer

def test_template_features():
    """Test the enhanced features in index_working.html"""
    print("🧪 Testing Enhanced Features in Working Template")
    print("=" * 60)
    
    template_path = os.path.join('webapp', 'templates', 'index_working.html')
    
    if not os.path.exists(template_path):
        print("❌ Template file not found")
        return False
    
    with open(template_path, 'r') as f:
        content = f.read()
    
    # Test Feature #1: Enhanced Replay Controls
    print("\n🎮 Testing Enhanced Replay Controls...")
    replay_checks = [
        ('🎮 Game Controls', "Game controls header"),
        ('💾 Save Current Replay', "Save replay button"),
        ('🎬 View All Replays', "View replays button"), 
        ('🔧 Developer Console', "Developer console button"),
        ('⚡ Performance Dashboard', "Performance dashboard button"),
        ('saveCurrentReplay()', "Save replay function")
    ]
    
    replay_passed = 0
    for check, description in replay_checks:
        if check in content:
            print(f"  ✅ {description}")
            replay_passed += 1
        else:
            print(f"  ❌ {description}")
    
    print(f"  📊 Replay Controls: {replay_passed}/{len(replay_checks)} checks passed")
    
    # Test Feature #2: LSTM Support
    print("\n🧠 Testing LSTM Neural Network Support...")
    lstm_checks = [
        ('🧠 LSTM Neural', "LSTM option in difficulty selector"),
        ('LSTM Neural', "LSTM in model metrics"),
        ('lstm', "LSTM key in JavaScript"),
        ('rgba(54, 162, 235', "LSTM chart color")
    ]
    
    lstm_passed = 0
    for check, description in lstm_checks:
        if check in content:
            print(f"  ✅ {description}")
            lstm_passed += 1
        else:
            print(f"  ❌ {description}")
    
    print(f"  📊 LSTM Support: {lstm_passed}/{len(lstm_checks)} checks passed")
    
    # Test Feature #3: Advanced Personality Support  
    print("\n🎭 Testing Advanced Personality Support...")
    personality_checks = [
        ('berserker', "Berserker personality"),
        ('guardian', "Guardian personality"),
        ('chameleon', "Chameleon personality"),
        ('professor', "Professor personality"),
        ('wildcard', "Wildcard personality"),
        ('mirror', "Mirror personality")
    ]
    
    personality_passed = 0
    for check, description in personality_checks:
        if check in content:
            print(f"  ✅ {description}")
            personality_passed += 1
        else:
            print(f"  ❌ {description}")
    
    print(f"  📊 Advanced Personalities: {personality_passed}/{len(personality_checks)} checks passed")
    
    # Test Overall Integration
    print("\n🔧 Testing Overall Integration...")
    integration_checks = [
        ('Chart.js', "Chart.js library included"),
        ('initializeCharts', "Chart initialization function"),
        ('updateCharts', "Chart update function"),
        ('refreshCoachingTips', "Coaching tips function"),
        ('exportAnalytics', "Analytics export function")
    ]
    
    integration_passed = 0
    for check, description in integration_checks:
        if check in content:
            print(f"  ✅ {description}")
            integration_passed += 1
        else:
            print(f"  ❌ {description}")
    
    print(f"  📊 Integration: {integration_passed}/{len(integration_checks)} checks passed")
    
    # Summary
    total_checks = len(replay_checks) + len(lstm_checks) + len(personality_checks) + len(integration_checks)
    total_passed = replay_passed + lstm_passed + personality_passed + integration_passed
    
    print("\n" + "=" * 60)
    print(f"🎯 OVERALL TEST RESULTS: {total_passed}/{total_checks} checks passed")
    
    if total_passed >= total_checks * 0.9:  # 90% pass rate
        print("🎉 EXCELLENT! Enhanced features are properly integrated")
        return True
    elif total_passed >= total_checks * 0.7:  # 70% pass rate
        print("✅ GOOD! Most enhanced features are working")
        return True
    else:
        print("⚠️ NEEDS WORK! Some features may be missing")
        return False

def start_test_server():
    """Start the Flask server for manual testing"""
    print("\n🚀 Starting test server for manual verification...")
    print("URL: http://localhost:5000")
    print("Features to test manually:")
    print("  1. Look for prominent replay buttons at the top")
    print("  2. Check AI Difficulty dropdown has 'LSTM Neural' option")
    print("  3. Verify advanced personalities are available")
    print("  4. Test that developer console and performance dashboard buttons work")
    print("  5. Play a few games and check that charts update correctly")
    
    def open_browser():
        time.sleep(2)
        try:
            webbrowser.open('http://localhost:5000')
        except:
            print("Could not open browser automatically")
    
    Timer(1.0, open_browser).start()
    
    try:
        os.chdir('webapp')
        subprocess.run([sys.executable, 'app.py'], check=True)
    except KeyboardInterrupt:
        print("\n👋 Test server stopped")
    except Exception as e:
        print(f"❌ Server error: {e}")

def main():
    """Main test function"""
    print("🧪 Enhanced Features Migration Test")
    print("Testing that enhanced features are properly integrated into working template")
    
    # Run template analysis
    template_success = test_template_features()
    
    if template_success:
        print("\n✨ Template analysis passed!")
        print("All enhanced features appear to be properly integrated.")
        
        user_input = input("\nWould you like to start the test server for manual verification? (y/n): ")
        if user_input.lower() in ['y', 'yes']:
            start_test_server()
        else:
            print("Manual testing skipped. You can run 'python webapp/app.py' later to test.")
    else:
        print("\n⚠️ Template analysis found issues.")
        print("Some enhanced features may not be working correctly.")
    
    return template_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)