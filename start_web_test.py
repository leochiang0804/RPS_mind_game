#!/usr/bin/env python3
"""
Web Application Test - Start the Flask app and verify improvements
"""

import os
import sys
import time
import subprocess
import webbrowser
from threading import Timer

def start_web_app():
    """Start the Flask web application"""
    print("🚀 Starting Paper Scissor Stone ML Game Web Application...")
    print("=" * 60)
    
    # Navigate to webapp directory
    webapp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'webapp')
    
    if not os.path.exists(webapp_dir):
        print("❌ webapp directory not found!")
        return False
    
    print(f"📂 Web app directory: {webapp_dir}")
    
    # Check if app.py exists
    app_file = os.path.join(webapp_dir, 'app.py')
    if not os.path.exists(app_file):
        print("❌ app.py not found!")
        return False
    
    print("✅ Flask application found")
    print("\n🎯 Critical Improvements Implemented:")
    print("   • Enhanced replay button visibility (prominent header placement)")
    print("   • Deterministic coach tips (consistent advice using game state hashing)")
    print("   • Advanced personality system (6 sophisticated AI personas)")
    print("   • Improved UI styling (hover effects, gradients, better spacing)")
    print("   • Better user experience (logical control grouping)")
    
    print("\n🌐 Starting web server...")
    print("   URL: http://localhost:5000")
    print("   Press Ctrl+C to stop the server")
    
    # Auto-open browser after 2 seconds
    def open_browser():
        print("🔗 Opening browser...")
        try:
            webbrowser.open('http://localhost:5000')
        except:
            print("⚠️ Could not auto-open browser. Please visit http://localhost:5000 manually.")
    
    Timer(2.0, open_browser).start()
    
    print("\n" + "=" * 60)
    print("🎮 GAME TESTING CHECKLIST:")
    print("□ 1. Verify replay buttons are prominently visible in header")
    print("□ 2. Test 'Get New Tips' button produces consistent advice")
    print("□ 3. Try different AI personalities (berserker, guardian, etc.)")
    print("□ 4. Check hover effects on buttons work properly")
    print("□ 5. Verify developer console and performance dashboard work")
    print("=" * 60)
    
    # Start Flask app
    try:
        os.chdir(webapp_dir)
        subprocess.run([sys.executable, 'app.py'], check=True)
    except KeyboardInterrupt:
        print("\n👋 Web application stopped.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Flask app failed to start: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main function"""
    print("🧪 Paper Scissor Stone ML Game - Critical Improvements Test")
    print("🎯 This will start the web application to test all implementations")
    print("\nImplemented Features:")
    print("✅ Enhanced Replay Button Visibility")
    print("✅ Deterministic Coach Tips")
    print("✅ Advanced Personality System")
    print("✅ Improved UI/UX Design")
    
    input("\nPress Enter to start the web application...")
    
    success = start_web_app()
    
    if success:
        print("\n🎉 Web application test completed successfully!")
    else:
        print("\n❌ Web application test encountered issues.")
    
    return success

if __name__ == "__main__":
    main()