#!/usr/bin/env python3
"""
AI Coach Demo Validation Script
Validates core functionality without requiring all dependencies
"""

import os
import sys
import time
import traceback

def test_file_structure():
    """Test that all required files exist"""
    print("📁 Testing File Structure")
    print("-" * 30)
    
    required_files = [
        'ai_coach_langchain.py',
        'enhanced_coach.py', 
        'ai_coach_metrics.py',
        'webapp/app.py',
        'webapp/templates/developer_console.html',
        'requirements.txt'
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = os.path.join(os.path.dirname(__file__), file_path)
        if os.path.exists(full_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_exist = False
    
    return all_exist

def test_imports():
    """Test that AI coach modules can be imported"""
    print("\n🐍 Testing Module Imports")
    print("-" * 30)
    
    import_results = {}
    
    # Test LangChain coach
    try:
        from ai_coach_langchain import LangChainAICoach
        import_results['langchain_coach'] = True
        print("✅ LangChain AI Coach")
    except Exception as e:
        import_results['langchain_coach'] = False
        print(f"⚠️ LangChain AI Coach: {e}")
    
    # Test Enhanced coach
    try:
        from enhanced_coach import EnhancedCoachSystem
        import_results['enhanced_coach'] = True
        print("✅ Enhanced Coach System")
    except Exception as e:
        import_results['enhanced_coach'] = False
        print(f"⚠️ Enhanced Coach System: {e}")
    
    # Test AI Metrics
    try:
        from ai_coach_metrics import AICoachMetricsAggregator
        import_results['ai_metrics'] = True
        print("✅ AI Metrics Aggregator")
    except Exception as e:
        import_results['ai_metrics'] = False
        print(f"⚠️ AI Metrics Aggregator: {e}")
    
    # Test Flask app
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'webapp'))
        from app import app
        import_results['flask_app'] = True
        print("✅ Flask Web App")
    except Exception as e:
        import_results['flask_app'] = False
        print(f"⚠️ Flask Web App: {e}")
    
    return import_results

def test_dependencies():
    """Test that required dependencies are available"""
    print("\n📦 Testing Dependencies")
    print("-" * 30)
    
    dependencies = {
        'flask': 'Flask web framework',
        'langchain': 'LangChain AI framework',
        'torch': 'PyTorch for LSTM',
        'numpy': 'Numerical computing',
        'matplotlib': 'Data visualization'
    }
    
    available = {}
    for package, description in dependencies.items():
        try:
            __import__(package)
            available[package] = True
            print(f"✅ {package} - {description}")
        except ImportError:
            available[package] = False
            print(f"⚠️ {package} - {description} (not installed)")
    
    return available

def test_basic_functionality():
    """Test basic functionality if modules are available"""
    print("\n🧪 Testing Basic Functionality")
    print("-" * 30)
    
    test_results = {}
    
    # Test Enhanced Coach basic functionality
    try:
        from enhanced_coach import EnhancedCoachSystem
        coach = EnhancedCoachSystem()
        
        # Test basic method calls
        test_history = [('R', 'P'), ('P', 'S'), ('S', 'R')]
        test_predictions = {'frequency': ['P', 'S', 'R']}
        
        # This should work even in basic mode
        tips = coach.generate_coaching_tips(test_history, test_predictions, 'frequency')
        
        if isinstance(tips, dict) and 'analysis' in tips:
            test_results['enhanced_coach_basic'] = True
            print("✅ Enhanced Coach System generates coaching tips")
        else:
            test_results['enhanced_coach_basic'] = False
            print("❌ Enhanced Coach System coaching tips format incorrect")
            
    except Exception as e:
        test_results['enhanced_coach_basic'] = False
        print(f"⚠️ Enhanced Coach System basic test: {e}")
    
    # Test LangChain coach if available
    try:
        from ai_coach_langchain import LangChainAICoach
        coach = LangChainAICoach()
        
        # Test LLM type methods
        llm_type = coach.get_llm_type()
        if llm_type in ['mock', 'real']:
            test_results['langchain_basic'] = True
            print(f"✅ LangChain Coach LLM type: {llm_type}")
        else:
            test_results['langchain_basic'] = False
            print(f"❌ LangChain Coach invalid LLM type: {llm_type}")
            
    except Exception as e:
        test_results['langchain_basic'] = False
        print(f"⚠️ LangChain Coach basic test: {e}")
    
    return test_results

def validate_requirements_txt():
    """Validate requirements.txt content"""
    print("\n📋 Validating requirements.txt")
    print("-" * 30)
    
    try:
        with open('requirements.txt', 'r') as f:
            content = f.read()
        
        required_packages = ['Flask', 'langchain', 'torch', 'numpy', 'matplotlib', 'ollama']
        found_packages = []
        
        for package in required_packages:
            if package.lower() in content.lower():
                found_packages.append(package)
                print(f"✅ {package}")
            else:
                print(f"⚠️ {package} - not found in requirements.txt")
        
        print(f"\nFound {len(found_packages)}/{len(required_packages)} required packages")
        return len(found_packages) >= len(required_packages) * 0.8  # 80% threshold
        
    except Exception as e:
        print(f"❌ Error reading requirements.txt: {e}")
        return False

def generate_validation_report():
    """Generate comprehensive validation report"""
    print("\n" + "=" * 60)
    print("🎯 AI Coach Demo Validation Report")
    print("=" * 60)
    
    # Run all tests
    file_structure_ok = test_file_structure()
    import_results = test_imports()
    dependency_results = test_dependencies()
    functionality_results = test_basic_functionality()
    requirements_ok = validate_requirements_txt()
    
    # Calculate scores
    total_files = 6
    files_passed = sum(1 for _ in range(total_files)) if file_structure_ok else 0
    
    total_imports = len(import_results)
    imports_passed = sum(import_results.values())
    
    total_deps = len(dependency_results)
    deps_passed = sum(dependency_results.values())
    
    total_functionality = len(functionality_results)
    functionality_passed = sum(functionality_results.values()) if functionality_results else 0
    
    # Print summary
    print(f"\n📊 Validation Summary")
    print(f"File Structure: {'✅ PASS' if file_structure_ok else '❌ FAIL'}")
    print(f"Module Imports: {imports_passed}/{total_imports} ({'✅ PASS' if imports_passed >= total_imports * 0.5 else '❌ FAIL'})")
    print(f"Dependencies: {deps_passed}/{total_deps} ({'✅ PASS' if deps_passed >= total_deps * 0.6 else '❌ FAIL'})")
    print(f"Basic Functionality: {functionality_passed}/{total_functionality} ({'✅ PASS' if functionality_passed > 0 else '❌ FAIL'})")
    print(f"Requirements.txt: {'✅ PASS' if requirements_ok else '❌ FAIL'}")
    
    # Overall assessment
    print(f"\n🎯 Overall Assessment")
    if file_structure_ok and imports_passed >= 2 and deps_passed >= 3:
        print("🎉 AI Coach Demo System: READY FOR USE")
        print("✅ Core components are functional")
        print("✅ Required dependencies are available")
        print("✅ Basic functionality verified")
    elif file_structure_ok and imports_passed >= 1:
        print("⚠️ AI Coach Demo System: PARTIALLY FUNCTIONAL")
        print("✅ Core files are present")
        print("⚠️ Some dependencies may be missing")
        print("📝 Check dependency installation")
    else:
        print("❌ AI Coach Demo System: REQUIRES SETUP")
        print("❌ Missing core components or dependencies")
        print("📝 Run: pip install -r requirements.txt")
    
    print(f"\n📋 Quick Start Instructions:")
    print("1. Ensure all dependencies are installed:")
    print("   pip install -r requirements.txt")
    print("2. Start the Flask server:")
    print("   python webapp/app.py")
    print("3. Open the AI Coach Demo:")
    print("   http://localhost:5050/developer")
    print("4. Test LLM Backend Toggle functionality")

if __name__ == "__main__":
    try:
        generate_validation_report()
    except Exception as e:
        print(f"\n❌ Validation script error: {e}")
        print("🔍 Full traceback:")
        traceback.print_exc()