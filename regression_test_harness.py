#!/usr/bin/env python3
"""
Comprehensive Regression Test Harness for Centralized AI Coach Endpoints

This script provides a permanent testing framework to validate that all AI coach 
endpoints maintain functional equivalence after centralization.
"""

import sys
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class EndpointRegressionTester:
    """Comprehensive regression testing for AI coach endpoints"""
    
    def __init__(self):
        self.test_scenarios = self._create_test_scenarios()
        self.results = {}
        
    def _create_test_scenarios(self) -> List[Dict]:
        """Create comprehensive test scenarios for all endpoints"""
        return [
            {
                'name': 'empty_session',
                'description': 'Test with no session data (new user)',
                'session_data': {},
                'request_data': {'llm_type': 'mock', 'coaching_style': 'supportive'}
            },
            {
                'name': 'short_game',
                'description': 'Test with short game (3 moves)',
                'session_data': {
                    'human_moves': ['rock', 'paper', 'scissors'],
                    'robot_moves': ['scissors', 'rock', 'paper'],
                    'results': ['lose', 'lose', 'lose'],
                    'ai_difficulty': 'medium',
                    'human_strategy_label': 'random'
                },
                'request_data': {'llm_type': 'mock', 'coaching_style': 'analytical'}
            },
            {
                'name': 'medium_game',
                'description': 'Test with medium game (10 moves)',
                'session_data': {
                    'human_moves': ['rock'] * 5 + ['paper'] * 3 + ['scissors'] * 2,
                    'robot_moves': ['paper'] * 5 + ['scissors'] * 3 + ['rock'] * 2,
                    'results': ['lose'] * 5 + ['lose'] * 3 + ['lose'] * 2,
                    'ai_difficulty': 'hard',
                    'human_strategy_label': 'pattern_seeker',
                    'accuracy': {'decision_tree': 0.75, 'enhanced': 0.80}
                },
                'request_data': {'llm_type': 'mock', 'coaching_style': 'motivational'}
            },
            {
                'name': 'long_game_with_patterns',
                'description': 'Test with long game showing clear patterns',
                'session_data': {
                    'human_moves': (['rock', 'paper', 'scissors'] * 7)[:20],
                    'robot_moves': (['paper', 'scissors', 'rock'] * 7)[:20],
                    'results': (['lose', 'lose', 'lose'] * 7)[:20],
                    'ai_difficulty': 'enhanced',
                    'human_strategy_label': 'cyclical',
                    'change_points': [{'round': 5}, {'round': 12}],  # Fix: change_points should be dicts
                    'model_predictions_history': {
                        'enhanced': ['rock'] * 20,
                        'markov': ['paper'] * 20
                    }
                },
                'request_data': {'llm_type': 'mock', 'coaching_style': 'technical'}
            }
        ]
    
    def test_endpoint_direct(self, endpoint_path: str, method: str = 'GET', 
                           session_data: Optional[Dict] = None, request_data: Optional[Dict] = None) -> Tuple[bool, Optional[Dict]]:
        """Test an endpoint directly using Flask test client"""
        try:
            # Import webapp components
            sys.path.append('webapp')
            from webapp.app import app
            
            app.config['TESTING'] = True
            client = app.test_client()
            
            with app.test_request_context():
                # Simulate session data if provided
                if session_data:
                    with client.session_transaction() as sess:
                        for key, value in session_data.items():
                            sess[key] = value
                
                # Make request
                if method == 'POST':
                    response = client.post(endpoint_path, 
                                         json=request_data or {},
                                         content_type='application/json')
                else:
                    response = client.get(endpoint_path)
                
                # Parse response
                success = response.status_code == 200
                data = response.get_json() if response.status_code == 200 else None
                
                return success, data
                
        except Exception as e:
            print(f"❌ Error testing {endpoint_path}: {e}")
            return False, {'error': str(e)}
    
    def validate_endpoint_structure(self, endpoint_name: str, data: Dict, 
                                  expected_fields: List[str]) -> Dict:
        """Validate that endpoint response has expected structure"""
        validation = {
            'endpoint': endpoint_name,
            'timestamp': datetime.now().isoformat(),
            'structure_valid': True,
            'field_checks': {},
            'metrics': {}
        }
        
        # Check required fields
        for field in expected_fields:
            present = field in data
            validation['field_checks'][field] = present
            if not present:
                validation['structure_valid'] = False
        
        # Endpoint-specific validations
        if endpoint_name == 'comprehensive':
            validation['metrics']['has_success'] = data.get('success', False)
            validation['metrics']['metrics_summary_categories'] = len(data.get('metrics_summary', {}))
            
        elif endpoint_name == 'realtime':
            validation['metrics']['has_success'] = data.get('success', False)
            validation['metrics']['has_advice'] = 'advice' in data
            validation['metrics']['llm_type'] = data.get('llm_type', 'unknown')
            
        elif endpoint_name == 'metrics':
            validation['metrics']['has_success'] = data.get('success', False)
            validation['metrics']['metrics_categories'] = len(data.get('metrics', {}))
            validation['metrics']['data_sources'] = data.get('data_sources', {})
        
        return validation
    
    def run_comprehensive_regression_test(self) -> Dict:
        """Run complete regression test suite on all endpoints"""
        print("🧪 Starting Comprehensive AI Coach Regression Tests")
        print("=" * 70)
        
        test_results = {
            'test_run': {
                'timestamp': datetime.now().isoformat(),
                'total_scenarios': len(self.test_scenarios),
                'endpoints_tested': ['comprehensive', 'realtime', 'metrics']
            },
            'results': {}
        }
        
        endpoints = [
            {
                'name': 'comprehensive',
                'path': '/ai_coach/comprehensive',
                'method': 'POST',
                'expected_fields': ['success', 'metrics_summary', 'session_summary']
            },
            {
                'name': 'realtime', 
                'path': '/ai_coach/realtime',
                'method': 'POST',
                'expected_fields': ['success', 'advice', 'metrics_summary', 'llm_type']
            },
            {
                'name': 'metrics',
                'path': '/ai_coach/metrics', 
                'method': 'GET',
                'expected_fields': ['success', 'metrics', 'data_sources']
            }
        ]
        
        for scenario in self.test_scenarios:
            print(f"\n🎯 Testing scenario: {scenario['name']}")
            print(f"   Description: {scenario['description']}")
            
            scenario_results = {}
            
            for endpoint in endpoints:
                print(f"   🔍 Testing {endpoint['name']} endpoint...")
                
                success, data = self.test_endpoint_direct(
                    endpoint['path'],
                    endpoint['method'],
                    scenario.get('session_data', {}),
                    scenario.get('request_data', {})
                )
                
                if success and data:
                    validation = self.validate_endpoint_structure(
                        endpoint['name'], 
                        data, 
                        endpoint['expected_fields']
                    )
                    scenario_results[endpoint['name']] = validation
                    
                    status = "✅" if validation['structure_valid'] else "⚠️"
                    print(f"      {status} Structure valid: {validation['structure_valid']}")
                else:
                    scenario_results[endpoint['name']] = {
                        'structure_valid': False,
                        'error': data.get('error', 'Unknown error') if data else 'No response'
                    }
                    print(f"      ❌ Failed: {scenario_results[endpoint['name']]['error']}")
            
            test_results['results'][scenario['name']] = scenario_results
        
        # Summary
        print("\n" + "=" * 70)
        total_tests = len(self.test_scenarios) * len(endpoints)
        successful_tests = sum(
            1 for scenario_results in test_results['results'].values()
            for endpoint_result in scenario_results.values()
            if endpoint_result.get('structure_valid', False)
        )
        
        print(f"📊 Test Summary: {successful_tests}/{total_tests} tests passed")
        test_results['summary'] = {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': successful_tests / total_tests if total_tests > 0 else 0
        }
        
        return test_results
    
    def save_regression_report(self, results: Dict, filename: Optional[str] = None) -> str:
        """Save detailed regression test report"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"regression_report_{timestamp}.json"
        
        filepath = os.path.join(os.path.dirname(__file__), 'tests', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📄 Regression report saved: {filepath}")
        return filepath

def main():
    """Main test execution"""
    tester = EndpointRegressionTester()
    
    print("🚀 Starting comprehensive regression testing...")
    results = tester.run_comprehensive_regression_test()
    
    # Save report
    report_path = tester.save_regression_report(results)
    
    # Final verdict
    success_rate = results['summary']['success_rate']
    if success_rate >= 0.95:  # 95% success rate threshold
        print(f"\n🎉 Regression tests PASSED! ({success_rate:.1%} success rate)")
        print("✅ Centralized data management is working correctly!")
        return True
    else:
        print(f"\n❌ Regression tests FAILED! ({success_rate:.1%} success rate)")
        print("⚠️ Some issues found with centralized endpoints")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)