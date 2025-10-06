#!/usr/bin/env python3
"""
Batch Game Simulation Runner
============================

This script runs comprehensive simulations across all difficulty levels and game lengths
to find optimal human strategies for beating the AI system.
"""

import os
import json
import time
import subprocess
from datetime import datetime
from typing import Dict, List, Any

# Simulation configurations to test
SIMULATION_CONFIGS = [
    # 50-move games
    {'difficulty': 'easy', 'game_length': 50, 'strategy': 'to_win', 'personality': 'neutral'},
    {'difficulty': 'medium', 'game_length': 50, 'strategy': 'to_win', 'personality': 'neutral'},
    {'difficulty': 'hard', 'game_length': 50, 'strategy': 'to_win', 'personality': 'neutral'},
    
    # 75-move games
    {'difficulty': 'easy', 'game_length': 75, 'strategy': 'to_win', 'personality': 'neutral'},
    {'difficulty': 'medium', 'game_length': 75, 'strategy': 'to_win', 'personality': 'neutral'},
    {'difficulty': 'hard', 'game_length': 75, 'strategy': 'to_win', 'personality': 'neutral'},
]


class BatchSimulationRunner:
    """Runs multiple game simulations and aggregates results"""
    
    def __init__(self, output_dir: str = 'simulation_results'):
        self.output_dir = output_dir
        self.results = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def run_all_simulations(self):
        """Run all simulation configurations"""
        print("Starting batch simulation run...")
        print(f"Total configurations: {len(SIMULATION_CONFIGS)}")
        print()
        
        start_time = time.time()
        
        for i, config in enumerate(SIMULATION_CONFIGS):
            print(f"=== Configuration {i+1}/{len(SIMULATION_CONFIGS)} ===")
            print(f"Difficulty: {config['difficulty']}")
            print(f"Game Length: {config['game_length']}")
            print(f"Strategy: {config['strategy']}")
            print(f"Personality: {config['personality']}")
            
            # Run simulation
            result = self.run_single_simulation(config)
            self.results.append({
                'config': config,
                'result': result,
                'timestamp': time.time()
            })
            
            print(f"Result: {result.get('best_win_rate', 0):.1%} win rate")
            print(f"Robust: {'Yes' if result.get('is_robust', False) else 'No'}")
            print()
        
        # Generate comprehensive report
        total_time = time.time() - start_time
        self.generate_report(total_time)
    
    def run_single_simulation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single simulation configuration"""
        
        # Create output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sim_{config['difficulty']}_{config['game_length']}_{timestamp}.json"
        output_path = os.path.join(self.output_dir, filename)
        
        # Build command
        cmd = [
            'python', 'game_simulator.py',
            '--difficulty', config['difficulty'],
            '--game_length', str(config['game_length']),
            '--strategy', config['strategy'],
            '--personality', config['personality'],
            '--num_sequences', '100',  # Test 100 sequences per configuration
            '--output', output_path,
            '--seed', '42'  # Fixed seed for reproducible results
        ]
        
        try:
            # Run simulation
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Load results from file
                with open(output_path, 'r') as f:
                    return json.load(f)
            else:
                print(f"Error running simulation: {result.stderr}")
                return {'error': result.stderr, 'best_win_rate': 0.0, 'is_robust': False}
                
        except subprocess.TimeoutExpired:
            print("Simulation timed out")
            return {'error': 'timeout', 'best_win_rate': 0.0, 'is_robust': False}
        except Exception as e:
            print(f"Unexpected error: {e}")
            return {'error': str(e), 'best_win_rate': 0.0, 'is_robust': False}
    
    def generate_report(self, total_time: float):
        """Generate a comprehensive analysis report"""
        
        print("=" * 60)
        print("COMPREHENSIVE SIMULATION REPORT")
        print("=" * 60)
        print()
        
        # Overall statistics
        total_configs = len(self.results)
        successful_runs = sum(1 for r in self.results if 'error' not in r['result'])
        robust_strategies = sum(1 for r in self.results if r['result'].get('is_robust', False))
        
        print(f"Total Configurations Tested: {total_configs}")
        print(f"Successful Runs: {successful_runs}")
        print(f"Configurations with Robust Strategies (>45% win rate): {robust_strategies}")
        print(f"Total Execution Time: {total_time:.1f} seconds")
        print()
        
        # Results by difficulty
        print("RESULTS BY DIFFICULTY LEVEL:")
        print("-" * 40)
        
        for difficulty in ['easy', 'medium', 'hard']:
            difficulty_results = [r for r in self.results if r['config']['difficulty'] == difficulty]
            
            if difficulty_results:
                win_rates = [r['result'].get('best_win_rate', 0) for r in difficulty_results]
                avg_win_rate = sum(win_rates) / len(win_rates) if win_rates else 0
                max_win_rate = max(win_rates) if win_rates else 0
                robust_count = sum(1 for r in difficulty_results if r['result'].get('is_robust', False))
                
                print(f"{difficulty.upper()}:")
                print(f"  Average Best Win Rate: {avg_win_rate:.1%}")
                print(f"  Maximum Win Rate: {max_win_rate:.1%}")
                print(f"  Robust Strategies Found: {robust_count}/{len(difficulty_results)}")
                print()
        
        # Results by game length
        print("RESULTS BY GAME LENGTH:")
        print("-" * 40)
        
        for game_length in [50, 75]:
            length_results = [r for r in self.results if r['config']['game_length'] == game_length]
            
            if length_results:
                win_rates = [r['result'].get('best_win_rate', 0) for r in length_results]
                avg_win_rate = sum(win_rates) / len(win_rates) if win_rates else 0
                max_win_rate = max(win_rates) if win_rates else 0
                robust_count = sum(1 for r in length_results if r['result'].get('is_robust', False))
                
                print(f"{game_length} MOVES:")
                print(f"  Average Best Win Rate: {avg_win_rate:.1%}")
                print(f"  Maximum Win Rate: {max_win_rate:.1%}")
                print(f"  Robust Strategies Found: {robust_count}/{len(length_results)}")
                print()
        
        # Best performing configurations
        print("TOP PERFORMING CONFIGURATIONS:")
        print("-" * 40)
        
        # Sort by win rate
        sorted_results = sorted(self.results, key=lambda r: r['result'].get('best_win_rate', 0), reverse=True)
        
        for i, result in enumerate(sorted_results[:5]):
            config = result['config']
            win_rate = result['result'].get('best_win_rate', 0)
            is_robust = result['result'].get('is_robust', False)
            
            print(f"{i+1}. {config['difficulty'].title()} / {config['game_length']} moves")
            print(f"   Win Rate: {win_rate:.1%}")
            print(f"   Robust: {'Yes' if is_robust else 'No'}")
            print()
        
        # Analysis and recommendations
        print("ANALYSIS & RECOMMENDATIONS:")
        print("-" * 40)
        
        # Find if any configuration allows humans to win consistently
        best_overall = max(self.results, key=lambda r: r['result'].get('best_win_rate', 0))
        best_win_rate = best_overall['result'].get('best_win_rate', 0)
        
        if best_win_rate > 0.55:
            print("✅ HUMANS CAN WIN ROBUSTLY!")
            print(f"   Best configuration: {best_overall['config']['difficulty']} difficulty, {best_overall['config']['game_length']} moves")
            print(f"   Best win rate achieved: {best_win_rate:.1%}")
        elif best_win_rate > 0.45:
            print("⚖️  HUMANS CAN ACHIEVE COMPETITIVE PERFORMANCE")
            print(f"   Best configuration: {best_overall['config']['difficulty']} difficulty, {best_overall['config']['game_length']} moves")
            print(f"   Best win rate achieved: {best_win_rate:.1%}")
        else:
            print("❌ AI SYSTEM IS DOMINANT")
            print(f"   Best human win rate achieved: {best_win_rate:.1%}")
            print("   No robust human strategies found across all configurations")
        
        print()
        
        # Difficulty progression analysis
        easy_best = max([r['result'].get('best_win_rate', 0) for r in self.results if r['config']['difficulty'] == 'easy'], default=0)
        medium_best = max([r['result'].get('best_win_rate', 0) for r in self.results if r['config']['difficulty'] == 'medium'], default=0)
        hard_best = max([r['result'].get('best_win_rate', 0) for r in self.results if r['config']['difficulty'] == 'hard'], default=0)
        
        print("DIFFICULTY SCALING:")
        print(f"Easy:   {easy_best:.1%}")
        print(f"Medium: {medium_best:.1%}")
        print(f"Hard:   {hard_best:.1%}")
        
        if easy_best > medium_best > hard_best:
            print("✅ AI difficulty scaling is working correctly")
        else:
            print("⚠️  Unexpected difficulty scaling detected")
        
        print()
        
        # Save comprehensive results
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'total_execution_time': total_time,
            'summary': {
                'total_configs': total_configs,
                'successful_runs': successful_runs,
                'robust_strategies_found': robust_strategies,
                'best_overall_win_rate': best_win_rate,
                'best_config': best_overall['config']
            },
            'all_results': self.results
        }
        
        report_path = os.path.join(self.output_dir, 'comprehensive_report.json')
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"Detailed report saved to: {report_path}")


def main():
    """Main function to run batch simulations"""
    runner = BatchSimulationRunner()
    runner.run_all_simulations()


if __name__ == '__main__':
    main()