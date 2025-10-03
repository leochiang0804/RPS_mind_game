#!/usr/bin/env python3
"""
Simplified AI Confidence & Strategy Testing Script
==================================================

This script directly simulates the webapp's robot_strategy function
to test AI confidence patterns and robot move strategies across different
difficulties and personalities with 25 moves per test configuration.
"""

import sys
import os
import random
import json
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the same modules used by the webapp
try:
    from optimized_strategies import ToWinStrategy, NotToLoseStrategy
    from personality_engine import get_personality_engine
    from game_context import GameContextBuilder
    from move_mapping import get_counter_move, MOVES
    from strategy import FrequencyStrategy
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running this from the Paper_Scissor_Stone directory")
    sys.exit(1)

class SimplifiedAITester:
    """
    Simplified tester that matches the exact webapp robot_strategy logic
    """
    
    def __init__(self):
        self.moves = MOVES
        
        # Initialize strategies exactly like webapp
        self.to_win_strategy = ToWinStrategy()
        self.not_to_lose_strategy = NotToLoseStrategy()
        self.frequency_strategy = FrequencyStrategy()
        
        # Initialize personality engine
        self.personality_engine = get_personality_engine()
        self.game_context_builder = GameContextBuilder()
        
        # Get available personalities from the engine
        try:
            self.available_personalities = self.personality_engine.get_all_personalities()
            print(f"✅ Available personalities: {self.available_personalities}")
        except Exception as e:
            print(f"⚠️ Could not get personality list: {e}")
            self.available_personalities = ['professor']  # Fallback to working personality
        
        # Define test configurations based on actual webapp options
        self.difficulties = ['random', 'frequency', 'markov', 'lstm']  # Actual difficulties from webapp
        self.strategies = ['to_win', 'not_to_lose']  # Strategy preferences from webapp
        self.test_results = {}
    
    def simulate_webapp_robot_strategy(self, human_history: List[str], game_history: List[Dict], 
                                     difficulty: str, strategy_preference: str, personality: str) -> Tuple[str, float]:
        """
        Simulate the exact robot_strategy function from webapp/app.py
        """
        base_move = None
        confidence = 0.33  # Default confidence
        
        try:
            if difficulty == 'random':
                base_move = random.choice(self.moves)
                confidence = 0.0  # Random has no confidence
                
            elif difficulty == 'frequency':
                if len(human_history) >= 3:
                    # Use frequency strategy
                    try:
                        human_pred = self.frequency_strategy.predict(human_history)
                        if human_pred:
                            base_move = get_counter_move(human_pred)
                            confidence = 0.0  # Frequency analysis has no strategic confidence
                        else:
                            base_move = random.choice(self.moves)
                            confidence = 0.0
                    except Exception:
                        base_move = random.choice(self.moves)
                        confidence = 0.0
                else:
                    base_move = random.choice(self.moves)
                    confidence = 0.0
                    
            elif difficulty == 'markov':
                # Use strategy preference for markov
                if strategy_preference == 'to_win':
                    human_pred = self.to_win_strategy.predict(human_history)
                    confidence = self.to_win_strategy.get_confidence()
                    base_move = get_counter_move(human_pred)
                else:  # not_to_lose
                    human_pred = self.not_to_lose_strategy.predict(human_history)
                    confidence = self.not_to_lose_strategy.get_confidence()
                    base_move = get_counter_move(human_pred)
                
            elif difficulty == 'lstm':
                # For now, fall back to not_to_lose strategy (LSTM integration would be complex for testing)
                human_pred = self.not_to_lose_strategy.predict(human_history)
                confidence = self.not_to_lose_strategy.get_confidence()
                base_move = get_counter_move(human_pred)
                
        except Exception as e:
            print(f"Error in strategy calculation for {difficulty}: {e}")
            base_move = random.choice(self.moves)
            confidence = 0.33
        
        # Apply personality to the base move
        try:
            # Set the personality first
            self.personality_engine.set_personality(personality)
            # Convert game_history to the expected format (list of tuples)
            history_tuples = [(round_data['human_move'], round_data['robot_move']) 
                             for round_data in game_history if 'human_move' in round_data and 'robot_move' in round_data]
            final_move, modified_confidence = self.personality_engine.apply_personality_to_move(
                base_move or random.choice(self.moves), 
                confidence, 
                human_history, 
                history_tuples
            )
        except Exception as e:
            print(f"Error applying personality {personality}: {e}")
            final_move = base_move or random.choice(self.moves)
            modified_confidence = confidence
        
        return final_move, confidence
    
    def determine_winner(self, human_move: str, robot_move: str) -> str:
        """Determine the winner of a round"""
        if human_move == robot_move:
            return 'tie'
        elif (human_move == 'rock' and robot_move == 'scissors') or \
             (human_move == 'paper' and robot_move == 'rock') or \
             (human_move == 'scissors' and robot_move == 'paper'):
            return 'human'
        else:
            return 'robot'
    
    def simulate_game_session(self, difficulty: str, strategy_preference: str, personality: str, num_moves: int = 25) -> Dict[str, Any]:
        """
        Simulate a complete game session with specified difficulty, strategy preference, and personality
        """
        print(f"  🎮 Testing {difficulty.upper()}/{strategy_preference.upper()} with {personality.upper()} personality...")
        
        # Initialize game state
        human_history = []
        robot_history = []
        confidence_history = []
        results_history = []
        game_history = []
        round_details = []
        
        # Simulate varied human moves to test AI adaptation
        human_moves_pattern = []
        for i in range(num_moves):
            if i < 5:
                # Start with rock 
                human_moves_pattern.append('rock')
            elif i < 10:
                # Switch to paper
                human_moves_pattern.append('paper')
            elif i < 15:
                # Switch to scissors
                human_moves_pattern.append('scissors')
            elif i < 20:
                # Alternate pattern
                human_moves_pattern.append(['rock', 'paper'][i % 2])
            else:
                # Random for last 5 moves
                human_moves_pattern.append(random.choice(self.moves))
        
        # Simulate each round
        for round_num in range(num_moves):
            human_move = human_moves_pattern[round_num]
            
            # Get robot strategy and confidence
            robot_move, confidence = self.simulate_webapp_robot_strategy(
                human_history, game_history, difficulty, strategy_preference, personality
            )
            
            # Determine result
            result = self.determine_winner(human_move, robot_move)
            
            # Update histories
            human_history.append(human_move)
            robot_history.append(robot_move)
            confidence_history.append(confidence)
            results_history.append(result)
            
            # Create game record
            game_record = {
                'round': round_num + 1,
                'human_move': human_move,
                'robot_move': robot_move,
                'result': result,
                'confidence': confidence,
                'confidence_percent': int(confidence * 100)
            }
            game_history.append(game_record)
            round_details.append(game_record)
        
        # Calculate final metrics using the same function as webapp
        final_metrics = self.game_context_builder.calculate_metrics(
            human_history, robot_history, results_history, confidence_history
        )
        
        return {
            'difficulty': difficulty,
            'strategy_preference': strategy_preference,
            'personality': personality,
            'round_details': round_details,
            'final_metrics': final_metrics,
            'confidence_stats': {
                'min': min(confidence_history) if confidence_history else 0,
                'max': max(confidence_history) if confidence_history else 0,
                'avg': sum(confidence_history) / len(confidence_history) if confidence_history else 0,
                'final': confidence_history[-1] if confidence_history else 0
            },
            'move_distribution': {
                'human': dict(Counter(human_history)),
                'robot': dict(Counter(robot_history))
            },
            'results_summary': dict(Counter(results_history))
        }
    
    def run_focused_test(self, moves_per_test: int = 25):
        """
        Run focused tests on working combinations
        """
        print("🚀 Starting Focused AI Confidence & Strategy Analysis")
        print(f"📊 Testing {len(self.difficulties)} difficulties × {len(self.strategies)} strategies × {len(self.available_personalities)} personalities")
        print(f"🎯 {moves_per_test} moves per configuration\n")
        
        all_results = {}
        
        # Test each difficulty
        for difficulty in self.difficulties:
            print(f"\n" + "="*60)
            print(f"Testing DIFFICULTY: {difficulty.upper()}")
            print("="*60)
            
            difficulty_results = {}
            
            # Test each strategy preference
            for strategy_preference in self.strategies:
                strategy_results = {}
                
                print(f"\n  Strategy: {strategy_preference.upper()}")
                print("  " + "-"*50)
                
                # Test each available personality with this difficulty and strategy
                for personality in self.available_personalities:
                    try:
                        result = self.simulate_game_session(difficulty, strategy_preference, personality, moves_per_test)
                        strategy_results[personality] = result
                        
                        # Print summary
                        conf_stats = result['confidence_stats']
                        results_summary = result['results_summary']
                        
                        robot_wins = results_summary.get('robot', 0)
                        total_games = sum(results_summary.values())
                        win_rate = (robot_wins / total_games) * 100 if total_games > 0 else 0
                        
                        print(f"    {personality:12} | Confidence: {conf_stats['avg']:.3f} avg "
                              f"({conf_stats['min']:.3f}-{conf_stats['max']:.3f}) | "
                              f"Robot wins: {win_rate:.1f}% | Results: {results_summary}")
                        
                    except Exception as e:
                        print(f"    ❌ Error testing {personality}: {e}")
                        strategy_results[personality] = {'error': str(e)}
                
                difficulty_results[strategy_preference] = strategy_results
            
            all_results[difficulty] = difficulty_results
        
        self.test_results = all_results
        return all_results
    
    def analyze_confidence_patterns(self):
        """
        Analyze confidence patterns and provide detailed insights
        """
        if not self.test_results:
            print("No test results to analyze!")
            return
        
        print("\\n" + "="*80)
        print("📈 DETAILED CONFIDENCE ANALYSIS")
        print("="*80)
        
        # Confidence Analysis by Difficulty
        print("\\n🎯 CONFIDENCE PATTERNS BY DIFFICULTY:")
        print("-" * 60)
        
        difficulty_analysis = {}
        for difficulty, personalities in self.test_results.items():
            confidence_values = []
            for personality, result in personalities.items():
                if 'error' not in result and 'confidence_stats' in result:
                    confidence_values.append(result['confidence_stats']['avg'])
            
            if confidence_values:
                avg_confidence = sum(confidence_values) / len(confidence_values)
                min_confidence = min(confidence_values)
                max_confidence = max(confidence_values)
                difficulty_analysis[difficulty] = {
                    'avg': avg_confidence,
                    'min': min_confidence, 
                    'max': max_confidence,
                    'samples': len(confidence_values)
                }
                
                print(f"{difficulty:12} | Avg: {avg_confidence:.3f} | Range: {min_confidence:.3f}-{max_confidence:.3f} | Samples: {len(confidence_values)}")
        
        # Win Rate Analysis
        print("\\n🏆 ROBOT WIN RATES BY DIFFICULTY:")
        print("-" * 60)
        
        for difficulty, personalities in self.test_results.items():
            win_rates = []
            for personality, result in personalities.items():
                if 'error' not in result and 'results_summary' in result:
                    total_games = sum(result['results_summary'].values())
                    robot_wins = result['results_summary'].get('robot', 0)
                    win_rate = (robot_wins / total_games) * 100 if total_games > 0 else 0
                    win_rates.append(win_rate)
            
            if win_rates:
                avg_win_rate = sum(win_rates) / len(win_rates)
                min_win_rate = min(win_rates)
                max_win_rate = max(win_rates)
                
                print(f"{difficulty:12} | Avg: {avg_win_rate:.1f}% | Range: {min_win_rate:.1f}%-{max_win_rate:.1f}%")
        
        # Detailed Round-by-Round Analysis for one example
        print("\\n🔍 ROUND-BY-ROUND CONFIDENCE EXAMPLE (Hard + Professor):")
        print("-" * 60)
        
        try:
            example_result = self.test_results['hard']['professor']
            if 'round_details' in example_result:
                for i, round_data in enumerate(example_result['round_details'][:10]):  # Show first 10 rounds
                    print(f"Round {round_data['round']:2d}: Human={round_data['human_move']:8} | "
                          f"Robot={round_data['robot_move']:8} | "
                          f"Result={round_data['result']:6} | "
                          f"Confidence={round_data['confidence_percent']:3d}%")
                if len(example_result['round_details']) > 10:
                    print(f"... and {len(example_result['round_details']) - 10} more rounds")
        except Exception as e:
            print(f"Could not show example: {e}")
    
    def save_detailed_results(self, filename: str = "ai_confidence_analysis.json"):
        """
        Save detailed test results to JSON file
        """
        try:
            with open(filename, 'w') as f:
                json.dump(self.test_results, f, indent=2, default=str)
            print(f"\\n💾 Detailed results saved to: {filename}")
        except Exception as e:
            print(f"Error saving results: {e}")


def main():
    """Main function to run the focused AI testing"""
    print("🧠 Simplified AI Confidence & Strategy Testing")
    print("==============================================")
    
    # Initialize tester
    tester = SimplifiedAITester()
    
    # Run focused tests
    results = tester.run_focused_test(moves_per_test=25)
    
    # Generate detailed analysis
    tester.analyze_confidence_patterns()
    
    # Save results
    tester.save_detailed_results()
    
    print(f"\n✅ Testing complete! Analyzed {len(tester.difficulties)} difficulties × {len(tester.strategies)} strategies × {len(tester.available_personalities)} personalities")
    print("Check the generated JSON file for complete round-by-round data.")


if __name__ == "__main__":
    main()