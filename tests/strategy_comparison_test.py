#!/usr/bin/env python3
"""
Strategy Comparison Analysis - Round-by-Round Move & Confidence Tracking
=======================================================================

This script analyzes how AI moves and confidence scores change round-by-round
when comparing "to_win" vs "not_to_lose" strategies across different difficulties
and personalities over 25 moves to better understand LSTM behavior patterns.
"""

import sys
import os
import random
import json
from typing import List, Dict, Any, Tuple
from collections import Counter

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from optimized_strategies import ToWinStrategy, NotToLoseStrategy
    from personality_engine import get_personality_engine
    from game_context import GameContextBuilder
    from move_mapping import get_counter_move, MOVES
    from strategy import FrequencyStrategy
    # Import LSTM components directly
    from lstm_web_integration import LSTMPredictor
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running this from the Paper_Scissor_Stone directory")
    sys.exit(1)

class StrategyComparisonTester:
    """
    Focused tester to compare how strategies affect moves and confidence round-by-round
    """
    
    def __init__(self):
        self.moves = MOVES
        
        # Initialize strategies exactly like webapp
        self.to_win_strategy = ToWinStrategy()
        self.not_to_lose_strategy = NotToLoseStrategy()
        self.frequency_strategy = FrequencyStrategy()
        
        # Initialize LSTM predictor
        try:
            self.lstm_predictor = LSTMPredictor()
            self.lstm_available = True
            print("✅ LSTM predictor initialized for testing")
        except Exception as e:
            print(f"⚠️ LSTM predictor not available: {e}")
            self.lstm_predictor = None
            self.lstm_available = False
        
        # Initialize personality engine
        self.personality_engine = get_personality_engine()
        self.game_context_builder = GameContextBuilder()
        
        # Get available personalities
        self.available_personalities = self.personality_engine.get_all_personalities()
        
        # Define test configurations
        self.difficulties = ['markov', 'lstm']  # Focus on strategic difficulties only
        self.strategies = ['to_win', 'not_to_lose']
        
        # Generate a 25-move human sequence for extended analysis
        # Pattern: 10 moves rock, 10 moves paper, 10 moves scissors, repeat pattern, then random
        self.human_sequence = []
        base_pattern = ['rock'] * 10 + ['paper'] * 10 + ['scissors'] * 10
        
        # Repeat the 30-move pattern 3 times (90 moves)
        for i in range(3):
            self.human_sequence.extend(base_pattern)
        
        # Add 10 random moves at the end
        random.seed(42)  # For reproducible results
        for i in range(10):
            self.human_sequence.append(random.choice(['rock', 'paper', 'scissors']))
        
        print(f"✅ Generated {len(self.human_sequence)}-move human sequence")
        print(f"✅ Pattern: 3x(10 rock + 10 paper + 10 scissors) + 10 random")
        print(f"✅ Testing with difficulties: {self.difficulties}")
        print(f"✅ Available personalities: {self.available_personalities}")
    
    def get_strategy_move_and_confidence(self, human_history: List[str], 
                                       difficulty: str, strategy_preference: str) -> Tuple[str, float]:
        """
        Get the base move and confidence from strategy before personality is applied
        """
        base_move = None
        confidence = 0.33
        
        try:
            if difficulty == 'markov':
                if strategy_preference == 'to_win':
                    human_pred = self.to_win_strategy.predict(human_history)
                    confidence = self.to_win_strategy.get_confidence()
                    base_move = get_counter_move(human_pred)
                else:  # not_to_lose
                    human_pred = self.not_to_lose_strategy.predict(human_history)
                    confidence = self.not_to_lose_strategy.get_confidence()
                    base_move = get_counter_move(human_pred)
                    
            elif difficulty == 'lstm':
                # Use LSTM predictor with strategy-specific confidence calculation
                if self.lstm_available and self.lstm_predictor and len(human_history) >= 3:
                    try:
                        # Get LSTM probabilities for human moves
                        lstm_probs = self.lstm_predictor.predict(human_history)
                        
                        # Calculate strategy-specific confidence
                        if strategy_preference == 'to_win':
                            # ToWin strategy: confidence = abs(2 * highest_prob - 1)
                            highest_prob = max(lstm_probs.values())
                            confidence = abs(2 * highest_prob - 1)
                        elif strategy_preference == 'not_to_lose':
                            # NotToLose strategy: confidence = abs(2 * (sum of highest two probs) - 1)
                            sorted_probs = sorted(lstm_probs.values(), reverse=True)
                            highest_two_sum = sorted_probs[0] + sorted_probs[1]
                            confidence = abs(2 * highest_two_sum - 1)
                        else:
                            # Default to original LSTM confidence
                            confidence = max(lstm_probs.values())
                        
                        # Get the counter move
                        base_move = self.lstm_predictor.get_counter_move(human_history)
                        
                    except Exception as e:
                        print(f"LSTM prediction error: {e}")
                        # Fallback to not_to_lose strategy
                        human_pred = self.not_to_lose_strategy.predict(human_history)
                        confidence = self.not_to_lose_strategy.get_confidence()
                        base_move = get_counter_move(human_pred)
                else:
                    # Fallback to not_to_lose strategy if LSTM not available
                    human_pred = self.not_to_lose_strategy.predict(human_history)
                    confidence = self.not_to_lose_strategy.get_confidence()
                    base_move = get_counter_move(human_pred)
                
        except Exception as e:
            print(f"Strategy error: {e}")
            base_move = random.choice(self.moves)
            confidence = 0.33
            
        return base_move or random.choice(self.moves), confidence
    
    def apply_personality_to_move(self, base_move: str, confidence: float, 
                                human_history: List[str], game_history: List[Dict],
                                personality: str) -> str:
        """
        Apply personality to get the final move
        """
        try:
            self.personality_engine.set_personality(personality)
            # Convert game_history to expected format
            history_tuples = [(round_data['human_move'], round_data['robot_move']) 
                             for round_data in game_history if 'human_move' in round_data and 'robot_move' in round_data]
            final_move = self.personality_engine.apply_personality_to_move(
                base_move, confidence, human_history, history_tuples
            )
            return final_move
        except Exception as e:
            return base_move  # Fallback to base move if personality fails
    
    def simulate_strategy_comparison(self, difficulty: str, personality: str) -> Dict[str, Any]:
        """
        Simulate both strategies with the same difficulty and personality to compare differences
        """
        print(f"\n🔍 Comparing strategies for {difficulty.upper()} + {personality.upper()}")
        print("=" * 60)
        
        # Reset strategies for each personality test to ensure clean state
        self.to_win_strategy = ToWinStrategy()
        self.not_to_lose_strategy = NotToLoseStrategy()
        
        results = {
            'difficulty': difficulty,
            'personality': personality,
            'human_sequence': self.human_sequence,
            'strategies': {}
        }
        
        for strategy in self.strategies:
            print(f"\n  📊 Testing {strategy.upper()} strategy:")
            print("  " + "-" * 45)
            
            # Initialize game state for this strategy
            human_history = []
            robot_history = []
            confidence_history = []
            results_history = []
            game_history = []
            round_details = []
            
            # Simulate each round
            for round_num in range(25):  # Max 25 moves
                human_move = self.human_sequence[round_num]
                
                # Get strategy-based move and confidence
                base_move, confidence = self.get_strategy_move_and_confidence(
                    human_history, difficulty, strategy
                )
                
                # Apply personality to get final move
                final_move = self.apply_personality_to_move(
                    base_move, confidence, human_history, game_history, personality
                )
                
                # Determine result
                result = self.determine_winner(human_move, final_move)
                
                # Print round details
                print(f"    Round {round_num+1:2d}: Human={human_move:8} | "
                      f"Base={base_move:8} | Final={final_move:8} | "
                      f"Result={result:6} | Confidence={confidence:.3f} ({int(confidence*100):3d}%)")
                
                # Update histories
                human_history.append(human_move)
                robot_history.append(final_move)
                confidence_history.append(confidence)
                results_history.append(result)
                
                # Create game record
                game_record = {
                    'round': round_num + 1,
                    'human_move': human_move,
                    'base_move': base_move,
                    'final_move': final_move,
                    'result': result,
                    'confidence': confidence,
                    'confidence_percent': int(confidence * 100)
                }
                game_history.append(game_record)
                round_details.append(game_record)
            
            # Calculate final metrics
            final_metrics = self.game_context_builder.calculate_metrics(
                human_history, robot_history, results_history, confidence_history
            )
            
            # Store strategy results
            results['strategies'][strategy] = {
                'round_details': round_details,
                'final_metrics': final_metrics,
                'confidence_stats': {
                    'min': min(confidence_history),
                    'max': max(confidence_history),
                    'avg': sum(confidence_history) / len(confidence_history),
                    'progression': confidence_history
                },
                'results_summary': dict(Counter(results_history))
            }
            
            # Print strategy summary
            robot_wins = dict(Counter(results_history)).get('robot', 0)
            win_rate = (robot_wins / 25) * 100
            avg_confidence = sum(confidence_history) / len(confidence_history)
            
            print(f"    → Win Rate: {win_rate:.1f}% | Avg Confidence: {avg_confidence:.3f} | "
                  f"Range: {min(confidence_history):.3f}-{max(confidence_history):.3f}")
        
        return results
    
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
    
    def compare_strategies_across_configurations(self):
        """
        Compare strategies across all difficulty-personality combinations
        """
        print("🚀 STRATEGY COMPARISON ANALYSIS")
        print("🎯 Fixed Human Sequence: " + " → ".join(self.human_sequence))
        print("📊 Comparing TO_WIN vs NOT_TO_LOSE strategies")
        print("=" * 80)
        
        all_comparisons = {}
        
        for difficulty in self.difficulties:
            difficulty_comparisons = {}
            
            for personality in self.available_personalities:
                try:
                    comparison = self.simulate_strategy_comparison(difficulty, personality)
                    difficulty_comparisons[personality] = comparison
                except Exception as e:
                    print(f"❌ Error testing {difficulty}/{personality}: {e}")
            
            all_comparisons[difficulty] = difficulty_comparisons
        
        return all_comparisons
    
    def analyze_strategy_differences(self, all_comparisons: Dict):
        """
        Analyze the key differences between strategies
        """
        print("\n\n📈 STRATEGY DIFFERENCE ANALYSIS")
        print("=" * 60)
        
        print("\n🎯 CONFIDENCE SCORE COMPARISON:")
        print("-" * 50)
        
        for difficulty, personalities in all_comparisons.items():
            print(f"\n{difficulty.upper()}:")
            
            for personality, comparison in personalities.items():
                if 'strategies' in comparison:
                    to_win_conf = comparison['strategies'].get('to_win', {}).get('confidence_stats', {}).get('avg', 0)
                    not_to_lose_conf = comparison['strategies'].get('not_to_lose', {}).get('confidence_stats', {}).get('avg', 0)
                    
                    confidence_diff = not_to_lose_conf - to_win_conf
                    
                    print(f"  {personality:12} | TO_WIN: {to_win_conf:.3f} | NOT_TO_LOSE: {not_to_lose_conf:.3f} | "
                          f"Diff: {confidence_diff:+.3f}")
        
        print("\n🏆 WIN RATE COMPARISON:")
        print("-" * 50)
        
        for difficulty, personalities in all_comparisons.items():
            print(f"\n{difficulty.upper()}:")
            
            for personality, comparison in personalities.items():
                if 'strategies' in comparison:
                    to_win_wins = comparison['strategies'].get('to_win', {}).get('results_summary', {}).get('robot', 0)
                    not_to_lose_wins = comparison['strategies'].get('not_to_lose', {}).get('results_summary', {}).get('robot', 0)
                    
                    to_win_rate = (to_win_wins / 25) * 100
                    not_to_lose_rate = (not_to_lose_wins / 25) * 100
                    win_diff = not_to_lose_rate - to_win_rate
                    
                    print(f"  {personality:12} | TO_WIN: {to_win_rate:.1f}% | NOT_TO_LOSE: {not_to_lose_rate:.1f}% | "
                          f"Diff: {win_diff:+.1f}%")
    
    def save_detailed_comparison(self, all_comparisons: Dict, filename: str = "strategy_comparison_analysis.json"):
        """
        Save detailed comparison results
        """
        try:
            with open(filename, 'w') as f:
                json.dump(all_comparisons, f, indent=2, default=str)
            print(f"\n💾 Detailed comparison saved to: {filename}")
        except Exception as e:
            print(f"Error saving results: {e}")


def main():
    """Main function to run the strategy comparison analysis"""
    print("🧠 AI Strategy Comparison Analysis")
    print("==================================")
    print("🎯 Comparing TO_WIN vs NOT_TO_LOSE strategies")
    print("📊 Extended 25-move sequence for detailed LSTM analysis")
    print("🔍 Focus on understanding LSTM difficulty behavior patterns")
    
    # Initialize tester
    tester = StrategyComparisonTester()
    
    # Run comprehensive comparison
    comparisons = tester.compare_strategies_across_configurations()
    
    # Analyze differences
    tester.analyze_strategy_differences(comparisons)
    
    # Save results
    tester.save_detailed_comparison(comparisons)
    
    print(f"\n✅ Strategy comparison complete!")
    print("📊 Analyzed round-by-round differences in moves and confidence")
    print("💾 Check JSON file for detailed round-by-round data")


if __name__ == "__main__":
    main()