"""
Advanced LSTM Integration for AI Test Platform
Trains and tests the new advanced LSTM system across different game lengths
"""

import os
import sys
import json
import time
from typing import Dict, List, Any
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_lstm_trainer import AdvancedLSTMTrainer, GAME_CONFIGS
from ai_difficulty_test_platform import AITestPlatform
from move_mapping import get_counter_move, MOVES
import random

class AdvancedLSTMTester:
    """Test the advanced LSTM against the AI test platform"""
    
    def __init__(self):
        self.trainers = {}
        self.test_platform = AITestPlatform()
        
    def train_all_models(self):
        """Train advanced LSTM models for all game lengths"""
        print("🏗️ Training Advanced LSTM models for all game lengths...")
        
        for game_length in [25, 50, 100, 250, 500, 1000]:
            print(f"\n📏 Training model for {game_length}-move games...")
            
            trainer = AdvancedLSTMTrainer(game_length)
            
            # Pretrain on synthetic data
            trainer.pretrain(num_epochs=3, batch_size=32)
            
            # Save the model
            model_path = f"models/lstm/advanced_lstm_{game_length}.pth"
            trainer.save_model(model_path)
            
            self.trainers[game_length] = trainer
        
        print("✅ All models trained and saved!")
    
    def test_advanced_lstm(self, game_length: int, test_id: int = 1) -> Dict[str, Any]:
        """Test advanced LSTM model in a single game"""
        if game_length not in self.trainers:
            # Load the model if not in memory
            trainer = AdvancedLSTMTrainer(game_length)
            model_path = f"models/lstm/advanced_lstm_{game_length}.pth"
            if trainer.load_model(model_path):
                self.trainers[game_length] = trainer
            else:
                # Train if model doesn't exist
                print(f"⚠️ Model for {game_length} not found, training...")
                trainer.pretrain(num_epochs=2)
                trainer.save_model(model_path)
                self.trainers[game_length] = trainer
        
        trainer = self.trainers[game_length]
        trainer.reset_episode()  # Reset for new game
        
        # Initialize game state
        human_moves = []
        robot_moves = []
        outcomes = []
        lstm_predictions = []
        
        wins = 0
        draws = 0
        losses = 0
        
        print(f"  🧠 Running Advanced LSTM test (length: {game_length}, test: {test_id})")
        
        for round_num in range(game_length):
            # Simulate human player with realistic patterns
            human_move = self.test_platform.simulate_human_player(
                round_num, 
                game_length,
                lstm_predictions[-5:] if lstm_predictions else None,
                human_moves[-10:] if human_moves else None
            )
            human_moves.append(human_move)
            
            # Get Advanced LSTM prediction
            if len(human_moves) >= 2:
                try:
                    # Predict next human move
                    probs = trainer.predict(human_moves[:-1], robot_moves, outcomes)
                    
                    # Apply exploration (ε-greedy)
                    config = GAME_CONFIGS[game_length]
                    if random.random() < config.exploration_rate:
                        predicted_human_move = random.choice(MOVES)
                    else:
                        predicted_human_move = max(probs.keys(), key=lambda k: probs[k])
                    
                    lstm_predictions.append(predicted_human_move)
                    
                    # Robot plays counter to predicted move
                    robot_move = get_counter_move(predicted_human_move)
                    
                except Exception as e:
                    print(f"    ⚠️ Advanced LSTM error: {e}")
                    robot_move = random.choice(MOVES)
                    lstm_predictions.append(random.choice(MOVES))
            else:
                robot_move = random.choice(MOVES)
                lstm_predictions.append(random.choice(MOVES))
            
            robot_moves.append(robot_move)
            
            # Determine outcome
            if human_move == robot_move:
                result = 'draw'
                draws += 1
            elif robot_move == get_counter_move(human_move):
                result = 'win'
                wins += 1
            else:
                result = 'lose'
                losses += 1
            
            outcomes.append(result)
            
            # Online learning update
            trainer.update(human_moves, robot_moves, outcomes)
        
        # Calculate metrics
        total_games = wins + draws + losses
        robot_win_rate = (wins / total_games * 100) if total_games > 0 else 0
        
        # Calculate prediction accuracy
        correct_predictions = 0
        for i, pred in enumerate(lstm_predictions):
            if i < len(human_moves) and pred == human_moves[i]:
                correct_predictions += 1
        
        prediction_accuracy = (correct_predictions / len(lstm_predictions) * 100) if lstm_predictions else 0
        
        return {
            'model': 'Advanced_LSTM',
            'game_length': game_length,
            'test_id': test_id,
            'robot_win_rate': robot_win_rate,
            'robot_wins': wins,
            'robot_draws': draws,
            'robot_losses': losses,
            'prediction_accuracy': prediction_accuracy,
            'total_predictions': len(lstm_predictions),
            'correct_predictions': correct_predictions
        }
    
    def run_comprehensive_tests(self, tests_per_length: int = 5) -> Dict[str, Any]:
        """Run comprehensive tests across all game lengths"""
        print("🚀 Starting Advanced LSTM comprehensive testing...")
        print(f"📊 Testing lengths: {[25, 50, 100, 250, 500, 1000]}")
        print(f"🔄 Tests per length: {tests_per_length}")
        
        all_results = []
        
        for game_length in [25, 50, 100, 250, 500, 1000]:
            print(f"\n📏 Testing game length: {game_length}")
            
            for test_id in range(1, tests_per_length + 1):
                # Test Advanced LSTM
                lstm_result = self.test_advanced_lstm(game_length, test_id)
                all_results.append(lstm_result)
                
                # Test regular Markov for comparison
                markov_result = self.test_platform.run_markov_test(game_length, test_id)
                all_results.append(markov_result)
        
        return self.analyze_results(all_results)
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze test results"""
        print("\n📈 Analyzing results...")
        
        # Group results by model and game length
        advanced_lstm_results = defaultdict(list)
        markov_results = defaultdict(list)
        
        for result in results:
            if 'error' in result:
                continue
                
            if result['model'] == 'Advanced_LSTM':
                advanced_lstm_results[result['game_length']].append(result)
            elif result['model'] == 'Markov':
                markov_results[result['game_length']].append(result)
        
        # Calculate averages
        analysis = {
            'advanced_lstm_performance': {},
            'markov_performance': {},
            'comparison': {},
            'recommendations': []
        }
        
        for game_length in [25, 50, 100, 250, 500, 1000]:
            if game_length in advanced_lstm_results:
                lstm_data = advanced_lstm_results[game_length]
                lstm_avg = {
                    'avg_robot_win_rate': sum(r['robot_win_rate'] for r in lstm_data) / len(lstm_data),
                    'avg_prediction_accuracy': sum(r['prediction_accuracy'] for r in lstm_data) / len(lstm_data),
                    'std_robot_win_rate': self._calculate_std([r['robot_win_rate'] for r in lstm_data]),
                    'best_robot_win_rate': max(r['robot_win_rate'] for r in lstm_data),
                    'worst_robot_win_rate': min(r['robot_win_rate'] for r in lstm_data)
                }
                analysis['advanced_lstm_performance'][game_length] = lstm_avg
            
            if game_length in markov_results:
                markov_data = markov_results[game_length]
                markov_avg = {
                    'avg_robot_win_rate': sum(r['robot_win_rate'] for r in markov_data) / len(markov_data),
                    'avg_prediction_accuracy': sum(r['prediction_accuracy'] for r in markov_data) / len(markov_data),
                    'std_robot_win_rate': self._calculate_std([r['robot_win_rate'] for r in markov_data]),
                    'best_robot_win_rate': max(r['robot_win_rate'] for r in markov_data),
                    'worst_robot_win_rate': min(r['robot_win_rate'] for r in markov_data)
                }
                analysis['markov_performance'][game_length] = markov_avg
            
            # Compare Advanced LSTM vs Markov
            if game_length in advanced_lstm_results and game_length in markov_results:
                lstm_avg_win = analysis['advanced_lstm_performance'][game_length]['avg_robot_win_rate']
                markov_avg_win = analysis['markov_performance'][game_length]['avg_robot_win_rate']
                lstm_avg_acc = analysis['advanced_lstm_performance'][game_length]['avg_prediction_accuracy']
                markov_avg_acc = analysis['markov_performance'][game_length]['avg_prediction_accuracy']
                
                analysis['comparison'][game_length] = {
                    'lstm_advantage_win_rate': lstm_avg_win - markov_avg_win,
                    'lstm_advantage_accuracy': lstm_avg_acc - markov_avg_acc,
                    'better_model': 'Advanced_LSTM' if lstm_avg_win > markov_avg_win else 'Markov'
                }
        
        return analysis
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) <= 1:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def print_detailed_report(self, analysis: Dict[str, Any]):
        """Print detailed analysis report"""
        print("\n" + "="*80)
        print("🎯 ADVANCED LSTM vs MARKOV - DETAILED REPORT")
        print("="*80)
        
        print(f"\n🧠 ADVANCED LSTM PERFORMANCE:")
        if analysis['advanced_lstm_performance']:
            for game_length, metrics in analysis['advanced_lstm_performance'].items():
                print(f"  {game_length:4d} games: Win Rate: {metrics['avg_robot_win_rate']:5.1f}% ± {metrics['std_robot_win_rate']:4.1f}% | "
                      f"Accuracy: {metrics['avg_prediction_accuracy']:5.1f}% | Best: {metrics['best_robot_win_rate']:5.1f}%")
        else:
            print("  ❌ Advanced LSTM not available or no results")
        
        print(f"\n🔗 MARKOV PERFORMANCE:")
        for game_length, metrics in analysis['markov_performance'].items():
            print(f"  {game_length:4d} games: Win Rate: {metrics['avg_robot_win_rate']:5.1f}% ± {metrics['std_robot_win_rate']:4.1f}% | "
                  f"Accuracy: {metrics['avg_prediction_accuracy']:5.1f}% | Best: {metrics['best_robot_win_rate']:5.1f}%")
        
        print(f"\n⚔️  ADVANCED LSTM vs MARKOV COMPARISON:")
        for game_length, comparison in analysis['comparison'].items():
            advantage = comparison['lstm_advantage_win_rate']
            better = comparison['better_model']
            print(f"  {game_length:4d} games: {better} wins by {abs(advantage):5.1f}% | "
                  f"LSTM accuracy advantage: {comparison['lstm_advantage_accuracy']:+5.1f}%")
        
        print("\n" + "="*80)

def main():
    """Main function"""
    print("🚀 Advanced LSTM Training and Testing")
    print("Based on lstm_rps_setup.md recommendations")
    
    tester = AdvancedLSTMTester()
    
    # Check if we should train new models
    user_choice = input("Train new Advanced LSTM models? (y/n): ").lower().strip()
    if user_choice == 'y':
        tester.train_all_models()
    
    # Run comprehensive tests
    print("\n🧪 Running comprehensive tests...")
    analysis = tester.run_comprehensive_tests(tests_per_length=3)
    
    # Print results
    tester.print_detailed_report(analysis)
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"advanced_lstm_test_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\n💾 Results saved to: {filename}")

if __name__ == "__main__":
    main()