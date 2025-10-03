"""
AI Difficulty Testing Platform
Comprehensive testing and optimization platform for LSTM and Markov AI models.
Tests across different game lengths with To Win strategy and Neutral personality.
"""

import os
import sys
import json
import time
import random
import statistics
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategy import MarkovStrategy
from lstm_unified import LSTMManager
from optimized_strategies import ToWinStrategy, NotToLoseStrategy
from move_mapping import normalize_move, get_counter_move, MOVES
from personality_engine import get_personality_engine
from ml_model_enhanced import EnhancedMLModel

class AITestPlatform:
    """Comprehensive AI testing and optimization platform"""
    
    def __init__(self):
        self.test_lengths = [25, 50, 100, 250, 500, 1000]
        self.moves = ['rock', 'paper', 'scissors']
        self.results = defaultdict(list)
        
        # Initialize AI models
        self.lstm_predictor = None
        self.markov_strategy = MarkovStrategy()
        self.to_win_strategy = ToWinStrategy()
        self.personality_engine = get_personality_engine()
        
        # Test configurations  
        self.strategy_preference = 'to_win'
        self.personality = 'neutral'
        
        # Enhanced human simulation parameters
        self.counter_prediction_probability = 0.3  # 30% chance human tries to counter-predict
        self.strategy_change_probability = 0.2     # 20% chance to change strategy each round
        self.anti_pattern_probability = 0.15       # 15% chance to use anti-patterns
        self.psychological_tricks = ['mirror', 'reverse', 'frequency_avoid', 'random_burst', 'fake_pattern']
        
        # Initialize LSTM if available
        try:
            self.lstm_manager = LSTMManager()
            self.lstm_manager.load_model()  # Load the PyTorch model
            
            # Try to load ONNX model for faster inference
            try:
                self.lstm_manager.load_onnx_model()
                print("✅ ONNX LSTM model loaded for testing")
                self.onnx_available = True
            except Exception as onnx_error:
                print(f"⚠️ ONNX model not available, using PyTorch: {onnx_error}")
                self.onnx_available = False
            
            self.lstm_available = True
            print("✅ LSTM predictor loaded for testing")
        except Exception as e:
            print(f"⚠️ LSTM not available for testing: {e}")
            self.lstm_manager = None
            self.lstm_available = False
            self.onnx_available = False
    
    def simulate_human_player(self, round_num: int, game_length: int, ai_predictions: Optional[List[str]] = None, recent_moves: Optional[List[str]] = None) -> str:
        """
        Simulate realistic human playing patterns with counter-prediction and adaptive strategies
        
        Args:
            round_num: Current round number
            game_length: Total game length
            ai_predictions: Recent AI predictions (for counter-prediction)
            recent_moves: Recent human moves (for pattern awareness)
        """
        if recent_moves is None:
            recent_moves = []
        if ai_predictions is None:
            ai_predictions = []
            
        # Strategy 1: Counter-prediction behavior (humans try to outsmart AI)
        if ai_predictions and random.random() < self.counter_prediction_probability:
            return self._counter_predict_move(ai_predictions, recent_moves)
        
        # Strategy 2: Anti-pattern behavior (deliberately break patterns)
        if len(recent_moves) >= 3 and random.random() < self.anti_pattern_probability:
            return self._anti_pattern_move(recent_moves)
        
        # Strategy 3: Psychological tricks
        if random.random() < 0.25:  # 25% chance for psychological moves
            return self._psychological_trick_move(recent_moves, round_num, game_length)
        
        # Strategy 4: Adaptive strategy changes
        strategy_phase = self._get_current_strategy_phase(round_num, game_length, recent_moves)
        
        if strategy_phase == 'exploration':
            return self._exploration_phase_move(round_num, game_length)
        elif strategy_phase == 'pattern_building':
            return self._pattern_building_move(recent_moves, round_num)
        elif strategy_phase == 'pattern_breaking':
            return self._pattern_breaking_move(recent_moves)
        elif strategy_phase == 'endgame':
            return self._endgame_move(recent_moves, round_num, game_length)
        else:
            return self._adaptive_move(recent_moves, round_num)
    
    def _counter_predict_move(self, ai_predictions: List[str], recent_moves: List[str]) -> str:
        """Human tries to counter what they think AI will predict"""
        if not ai_predictions:
            return random.choice(self.moves)
        
        # Assume human thinks AI will predict their most frequent recent move
        if len(recent_moves) >= 3:
            recent_freq = {move: recent_moves[-5:].count(move) for move in self.moves}
            predicted_human_move = max(recent_freq.keys(), key=lambda x: recent_freq[x])
            
            # AI would counter with: rock->paper, paper->scissors, scissors->rock
            expected_ai_counter = {'rock': 'paper', 'paper': 'scissors', 'scissors': 'rock'}[predicted_human_move]
            
            # Human counters the AI's counter: paper->scissors, scissors->rock, rock->paper
            human_counter = {'paper': 'scissors', 'scissors': 'rock', 'rock': 'paper'}[expected_ai_counter]
            return human_counter
        
        return random.choice(self.moves)
    
    def _anti_pattern_move(self, recent_moves: List[str]) -> str:
        """Deliberately break detected patterns"""
        if len(recent_moves) < 3:
            return random.choice(self.moves)
        
        # Detect if there's a simple pattern and break it
        last_three = recent_moves[-3:]
        
        # Break cyclic patterns
        if last_three == ['rock', 'paper', 'scissors']:
            return random.choice(['rock', 'paper'])  # Don't continue cycle
        if last_three == ['paper', 'scissors', 'rock']:
            return random.choice(['scissors', 'paper'])
        
        # Break repetition patterns
        if len(set(last_three)) == 1:  # All same move
            other_moves = [m for m in self.moves if m != last_three[0]]
            return random.choice(other_moves)
        
        # Break alternating patterns
        if len(recent_moves) >= 4:
            if recent_moves[-4::2] == recent_moves[-2::2]:  # A-B-A-B pattern
                return random.choice(self.moves)
        
        return random.choice(self.moves)
    
    def _psychological_trick_move(self, recent_moves: List[str], round_num: int, game_length: int) -> str:
        """Apply psychological tactics"""
        trick = random.choice(self.psychological_tricks)
        
        if trick == 'mirror' and recent_moves:
            # Copy last move (humans sometimes mirror)
            return recent_moves[-1]
        
        elif trick == 'reverse' and len(recent_moves) >= 2:
            # Do the opposite of recent pattern
            last_move = recent_moves[-1]
            reverse_map = {'rock': 'scissors', 'paper': 'rock', 'scissors': 'paper'}
            return reverse_map[last_move]
        
        elif trick == 'frequency_avoid':
            # Avoid most frequent recent move
            if len(recent_moves) >= 5:
                freq = {move: recent_moves[-10:].count(move) for move in self.moves}
                most_frequent = max(freq.keys(), key=lambda x: freq[x])
                avoid_moves = [m for m in self.moves if m != most_frequent]
                return random.choice(avoid_moves)
        
        elif trick == 'random_burst':
            # Pure randomness to reset AI expectations
            return random.choice(self.moves)
        
        elif trick == 'fake_pattern':
            # Start a fake pattern that will be broken later
            phase = round_num % 6
            if phase < 3:
                return ['rock', 'paper', 'scissors'][phase]
            else:
                return random.choice(self.moves)
        
        return random.choice(self.moves)
    
    def _get_current_strategy_phase(self, round_num: int, game_length: int, recent_moves: List[str]) -> str:
        """Determine current strategic phase based on game progress"""
        progress = round_num / game_length
        
        # Change strategy randomly sometimes
        if recent_moves and random.random() < self.strategy_change_probability:
            return random.choice(['exploration', 'pattern_building', 'pattern_breaking', 'adaptive'])
        
        if progress < 0.2:
            return 'exploration'
        elif progress < 0.5:
            return 'pattern_building'
        elif progress < 0.8:
            return 'pattern_breaking'
        else:
            return 'endgame'
    
    def _exploration_phase_move(self, round_num: int, game_length: int) -> str:
        """Early game: test different moves to see AI reactions"""
        # Cycle through moves to test AI responses
        return self.moves[round_num % 3]
    
    def _pattern_building_move(self, recent_moves: List[str], round_num: int) -> str:
        """Mid game: establish patterns (some fake, some real)"""
        # Build a pattern for a few moves
        pattern_length = 4
        pattern_position = round_num % pattern_length
        
        patterns = [
            ['rock', 'rock', 'paper', 'scissors'],
            ['paper', 'scissors', 'rock', 'paper'],
            ['scissors', 'rock', 'rock', 'paper'],
            ['rock', 'paper', 'scissors', 'scissors']
        ]
        
        chosen_pattern = patterns[round_num // pattern_length % len(patterns)]
        return chosen_pattern[pattern_position]
    
    def _pattern_breaking_move(self, recent_moves: List[str]) -> str:
        """Late mid game: break established patterns"""
        if len(recent_moves) >= 3:
            # Analyze recent pattern and break it
            return self._anti_pattern_move(recent_moves)
        return random.choice(self.moves)
    
    def _endgame_move(self, recent_moves: List[str], round_num: int, game_length: int) -> str:
        """End game: unpredictable to maximize final wins"""
        # High randomness in endgame
        if random.random() < 0.7:
            return random.choice(self.moves)
        
        # Sometimes use counter-prediction in endgame
        return self._counter_predict_move([], recent_moves)
    
    def _adaptive_move(self, recent_moves: List[str], round_num: int) -> str:
        """Default adaptive strategy"""
        # Mix of randomness and weak patterns
        if random.random() < 0.6:
            return random.choice(self.moves)
        
        # Weak pattern based on round number
        return self.moves[(round_num * 2) % 3]
    
    def evaluate_prediction_accuracy(self, model_name: str, predictions: List[str], 
                                   actual_moves: List[str]) -> Dict[str, float]:
        """Evaluate prediction accuracy with detailed metrics"""
        if len(predictions) != len(actual_moves):
            min_len = min(len(predictions), len(actual_moves))
            predictions = predictions[:min_len]
            actual_moves = actual_moves[:min_len]
        
        correct = sum(1 for p, a in zip(predictions, actual_moves) if p == a)
        total = len(predictions)
        accuracy = (correct / total) * 100 if total > 0 else 0
        
        # Per-move accuracy
        move_accuracy = {}
        for move in self.moves:
            move_predictions = [p for p, a in zip(predictions, actual_moves) if a == move]
            move_correct = [p for p, a in zip(predictions, actual_moves) if p == a and a == move]
            move_accuracy[move] = (len(move_correct) / len(move_predictions) * 100) if len(move_predictions) > 0 else 0
        
        return {
            'overall_accuracy': accuracy,
            'correct_predictions': correct,
            'total_predictions': total,
            'rock_accuracy': move_accuracy['rock'],
            'paper_accuracy': move_accuracy['paper'],
            'scissors_accuracy': move_accuracy['scissors']
        }
    
    def run_lstm_test(self, game_length: int, test_id: int) -> Dict[str, Any]:
        """Run LSTM model test for specified game length"""
        if not self.lstm_available:
            return {'error': 'LSTM not available'}
        
        human_moves = []
        lstm_predictions = []
        robot_moves = []
        results = []
        ai_prediction_history = []  # Track AI predictions for realistic human counter-prediction
        
        print(f"  🧠 Running LSTM test (length: {game_length}, test: {test_id})")
        
        for round_num in range(game_length):
            # Generate realistic human move with AI prediction awareness
            human_move = self.simulate_human_player(
                round_num, 
                game_length, 
                ai_prediction_history[-5:] if ai_prediction_history else None,  # Last 5 AI predictions
                human_moves[-10:] if human_moves else None  # Last 10 human moves
            )
            human_moves.append(human_move)
            
            # Get LSTM prediction (if we have enough history)
            if len(human_moves) >= 2 and self.lstm_manager is not None:
                try:
                    # Try ONNX first (faster), fallback to PyTorch
                    if self.onnx_available:
                        lstm_probs = self.lstm_manager.predict_with_onnx(human_moves[:-1])
                    else:
                        # Use PyTorch prediction with sequence length limitation  
                        sequence = human_moves[-5:] if len(human_moves) > 5 else human_moves[:-1]
                        lstm_probs = self.lstm_manager.predict(sequence, use_onnx=False)
                    
                    predicted_human_move = max(lstm_probs.items(), key=lambda x: x[1])[0]
                    lstm_predictions.append(predicted_human_move)
                    ai_prediction_history.append(predicted_human_move)  # Track for human counter-prediction
                    
                    # Robot counters the predicted human move
                    robot_move = get_counter_move(predicted_human_move)
                except Exception as e:
                    print(f"    ⚠️ LSTM error: {e}")
                    robot_move = random.choice(self.moves)
                    lstm_predictions.append(random.choice(self.moves))
            else:
                robot_move = random.choice(self.moves)
                lstm_predictions.append(random.choice(self.moves))
            
            robot_moves.append(robot_move)
            
            # Determine round result
            if human_move == robot_move:
                result = 'tie'
            elif (human_move == 'rock' and robot_move == 'scissors') or \
                 (human_move == 'paper' and robot_move == 'rock') or \
                 (human_move == 'scissors' and robot_move == 'paper'):
                result = 'human'
            else:
                result = 'robot'
            
            results.append(result)
        
        # Calculate metrics
        robot_wins = results.count('robot')
        human_wins = results.count('human')
        ties = results.count('tie')
        robot_win_rate = (robot_wins / game_length) * 100
        
        # Prediction accuracy (exclude first move)
        if len(lstm_predictions) > 1:
            accuracy_metrics = self.evaluate_prediction_accuracy('LSTM', lstm_predictions[1:], human_moves[1:])
        else:
            accuracy_metrics = {'overall_accuracy': 0, 'correct_predictions': 0, 'total_predictions': 0}
        
        return {
            'model': 'LSTM',
            'game_length': game_length,
            'test_id': test_id,
            'robot_wins': robot_wins,
            'human_wins': human_wins,
            'ties': ties,
            'robot_win_rate': robot_win_rate,
            'human_win_rate': (human_wins / game_length) * 100,
            'tie_rate': (ties / game_length) * 100,
            'prediction_accuracy': accuracy_metrics['overall_accuracy'],
            'correct_predictions': accuracy_metrics['correct_predictions'],
            'total_predictions': accuracy_metrics['total_predictions'],
            'detailed_accuracy': accuracy_metrics
        }
    
    def run_markov_test(self, game_length: int, test_id: int) -> Dict[str, Any]:
        """Run Markov model test for specified game length"""
        human_moves = []
        markov_predictions = []
        robot_moves = []
        results = []
        ai_prediction_history = []  # Track AI predictions for realistic human counter-prediction
        
        print(f"  🔗 Running Markov test (length: {game_length}, test: {test_id})")
        
        # Reset Markov model for this test
        self.markov_strategy = MarkovStrategy()
        
        for round_num in range(game_length):
            # Generate realistic human move with AI prediction awareness
            human_move = self.simulate_human_player(
                round_num, 
                game_length, 
                ai_prediction_history[-5:] if ai_prediction_history else None,  # Last 5 AI predictions
                human_moves[-10:] if human_moves else None  # Last 10 human moves
            )
            human_moves.append(human_move)
            
            # Get Markov prediction
            if len(human_moves) >= 2:
                try:
                    self.markov_strategy.train(human_moves[:-1])
                    markov_result = self.markov_strategy.predict(human_moves[:-1])
                    
                    # Handle different return formats
                    if isinstance(markov_result, tuple):
                        robot_move, confidence = markov_result
                        # Convert robot move back to predicted human move
                        reverse_counter = {'scissors': 'paper', 'rock': 'scissors', 'paper': 'rock'}
                        predicted_human_move = reverse_counter.get(robot_move, random.choice(self.moves))
                    else:
                        robot_move = markov_result
                        reverse_counter = {'scissors': 'paper', 'rock': 'scissors', 'paper': 'rock'}
                        predicted_human_move = reverse_counter.get(robot_move, random.choice(self.moves))
                    
                    markov_predictions.append(predicted_human_move)
                    ai_prediction_history.append(predicted_human_move)  # Track for human counter-prediction
                except Exception as e:
                    print(f"    ⚠️ Markov error: {e}")
                    robot_move = random.choice(self.moves)
                    markov_predictions.append(random.choice(self.moves))
            else:
                robot_move = random.choice(self.moves)
                markov_predictions.append(random.choice(self.moves))
            
            robot_moves.append(robot_move)
            
            # Determine round result
            if human_move == robot_move:
                result = 'tie'
            elif (human_move == 'rock' and robot_move == 'scissors') or \
                 (human_move == 'paper' and robot_move == 'rock') or \
                 (human_move == 'scissors' and robot_move == 'paper'):
                result = 'human'
            else:
                result = 'robot'
            
            results.append(result)
        
        # Calculate metrics
        robot_wins = results.count('robot')
        human_wins = results.count('human')
        ties = results.count('tie')
        robot_win_rate = (robot_wins / game_length) * 100
        
        # Prediction accuracy (exclude first move)
        if len(markov_predictions) > 1:
            accuracy_metrics = self.evaluate_prediction_accuracy('Markov', markov_predictions[1:], human_moves[1:])
        else:
            accuracy_metrics = {'overall_accuracy': 0, 'correct_predictions': 0, 'total_predictions': 0}
        
        return {
            'model': 'Markov',
            'game_length': game_length,
            'test_id': test_id,
            'robot_wins': robot_wins,
            'human_wins': human_wins,
            'ties': ties,
            'robot_win_rate': robot_win_rate,
            'human_win_rate': (human_wins / game_length) * 100,
            'tie_rate': (ties / game_length) * 100,
            'prediction_accuracy': accuracy_metrics['overall_accuracy'],
            'correct_predictions': accuracy_metrics['correct_predictions'],
            'total_predictions': accuracy_metrics['total_predictions'],
            'detailed_accuracy': accuracy_metrics
        }
    
    def run_comprehensive_tests(self, tests_per_length: int = 5) -> Dict[str, Any]:
        """Run comprehensive tests across all game lengths"""
        print("🚀 Starting comprehensive AI difficulty testing...")
        print(f"📊 Testing lengths: {self.test_lengths}")
        print(f"🔄 Tests per length: {tests_per_length}")
        print(f"🎯 Strategy: {self.strategy_preference}, Personality: {self.personality}")
        
        all_results = []
        
        for game_length in self.test_lengths:
            print(f"\n📏 Testing game length: {game_length}")
            
            for test_id in range(tests_per_length):
                # Test LSTM
                if self.lstm_available:
                    lstm_result = self.run_lstm_test(game_length, test_id + 1)
                    all_results.append(lstm_result)
                
                # Test Markov
                markov_result = self.run_markov_test(game_length, test_id + 1)
                all_results.append(markov_result)
        
        return self.analyze_results(all_results)
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze test results and provide recommendations"""
        print("\n📈 Analyzing results...")
        
        # Group results by model and game length
        lstm_results = defaultdict(list)
        markov_results = defaultdict(list)
        
        for result in results:
            if 'error' in result:
                continue
                
            if result['model'] == 'LSTM':
                lstm_results[result['game_length']].append(result)
            elif result['model'] == 'Markov':
                markov_results[result['game_length']].append(result)
        
        # Calculate averages
        analysis = {
            'lstm_performance': {},
            'markov_performance': {},
            'comparison': {},
            'recommendations': []
        }
        
        for game_length in self.test_lengths:
            if game_length in lstm_results:
                lstm_data = lstm_results[game_length]
                lstm_avg = {
                    'avg_robot_win_rate': statistics.mean([r['robot_win_rate'] for r in lstm_data]),
                    'avg_prediction_accuracy': statistics.mean([r['prediction_accuracy'] for r in lstm_data]),
                    'std_robot_win_rate': statistics.stdev([r['robot_win_rate'] for r in lstm_data]) if len(lstm_data) > 1 else 0,
                    'best_robot_win_rate': max([r['robot_win_rate'] for r in lstm_data]),
                    'worst_robot_win_rate': min([r['robot_win_rate'] for r in lstm_data])
                }
                analysis['lstm_performance'][game_length] = lstm_avg
            
            if game_length in markov_results:
                markov_data = markov_results[game_length]
                markov_avg = {
                    'avg_robot_win_rate': statistics.mean([r['robot_win_rate'] for r in markov_data]),
                    'avg_prediction_accuracy': statistics.mean([r['prediction_accuracy'] for r in markov_data]),
                    'std_robot_win_rate': statistics.stdev([r['robot_win_rate'] for r in markov_data]) if len(markov_data) > 1 else 0,
                    'best_robot_win_rate': max([r['robot_win_rate'] for r in markov_data]),
                    'worst_robot_win_rate': min([r['robot_win_rate'] for r in markov_data])
                }
                analysis['markov_performance'][game_length] = markov_avg
            
            # Compare LSTM vs Markov
            if game_length in lstm_results and game_length in markov_results:
                lstm_avg_win = analysis['lstm_performance'][game_length]['avg_robot_win_rate']
                markov_avg_win = analysis['markov_performance'][game_length]['avg_robot_win_rate']
                lstm_avg_acc = analysis['lstm_performance'][game_length]['avg_prediction_accuracy']
                markov_avg_acc = analysis['markov_performance'][game_length]['avg_prediction_accuracy']
                
                analysis['comparison'][game_length] = {
                    'lstm_advantage_win_rate': lstm_avg_win - markov_avg_win,
                    'lstm_advantage_accuracy': lstm_avg_acc - markov_avg_acc,
                    'better_model': 'LSTM' if lstm_avg_win > markov_avg_win else 'Markov'
                }
        
        # Generate recommendations
        analysis['recommendations'] = self.generate_recommendations(analysis)
        
        return analysis
    
    def generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations based on analysis"""
        recommendations = []
        
        # Check overall LSTM vs Markov performance
        lstm_better_count = 0
        markov_better_count = 0
        
        for game_length, comparison in analysis['comparison'].items():
            if comparison['better_model'] == 'LSTM':
                lstm_better_count += 1
            else:
                markov_better_count += 1
        
        if markov_better_count > lstm_better_count:
            recommendations.append("⚠️ CRITICAL: Markov is outperforming LSTM - LSTM needs significant improvement")
            recommendations.append("🔧 Recommendation: Increase LSTM model complexity (hidden_dim, num_layers)")
            recommendations.append("📚 Recommendation: Train LSTM on more diverse datasets")
            recommendations.append("🎯 Recommendation: Implement ensemble approach combining LSTM with pattern recognition")
        
        # Check for low win rates
        for model_type in ['lstm_performance', 'markov_performance']:
            performance = analysis[model_type]
            for game_length, metrics in performance.items():
                if metrics['avg_robot_win_rate'] < 35:  # Should be above random (33.33%)
                    model_name = 'LSTM' if model_type == 'lstm_performance' else 'Markov'
                    recommendations.append(f"⚠️ {model_name} win rate at {game_length} games is only {metrics['avg_robot_win_rate']:.1f}% - needs improvement")
        
        # Check prediction accuracy
        for model_type in ['lstm_performance', 'markov_performance']:
            performance = analysis[model_type]
            for game_length, metrics in performance.items():
                if metrics['avg_prediction_accuracy'] < 40:  # Should be significantly above random
                    model_name = 'LSTM' if model_type == 'lstm_performance' else 'Markov'
                    recommendations.append(f"🎯 {model_name} prediction accuracy at {game_length} games is only {metrics['avg_prediction_accuracy']:.1f}% - improve pattern recognition")
        
        # Check for inconsistency (high standard deviation)
        for model_type in ['lstm_performance', 'markov_performance']:
            performance = analysis[model_type]
            for game_length, metrics in performance.items():
                if metrics['std_robot_win_rate'] > 10:
                    model_name = 'LSTM' if model_type == 'lstm_performance' else 'Markov'
                    recommendations.append(f"📊 {model_name} performance is inconsistent at {game_length} games (std: {metrics['std_robot_win_rate']:.1f}%) - stabilize algorithm")
        
        if not recommendations:
            recommendations.append("✅ All models performing well - consider advanced optimizations")
        
        return recommendations
    
    def print_detailed_report(self, analysis: Dict[str, Any]):
        """Print detailed analysis report"""
        print("\n" + "="*80)
        print("🎯 AI DIFFICULTY TESTING - DETAILED REPORT")
        print("="*80)
        
        print(f"\n📊 LSTM PERFORMANCE:")
        if analysis['lstm_performance']:
            for game_length, metrics in analysis['lstm_performance'].items():
                print(f"  {game_length:4d} games: Win Rate: {metrics['avg_robot_win_rate']:5.1f}% ± {metrics['std_robot_win_rate']:4.1f}% | "
                      f"Accuracy: {metrics['avg_prediction_accuracy']:5.1f}% | Best: {metrics['best_robot_win_rate']:5.1f}%")
        else:
            print("  ❌ LSTM not available or no results")
        
        print(f"\n🔗 MARKOV PERFORMANCE:")
        for game_length, metrics in analysis['markov_performance'].items():
            print(f"  {game_length:4d} games: Win Rate: {metrics['avg_robot_win_rate']:5.1f}% ± {metrics['std_robot_win_rate']:4.1f}% | "
                  f"Accuracy: {metrics['avg_prediction_accuracy']:5.1f}% | Best: {metrics['best_robot_win_rate']:5.1f}%")
        
        print(f"\n⚔️  LSTM vs MARKOV COMPARISON:")
        for game_length, comparison in analysis['comparison'].items():
            advantage = comparison['lstm_advantage_win_rate']
            better = comparison['better_model']
            print(f"  {game_length:4d} games: {better} wins by {abs(advantage):5.1f}% | "
                  f"LSTM accuracy advantage: {comparison['lstm_advantage_accuracy']:+5.1f}%")
        
        print(f"\n🔧 RECOMMENDATIONS:")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "="*80)
    
    def save_results(self, analysis: Dict[str, Any], filename: Optional[str] = None):
        """Save detailed results to JSON file"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"ai_difficulty_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"💾 Results saved to: {filename}")


def main():
    """Main testing function"""
    platform = AITestPlatform()
    
    # Run comprehensive tests
    analysis = platform.run_comprehensive_tests(tests_per_length=2)
    
    # Print detailed report
    platform.print_detailed_report(analysis)
    
    # Save results
    platform.save_results(analysis)
    
    return analysis

if __name__ == "__main__":
    main()