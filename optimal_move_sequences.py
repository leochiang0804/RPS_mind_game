#!/usr/bin/env python3
"""
Optimal Move Sequences for Rock-Paper-Scissors AI Testing

This script generates and tests optimal move sequences designed to exploit common
AI strategy weaknesses in Rock-Paper-Scissors games. It tests these sequences
against all robot character combinations to identify the most effective patterns.
"""

import itertools
import json
import random
from typing import List, Dict, Tuple, Any
import sys
import os

# Add the current directory to the path to import game modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from strategy import EnhancedStrategy, FrequencyStrategy, MarkovStrategy
    from optimized_strategies import ToWinStrategy, NotToLoseStrategy
    from personality_engine import get_personality_engine
    from lstm_web_integration import get_lstm_predictor, init_lstm_model
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some game modules not available: {e}")
    IMPORTS_AVAILABLE = False

class MoveSequenceGenerator:
    """Generates optimal move sequences to beat AI strategies."""
    
    def __init__(self):
        self.moves = ['paper', 'stone', 'scissor']
        
    def generate_anti_frequency_sequence(self, length: int) -> List[str]:
        """
        Generate a sequence that defeats frequency-based AI strategies.
        Uses balanced distribution with anti-predictable patterns.
        """
        # Start with balanced distribution
        moves_per_type = length // 3
        remainder = length % 3
        
        sequence = (
            ['paper'] * (moves_per_type + (1 if remainder > 0 else 0)) +
            ['stone'] * (moves_per_type + (1 if remainder > 1 else 0)) +
            ['scissor'] * moves_per_type
        )
        
        # Shuffle to avoid predictable order
        random.shuffle(sequence)
        
        # Apply anti-frequency adjustments
        # Ensure no move appears more than 40% of the time
        max_per_move = int(length * 0.4)
        adjusted = []
        counts = {'paper': 0, 'stone': 0, 'scissor': 0}
        
        for move in sequence:
            if counts[move] < max_per_move:
                adjusted.append(move)
                counts[move] += 1
            else:
                # Find alternative move
                alternatives = [m for m in self.moves if counts[m] < max_per_move]
                if alternatives:
                    alt_move = random.choice(alternatives)
                    adjusted.append(alt_move)
                    counts[alt_move] += 1
                else:
                    adjusted.append(move)  # Fallback
        
        return adjusted[:length]
    
    def generate_anti_markov_sequence(self, length: int) -> List[str]:
        """
        Generate a sequence that defeats Markov chain predictions.
        Uses deliberate pattern breaks and high entropy transitions.
        """
        if length == 0:
            return []
        
        sequence = [random.choice(self.moves)]
        
        for i in range(1, length):
            # Never repeat the same move twice in a row
            available_moves = [m for m in self.moves if m != sequence[-1]]
            
            # If we have established a 2-move pattern, break it
            if len(sequence) >= 2:
                last_two = tuple(sequence[-2:])
                # Add some randomness to pattern breaking
                if random.random() < 0.7:  # 70% chance to break pattern
                    # Choose move that wasn't the predicted next in pattern
                    next_move = random.choice(available_moves)
                else:
                    # Sometimes continue pattern to avoid being too predictable
                    next_move = random.choice(available_moves)
            else:
                next_move = random.choice(available_moves)
            
            sequence.append(next_move)
        
        return sequence
    
    def generate_anti_lstm_sequence(self, length: int) -> List[str]:
        """
        Generate a sequence that defeats LSTM neural network predictions.
        Uses high entropy and deliberately breaks long-term patterns.
        """
        sequence = []
        
        # Use multiple random seeds to ensure unpredictability
        random_generators = [random.Random(i) for i in range(5)]
        
        for i in range(length):
            # Use different random generator for each position
            generator = random_generators[i % len(random_generators)]
            
            # Apply different strategies at different phases
            if i < 5:
                # Early game: establish false pattern
                move = ['paper', 'stone', 'scissor', 'paper', 'stone'][i]
            elif i < length // 2:
                # Mid game: high entropy, avoid recent moves
                recent_moves = set(sequence[-3:]) if len(sequence) >= 3 else set()
                available = [m for m in self.moves if m not in recent_moves]
                if not available:
                    available = self.moves
                move = generator.choice(available)
            else:
                # Late game: completely random to defeat adaptation
                move = generator.choice(self.moves)
            
            sequence.append(move)
        
        return sequence
    
    def generate_adaptive_sequence(self, length: int) -> List[str]:
        """
        Generate an adaptive sequence that changes strategy mid-game.
        This defeats AIs that try to adapt to your strategy.
        """
        sequence = []
        phase_length = length // 3
        
        # Phase 1: Establish predictable pattern
        for i in range(min(phase_length, length)):
            sequence.append(self.moves[i % 3])
        
        # Phase 2: Switch to anti-frequency if more moves available
        if len(sequence) < length:
            remaining = length - len(sequence)
            phase2_length = min(phase_length, remaining)
            anti_freq = self.generate_anti_frequency_sequence(phase2_length)
            sequence.extend(anti_freq)
        
        # Phase 3: Pure random for remaining moves
        while len(sequence) < length:
            sequence.append(random.choice(self.moves))
        
        return sequence[:length]
    
    def generate_all_sequences(self, length: int) -> Dict[str, List[str]]:
        """Generate all types of optimal sequences."""
        random.seed(42)  # For reproducible results
        
        sequences = {
            'anti_frequency': self.generate_anti_frequency_sequence(length),
            'anti_markov': self.generate_anti_markov_sequence(length),
            'anti_lstm': self.generate_anti_lstm_sequence(length),
            'adaptive': self.generate_adaptive_sequence(length),
            'balanced_random': [random.choice(self.moves) for _ in range(length)],
            'rock_paper_cycle': ['stone', 'paper'] * (length // 2) + (['stone'] if length % 2 else []),
            'entropy_maximizer': self._generate_entropy_maximizer(length)
        }
        
        return sequences
    
    def _generate_entropy_maximizer(self, length: int) -> List[str]:
        """Generate sequence with maximum entropy (most unpredictable)."""
        sequence = []
        for i in range(length):
            # Ensure no move appears more than it should in a balanced sequence
            counts = {move: sequence.count(move) for move in self.moves}
            target_count = (i + 1) // 3
            
            # Find moves that are under-represented
            available = [move for move in self.moves if counts[move] <= target_count]
            if not available:
                available = self.moves
            
            sequence.append(random.choice(available))
        
        return sequence


class AIStrategy:
    """Mock AI strategy for testing when imports are not available."""
    
    def __init__(self, name: str):
        self.name = name
        self.history = []
    
    def predict(self, human_history: List[str]) -> str:
        """Simple mock prediction."""
        if not human_history:
            return random.choice(['paper', 'stone', 'scissor'])
        
        # Simple frequency-based prediction
        counts = {'paper': 0, 'stone': 0, 'scissor': 0}
        for move in human_history:
            counts[move] = counts.get(move, 0) + 1
        
        # Predict most common move and counter it
        most_common = max(counts, key=counts.get)
        counter = {'paper': 'scissor', 'stone': 'paper', 'scissor': 'stone'}
        return counter[most_common]


class GameSimulator:
    """Simulates games to test move sequences against AI strategies."""
    
    def __init__(self):
        self.difficulties = ['random', 'frequency', 'markov', 'enhanced', 'lstm']
        self.strategies = ['balanced', 'to_win', 'not_to_lose']
        self.personalities = ['neutral', 'berserker', 'guardian', 'chameleon', 'professor', 'wildcard', 'mirror']
        
        # Initialize AI strategies if available
        if IMPORTS_AVAILABLE:
            self.enhanced_strategy = EnhancedStrategy(order=2, recency_weight=0.8)
            self.frequency_strategy = FrequencyStrategy()
            self.markov_strategy = MarkovStrategy()
            self.to_win_strategy = ToWinStrategy()
            self.not_to_lose_strategy = NotToLoseStrategy()
            self.personality_engine = get_personality_engine()
            
            # Initialize LSTM if available
            try:
                init_lstm_model()
                self.lstm_predictor = get_lstm_predictor()
            except:
                self.lstm_predictor = None
        else:
            # Create mock strategies
            self.enhanced_strategy = AIStrategy('enhanced')
            self.frequency_strategy = AIStrategy('frequency')
            self.markov_strategy = AIStrategy('markov')
            self.to_win_strategy = AIStrategy('to_win')
            self.not_to_lose_strategy = AIStrategy('not_to_lose')
            self.lstm_predictor = None
    
    def get_ai_move(self, human_history: List[str], difficulty: str, strategy: str, personality: str) -> str:
        """Get AI move based on difficulty, strategy, and personality."""
        if not IMPORTS_AVAILABLE:
            # Simple mock behavior
            if not human_history:
                return random.choice(['paper', 'stone', 'scissor'])
            
            # Counter the most recent move
            counter = {'paper': 'scissor', 'stone': 'paper', 'scissor': 'stone'}
            return counter.get(human_history[-1], random.choice(['paper', 'stone', 'scissor']))
        
        # Get base prediction based on difficulty
        if difficulty == 'random':
            predicted = random.choice(['paper', 'stone', 'scissor'])
        elif difficulty == 'frequency':
            predicted_counter = self.frequency_strategy.predict(human_history)
            reverse_counter = {'scissor': 'paper', 'stone': 'scissor', 'paper': 'stone'}
            predicted = reverse_counter.get(predicted_counter, random.choice(['paper', 'stone', 'scissor']))
        elif difficulty == 'markov':
            self.markov_strategy.train(human_history)
            predicted_counter = self.markov_strategy.predict(human_history)
            reverse_counter = {'scissor': 'paper', 'stone': 'scissor', 'paper': 'stone'}
            predicted = reverse_counter.get(predicted_counter, random.choice(['paper', 'stone', 'scissor']))
        elif difficulty == 'enhanced':
            self.enhanced_strategy.train(human_history)
            predicted_counter = self.enhanced_strategy.predict(human_history)
            reverse_counter = {'scissor': 'paper', 'stone': 'scissor', 'paper': 'stone'}
            predicted = reverse_counter.get(predicted_counter, random.choice(['paper', 'stone', 'scissor']))
        elif difficulty == 'lstm' and self.lstm_predictor:
            try:
                return self.lstm_predictor.get_counter_move(human_history)
            except:
                predicted = random.choice(['paper', 'stone', 'scissor'])
        else:
            predicted = random.choice(['paper', 'stone', 'scissor'])
        
        # Apply strategy modification
        counter = {'paper': 'scissor', 'scissor': 'stone', 'stone': 'paper'}
        base_move = counter.get(predicted, random.choice(['paper', 'stone', 'scissor']))
        
        # Apply personality modification (simplified)
        if personality == 'berserker':
            # More aggressive: stick with counter
            return base_move
        elif personality == 'guardian':
            # More defensive: sometimes choose defensively
            if random.random() < 0.3:
                return random.choice(['paper', 'stone', 'scissor'])
            return base_move
        elif personality == 'wildcard':
            # More random
            if random.random() < 0.4:
                return random.choice(['paper', 'stone', 'scissor'])
            return base_move
        else:
            return base_move
    
    def determine_winner(self, human_move: str, ai_move: str) -> str:
        """Determine the winner of a round."""
        if human_move == ai_move:
            return 'tie'
        
        win_conditions = {
            ('paper', 'stone'): 'human',
            ('stone', 'scissor'): 'human',
            ('scissor', 'paper'): 'human',
            ('stone', 'paper'): 'ai',
            ('scissor', 'stone'): 'ai',
            ('paper', 'scissor'): 'ai'
        }
        
        return win_conditions.get((human_move, ai_move), 'tie')
    
    def simulate_game(self, human_sequence: List[str], difficulty: str, strategy: str, personality: str) -> Dict[str, Any]:
        """Simulate a complete game with the given parameters."""
        ai_moves = []
        results = []
        wins = {'human': 0, 'ai': 0, 'tie': 0}
        
        for i, human_move in enumerate(human_sequence):
            # Get AI move based on history so far
            human_history = human_sequence[:i]
            ai_move = self.get_ai_move(human_history, difficulty, strategy, personality)
            ai_moves.append(ai_move)
            
            # Determine winner
            winner = self.determine_winner(human_move, ai_move)
            results.append(winner)
            wins[winner] += 1
        
        # Calculate statistics
        total_rounds = len(human_sequence)
        human_win_rate = (wins['human'] / total_rounds) * 100 if total_rounds > 0 else 0
        
        return {
            'human_moves': human_sequence,
            'ai_moves': ai_moves,
            'results': results,
            'wins': wins,
            'human_win_rate': human_win_rate,
            'total_rounds': total_rounds,
            'difficulty': difficulty,
            'strategy': strategy,
            'personality': personality
        }
    
    def test_sequence_against_all_combinations(self, sequence_name: str, human_sequence: List[str]) -> List[Dict[str, Any]]:
        """Test a sequence against all AI combinations."""
        results = []
        
        for difficulty in self.difficulties:
            for strategy in self.strategies:
                for personality in self.personalities:
                    try:
                        game_result = self.simulate_game(human_sequence, difficulty, strategy, personality)
                        game_result['sequence_name'] = sequence_name
                        game_result['combination'] = f"{difficulty}_{strategy}_{personality}"
                        results.append(game_result)
                    except Exception as e:
                        print(f"Error testing {difficulty}_{strategy}_{personality}: {e}")
                        continue
        
        return results


def main():
    """Main function to generate and test optimal move sequences."""
    print("🎮 Rock-Paper-Scissors Optimal Move Sequence Generator")
    print("=" * 60)
    
    # Generate sequences for both 25 and 50 moves
    generator = MoveSequenceGenerator()
    simulator = GameSimulator()
    
    all_results = {}
    
    for length in [25, 50]:
        print(f"\n📊 Testing {length}-move sequences...")
        
        # Generate all sequence types
        sequences = generator.generate_all_sequences(length)
        length_results = {}
        
        for seq_name, sequence in sequences.items():
            print(f"  Testing {seq_name} sequence...")
            
            # Test against all combinations
            test_results = simulator.test_sequence_against_all_combinations(seq_name, sequence)
            
            # Calculate summary statistics
            total_games = len(test_results)
            avg_win_rate = sum(r['human_win_rate'] for r in test_results) / total_games if total_games > 0 else 0
            
            # Find best and worst performing combinations
            best_combo = max(test_results, key=lambda x: x['human_win_rate']) if test_results else None
            worst_combo = min(test_results, key=lambda x: x['human_win_rate']) if test_results else None
            
            # Count how many combinations this sequence beats (>50% win rate)
            beats_count = sum(1 for r in test_results if r['human_win_rate'] > 50)
            
            length_results[seq_name] = {
                'sequence': sequence,
                'avg_win_rate': avg_win_rate,
                'beats_count': beats_count,
                'total_combinations': total_games,
                'best_combo': best_combo,
                'worst_combo': worst_combo,
                'all_results': test_results
            }
            
            print(f"    Average win rate: {avg_win_rate:.1f}%")
            print(f"    Beats {beats_count}/{total_games} combinations")
        
        all_results[f"{length}_moves"] = length_results
    
    # Generate comprehensive report
    generate_report(all_results)
    
    # Save detailed results to JSON file
    save_results_to_file(all_results)
    
    print("\n✅ Analysis complete! Check the generated report and results files.")


def generate_report(all_results: Dict[str, Any]):
    """Generate a comprehensive analysis report."""
    print("\n" + "="*80)
    print("📈 COMPREHENSIVE ANALYSIS REPORT")
    print("="*80)
    
    for length_key, length_results in all_results.items():
        length = length_key.replace('_moves', '')
        print(f"\n🎯 {length.upper()}-MOVE SEQUENCES ANALYSIS")
        print("-" * 50)
        
        # Sort sequences by average win rate
        sorted_sequences = sorted(length_results.items(), key=lambda x: x[1]['avg_win_rate'], reverse=True)
        
        print("🏆 SEQUENCE PERFORMANCE RANKING:")
        for rank, (seq_name, data) in enumerate(sorted_sequences, 1):
            print(f"  {rank}. {seq_name.upper().replace('_', ' ')}")
            print(f"     Average Win Rate: {data['avg_win_rate']:.1f}%")
            print(f"     Beats {data['beats_count']}/{data['total_combinations']} AI combinations")
            print(f"     Best vs: {data['best_combo']['combination']} ({data['best_combo']['human_win_rate']:.1f}%)")
            print(f"     Worst vs: {data['worst_combo']['combination']} ({data['worst_combo']['human_win_rate']:.1f}%)")
            print()
        
        # Find the most vulnerable AI combinations
        all_game_results = []
        for seq_data in length_results.values():
            all_game_results.extend(seq_data['all_results'])
        
        # Group by AI combination and calculate average loss rate
        combo_performance = {}
        for result in all_game_results:
            combo = result['combination']
            if combo not in combo_performance:
                combo_performance[combo] = []
            combo_performance[combo].append(result['human_win_rate'])
        
        # Calculate average human win rate against each combo
        combo_avg = {combo: sum(rates)/len(rates) for combo, rates in combo_performance.items()}
        most_vulnerable = sorted(combo_avg.items(), key=lambda x: x[1], reverse=True)
        
        print("🤖 MOST VULNERABLE AI COMBINATIONS:")
        for rank, (combo, avg_loss_rate) in enumerate(most_vulnerable[:5], 1):
            parts = combo.split('_')
            print(f"  {rank}. {parts[0].title()} + {parts[1].title()} + {parts[2].title()}")
            print(f"     Average human win rate against: {avg_loss_rate:.1f}%")
        
        print("\n🛡️ MOST RESILIENT AI COMBINATIONS:")
        for rank, (combo, avg_loss_rate) in enumerate(reversed(most_vulnerable[-5:]), 1):
            parts = combo.split('_')
            print(f"  {rank}. {parts[0].title()} + {parts[1].title()} + {parts[2].title()}")
            print(f"     Average human win rate against: {avg_loss_rate:.1f}%")


def save_results_to_file(all_results: Dict[str, Any]):
    """Save detailed results to JSON file."""
    # Create a simplified version for JSON serialization
    simplified_results = {}
    
    for length_key, length_results in all_results.items():
        simplified_results[length_key] = {}
        
        for seq_name, data in length_results.items():
            simplified_results[length_key][seq_name] = {
                'sequence': data['sequence'],
                'avg_win_rate': data['avg_win_rate'],
                'beats_count': data['beats_count'],
                'total_combinations': data['total_combinations'],
                'best_combo_name': data['best_combo']['combination'],
                'best_combo_win_rate': data['best_combo']['human_win_rate'],
                'worst_combo_name': data['worst_combo']['combination'],
                'worst_combo_win_rate': data['worst_combo']['human_win_rate']
            }
    
    # Save to file
    filename = 'optimal_sequences_results.json'
    with open(filename, 'w') as f:
        json.dump(simplified_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {filename}")
    
    # Also create a quick reference file with just the best sequences
    best_sequences = {}
    for length_key, length_results in all_results.items():
        # Find the sequence with highest average win rate
        best_seq_name = max(length_results.keys(), key=lambda x: length_results[x]['avg_win_rate'])
        best_data = length_results[best_seq_name]
        
        best_sequences[length_key] = {
            'name': best_seq_name,
            'sequence': best_data['sequence'],
            'avg_win_rate': best_data['avg_win_rate'],
            'beats_count': best_data['beats_count']
        }
    
    quick_ref_filename = 'best_sequences_quick_ref.json'
    with open(quick_ref_filename, 'w') as f:
        json.dump(best_sequences, f, indent=2)
    
    print(f"📋 Quick reference saved to: {quick_ref_filename}")


if __name__ == '__main__':
    main()