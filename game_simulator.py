#!/usr/bin/env python3
"""
Rock-Paper-Scissors Game Simulator
==================================

This script simulates games against the AI system to find optimal human move sequences
that maximize win rates for different difficulty levels and game lengths.

Usage:
    python game_simulator.py --difficulty easy --game_length 50 --strategy to_win --personality neutral
"""

import os
import sys
import time
import json
import random
import itertools
import argparse
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass
import numpy as np

# Add the current directory to Python path to import game modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from game_context import build_game_context, set_opponent_parameters, get_ai_prediction, reset_ai_system
    GAME_CONTEXT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import game_context: {e}")
    GAME_CONTEXT_AVAILABLE = False
    # Define dummy functions
    def build_game_context(*args, **kwargs):
        return {}
    def set_opponent_parameters(*args, **kwargs):
        return False
    def get_ai_prediction(*args, **kwargs):
        return {}
    def reset_ai_system(*args, **kwargs):
        pass

try:
    from webapp.app import get_result
except ImportError:
    # Define fallback get_result function
    def get_result(robot_move: str, human_move: str) -> str:
        """Fallback result calculation"""
        if human_move == robot_move:
            return 'tie'
        elif (human_move == 'rock' and robot_move == 'scissors') or \
             (human_move == 'paper' and robot_move == 'rock') or \
             (human_move == 'scissors' and robot_move == 'paper'):
            return 'human'
        else:
            return 'robot'


@dataclass
class SimulationConfig:
    """Configuration for game simulation"""
    difficulty: str = 'medium'  # 'easy', 'medium', 'hard'
    strategy: str = 'to_win'    # AI strategy preference
    personality: str = 'neutral'  # AI personality
    game_length: int = 50       # Number of moves per game
    num_simulations: int = 1000  # Number of games to simulate
    max_sequences_to_test: int = 100  # Maximum number of move sequences to test
    random_seed: Optional[int] = None  # For reproducible results


@dataclass
class SimulationResult:
    """Results from a game simulation"""
    human_moves: List[str]
    robot_moves: List[str]
    results: List[str]  # 'human', 'robot', 'tie'
    human_win_rate: float
    robot_win_rate: float
    tie_rate: float
    pattern_strength: float
    predictability_score: float
    adaptation_rate: float


class GameSimulator:
    """Simulates RPS games against the AI system"""
    
    MOVES = ['rock', 'paper', 'scissors']
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        if config.random_seed:
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)
    
    def simulate_game(self, human_sequence: List[str]) -> SimulationResult:
        """
        Simulate a single game with the given human move sequence.
        
        Args:
            human_sequence: List of human moves to play
            
        Returns:
            SimulationResult with game outcome and metrics
        """
        if not GAME_CONTEXT_AVAILABLE:
            return self._simulate_game_fallback(human_sequence)
        
        # Reset AI system for clean state
        reset_ai_system()
        
        # Set opponent parameters
        difficulty_map = {'easy': 'rookie', 'medium': 'challenger', 'hard': 'master'}
        ai_difficulty = difficulty_map.get(self.config.difficulty, 'challenger')
        
        success = set_opponent_parameters(ai_difficulty, self.config.strategy, self.config.personality)
        if not success:
            print(f"Warning: Failed to set opponent parameters, using fallback")
            return self._simulate_game_fallback(human_sequence)
        
        # Simulate the game
        human_moves = []
        robot_moves = []
        results = []
        
        for i, human_move in enumerate(human_sequence):
            if i >= self.config.game_length:
                break
                
            # Create session data for AI prediction
            session_data = {
                'human_moves': human_moves.copy(),
                'results': results.copy(),
                'ai_difficulty': ai_difficulty,
                'strategy_preference': self.config.strategy,
                'personality': self.config.personality
            }
            
            # Get AI prediction
            prediction_data = get_ai_prediction(session_data)
            robot_move = prediction_data.get('ai_move', random.choice(self.MOVES))
            
            # Determine result
            result = get_result(robot_move, human_move)
            
            # Update game state
            human_moves.append(human_move)
            robot_moves.append(robot_move)
            results.append(result)
        
        # Calculate metrics using game_context
        session = {
            'human_moves': human_moves,
            'robot_moves': robot_moves,
            'results': results,
            'round': len(human_moves),
            'ai_difficulty': ai_difficulty,
            'strategy_preference': self.config.strategy,
            'personality': self.config.personality
        }
        
        context = build_game_context(session)
        metrics = context['game_status']['metrics']
        
        # Calculate win rates
        human_wins = results.count('human')
        robot_wins = results.count('robot')
        ties = results.count('tie')
        total = len(results)
        
        return SimulationResult(
            human_moves=human_moves,
            robot_moves=robot_moves,
            results=results,
            human_win_rate=human_wins / total if total > 0 else 0.0,
            robot_win_rate=robot_wins / total if total > 0 else 0.0,
            tie_rate=ties / total if total > 0 else 0.0,
            pattern_strength=metrics.get('predictability_score', 0.0),
            predictability_score=metrics.get('predictability_score', 0.0),
            adaptation_rate=metrics.get('sbc_metrics', {}).get('strategic_analysis', {}).get('adaptation_speed', 0.0)
        )
    
    def _simulate_game_fallback(self, human_sequence: List[str]) -> SimulationResult:
        """Fallback simulation when game context is not available"""
        human_moves = []
        robot_moves = []
        results = []
        
        # Simple counter-strategy AI for fallback
        move_counter = Counter()
        
        for i, human_move in enumerate(human_sequence):
            if i >= self.config.game_length:
                break
            
            # Simple prediction based on frequency
            if i > 2:
                most_common = move_counter.most_common(1)
                if most_common:
                    predicted_human = most_common[0][0]
                    # Counter the predicted move
                    counter_map = {'rock': 'paper', 'paper': 'scissors', 'scissors': 'rock'}
                    robot_move = counter_map[predicted_human]
                else:
                    robot_move = random.choice(self.MOVES)
            else:
                robot_move = random.choice(self.MOVES)
            
            # Determine result
            if human_move == robot_move:
                result = 'tie'
            elif (human_move == 'rock' and robot_move == 'scissors') or \
                 (human_move == 'paper' and robot_move == 'rock') or \
                 (human_move == 'scissors' and robot_move == 'paper'):
                result = 'human'
            else:
                result = 'robot'
            
            human_moves.append(human_move)
            robot_moves.append(robot_move)
            results.append(result)
            move_counter[human_move] += 1
        
        # Calculate basic metrics
        human_wins = results.count('human')
        robot_wins = results.count('robot')
        ties = results.count('tie')
        total = len(results)
        
        # Calculate simple predictability
        if len(human_moves) > 0:
            move_counts = Counter(human_moves)
            max_count = max(move_counts.values())
            predictability = (max_count / len(human_moves)) * 100
        else:
            predictability = 0.0
        
        return SimulationResult(
            human_moves=human_moves,
            robot_moves=robot_moves,
            results=results,
            human_win_rate=human_wins / total if total > 0 else 0.0,
            robot_win_rate=robot_wins / total if total > 0 else 0.0,
            tie_rate=ties / total if total > 0 else 0.0,
            pattern_strength=predictability,
            predictability_score=predictability,
            adaptation_rate=0.5  # Default adaptation rate
        )
    
    def generate_move_sequences(self) -> List[List[str]]:
        """
        Generate various human move sequences to test.
        
        Returns:
            List of move sequences to simulate
        """
        sequences = []
        
        # 1. Pure random sequences
        for _ in range(20):
            sequence = [random.choice(self.MOVES) for _ in range(self.config.game_length)]
            sequences.append(sequence)
        
        # 2. Pattern-based sequences
        patterns = [
            # Repeating patterns
            ['rock'] * self.config.game_length,
            ['paper'] * self.config.game_length,
            ['scissors'] * self.config.game_length,
            
            # Cycling patterns
            (['rock', 'paper', 'scissors'] * (self.config.game_length // 3 + 1))[:self.config.game_length],
            (['scissors', 'paper', 'rock'] * (self.config.game_length // 3 + 1))[:self.config.game_length],
            (['rock', 'scissors', 'paper'] * (self.config.game_length // 3 + 1))[:self.config.game_length],
            
            # Alternating patterns
            (['rock', 'paper'] * (self.config.game_length // 2 + 1))[:self.config.game_length],
            (['rock', 'scissors'] * (self.config.game_length // 2 + 1))[:self.config.game_length],
            (['paper', 'scissors'] * (self.config.game_length // 2 + 1))[:self.config.game_length],
        ]
        sequences.extend(patterns)
        
        # 3. Counter-intuitive sequences (trying to be unpredictable)
        for _ in range(10):
            sequence = []
            for i in range(self.config.game_length):
                # Avoid recently used moves
                if len(sequence) >= 2:
                    recent = sequence[-2:]
                    available = [m for m in self.MOVES if m not in recent]
                    if available:
                        sequence.append(random.choice(available))
                    else:
                        sequence.append(random.choice(self.MOVES))
                else:
                    sequence.append(random.choice(self.MOVES))
            sequences.append(sequence)
        
        # 4. Frequency-biased sequences
        frequency_biases = [
            (0.5, 0.3, 0.2),  # Rock-heavy
            (0.3, 0.5, 0.2),  # Paper-heavy
            (0.2, 0.3, 0.5),  # Scissors-heavy
            (0.4, 0.4, 0.2),  # Balanced rock/paper
            (0.4, 0.2, 0.4),  # Balanced rock/scissors
            (0.2, 0.4, 0.4),  # Balanced paper/scissors
        ]
        
        for rock_prob, paper_prob, scissors_prob in frequency_biases:
            sequence = []
            for _ in range(self.config.game_length):
                rand = random.random()
                if rand < rock_prob:
                    sequence.append('rock')
                elif rand < rock_prob + paper_prob:
                    sequence.append('paper')
                else:
                    sequence.append('scissors')
            sequences.append(sequence)
        
        # 5. Meta-gaming sequences (trying to exploit AI patterns)
        # Anti-frequency: if we played rock a lot, avoid it
        for _ in range(5):
            sequence = []
            move_counts = Counter({'rock': 0, 'paper': 0, 'scissors': 0})
            
            for i in range(self.config.game_length):
                if i > 5:  # Start anti-frequency after some moves
                    # Choose the least played move
                    least_played = move_counts.most_common()[-1][0]
                    sequence.append(least_played)
                    move_counts[least_played] += 1
                else:
                    move = random.choice(self.MOVES)
                    sequence.append(move)
                    move_counts[move] += 1
            sequences.append(sequence)
        
        # Limit the number of sequences to test
        if len(sequences) > self.config.max_sequences_to_test:
            sequences = random.sample(sequences, self.config.max_sequences_to_test)
        
        return sequences
    
    def run_optimization(self) -> Dict[str, Any]:
        """
        Run the full optimization to find the best human strategies.
        
        Returns:
            Dictionary with optimization results
        """
        print(f"Starting game simulation optimization...")
        print(f"Difficulty: {self.config.difficulty}")
        print(f"Game Length: {self.config.game_length}")
        print(f"AI Strategy: {self.config.strategy}")
        print(f"AI Personality: {self.config.personality}")
        print()
        
        # Generate move sequences to test
        sequences = self.generate_move_sequences()
        print(f"Generated {len(sequences)} move sequences to test")
        
        results = []
        best_win_rate = 0.0
        best_sequence = None
        
        for i, sequence in enumerate(sequences):
            if i % 10 == 0:
                print(f"Testing sequence {i+1}/{len(sequences)}...")
            
            # Simulate the game with this sequence
            result = self.simulate_game(sequence)
            results.append({
                'sequence': sequence,
                'result': result,
                'sequence_type': self._classify_sequence(sequence)
            })
            
            # Track best performance
            if result.human_win_rate > best_win_rate:
                best_win_rate = result.human_win_rate
                best_sequence = sequence
        
        # Analyze results
        analysis = self._analyze_results(results)
        
        print(f"\n=== OPTIMIZATION RESULTS ===")
        print(f"Best Human Win Rate: {best_win_rate:.1%}")
        print(f"Best Sequence Type: {analysis['best_sequence_type']}")
        print(f"Average Win Rate: {analysis['average_win_rate']:.1%}")
        print(f"Robust Strategy (>45% win rate): {'Yes' if best_win_rate > 0.45 else 'No'}")
        
        return {
            'config': self.config,
            'best_win_rate': best_win_rate,
            'best_sequence': best_sequence,
            'all_results': results,
            'analysis': analysis,
            'is_robust': best_win_rate > 0.45
        }
    
    def _classify_sequence(self, sequence: List[str]) -> str:
        """Classify the type of move sequence"""
        if len(set(sequence)) == 1:
            return 'pure_repetition'
        elif len(set(sequence)) == 2:
            return 'two_move_pattern'
        elif sequence == ['rock', 'paper', 'scissors'] * (len(sequence) // 3):
            return 'rps_cycle'
        elif len(sequence) > 10 and all(sequence[i] != sequence[i+1] for i in range(len(sequence)-1)):
            return 'no_consecutive'
        else:
            # Check for frequency bias
            counts = Counter(sequence)
            max_freq = max(counts.values()) / len(sequence)
            if max_freq > 0.6:
                return 'frequency_biased'
            elif max_freq < 0.4:
                return 'balanced'
            else:
                return 'mixed_pattern'
    
    def _analyze_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze simulation results to find patterns"""
        win_rates = [r['result'].human_win_rate for r in results]
        
        # Group by sequence type
        type_performance = defaultdict(list)
        for r in results:
            type_performance[r['sequence_type']].append(r['result'].human_win_rate)
        
        # Find best performing type
        type_averages = {t: np.mean(rates) for t, rates in type_performance.items()}
        best_type = max(type_averages.keys(), key=lambda k: type_averages[k])
        
        return {
            'average_win_rate': np.mean(win_rates),
            'max_win_rate': max(win_rates),
            'min_win_rate': min(win_rates),
            'std_win_rate': np.std(win_rates),
            'best_sequence_type': best_type,
            'type_performance': dict(type_averages),
            'sequences_above_45_percent': sum(1 for wr in win_rates if wr > 0.45),
            'robust_strategies_found': sum(1 for wr in win_rates if wr > 0.45) > 0
        }


def main():
    """Main function to run the game simulation"""
    parser = argparse.ArgumentParser(description='Rock-Paper-Scissors Game Simulator')
    parser.add_argument('--difficulty', choices=['easy', 'medium', 'hard'], default='medium',
                        help='AI difficulty level')
    parser.add_argument('--game_length', type=int, default=50,
                        help='Number of moves per game')
    parser.add_argument('--strategy', choices=['to_win', 'not_to_lose'], default='to_win',
                        help='AI strategy preference')
    parser.add_argument('--personality', default='neutral',
                        help='AI personality')
    parser.add_argument('--num_sequences', type=int, default=100,
                        help='Number of move sequences to test')
    parser.add_argument('--output', type=str, help='Output file for results (JSON)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducible results')
    
    args = parser.parse_args()
    
    # Create simulation config
    config = SimulationConfig(
        difficulty=args.difficulty,
        strategy=args.strategy,
        personality=args.personality,
        game_length=args.game_length,
        max_sequences_to_test=args.num_sequences,
        random_seed=args.seed
    )
    
    # Run simulation
    simulator = GameSimulator(config)
    results = simulator.run_optimization()
    
    # Save results if output file specified
    if args.output:
        # Convert results to JSON-serializable format
        json_results = {
            'config': {
                'difficulty': config.difficulty,
                'strategy': config.strategy,
                'personality': config.personality,
                'game_length': config.game_length,
                'max_sequences_to_test': config.max_sequences_to_test,
                'random_seed': config.random_seed
            },
            'best_win_rate': results['best_win_rate'],
            'best_sequence': results['best_sequence'],
            'analysis': results['analysis'],
            'is_robust': results['is_robust'],
            'timestamp': time.time()
        }
        
        with open(args.output, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()