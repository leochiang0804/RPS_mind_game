#!/usr/bin/env python3
"""
Robot Distinctiveness Simulator

Uses your actual game codebase to test all 105 AI robot combinations
against optimal human move sequences to determine behavioral distinctiveness.
"""

import json
import time
import statistics
import random
from typing import List, Dict, Any, Tuple
import sys
import os
from collections import defaultdict, Counter

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class RobotDistinctivenessSimulator:
    """Simulates games using actual codebase to test robot distinctiveness."""
    
    def __init__(self):
        self.load_game_modules()
        self.load_optimal_sequences()
        self.robot_combinations = self._generate_robot_combinations()
        self.simulation_results = []
        self.choices = ['paper', 'stone', 'scissor']
    
    def load_game_modules(self):
        """Load the actual game modules from your codebase."""
        try:
            # Import the actual strategy modules
            from strategy import RandomStrategy, FrequencyStrategy, MarkovStrategy, EnhancedStrategy
            from optimized_strategies import ToWinStrategy, NotToLoseStrategy
            from personality_engine import get_personality_engine, PersonalityProfile
            
            # Try to import LSTM
            try:
                from lstm_web_integration import get_lstm_predictor
                self.LSTM_AVAILABLE = True
                print("✅ LSTM integration available")
            except ImportError:
                self.LSTM_AVAILABLE = False
                print("⚠️ LSTM not available - using enhanced strategy instead")
            
            # Store the imported classes
            self.RandomStrategy = RandomStrategy
            self.FrequencyStrategy = FrequencyStrategy
            self.MarkovStrategy = MarkovStrategy
            self.EnhancedStrategy = EnhancedStrategy
            self.ToWinStrategy = ToWinStrategy
            self.NotToLoseStrategy = NotToLoseStrategy
            self.get_personality_engine = get_personality_engine
            
            print("✅ Successfully loaded actual game modules")
            
        except ImportError as e:
            print(f"❌ Failed to import game modules: {e}")
            print("Creating fallback simulation...")
            self.create_fallback_simulation()
    
    def create_fallback_simulation(self):
        """Create a fallback simulation if modules aren't available."""
        print("🔧 Using simplified fallback simulation")
        
        class FallbackStrategy:
            def __init__(self, strategy_type='random'):
                self.strategy_type = strategy_type
                self.history = []
                
            def predict(self, history):
                if self.strategy_type == 'random':
                    return random.choice(['paper', 'stone', 'scissor'])
                elif self.strategy_type == 'frequency':
                    if len(history) < 3:
                        return random.choice(['paper', 'stone', 'scissor'])
                    counts = Counter(history[-10:])  # Look at last 10 moves
                    most_common = counts.most_common(1)[0][0]
                    # Counter the most common
                    counters = {'paper': 'scissor', 'stone': 'paper', 'scissor': 'stone'}
                    return counters.get(most_common, 'paper')
                else:
                    return random.choice(['paper', 'stone', 'scissor'])
        
        self.RandomStrategy = lambda: FallbackStrategy('random')
        self.FrequencyStrategy = lambda: FallbackStrategy('frequency')
        self.MarkovStrategy = lambda: FallbackStrategy('frequency')  # Simplified
        self.EnhancedStrategy = lambda **kwargs: FallbackStrategy('frequency')
        self.ToWinStrategy = lambda: FallbackStrategy('frequency')
        self.NotToLoseStrategy = lambda: FallbackStrategy('random')
        self.LSTM_AVAILABLE = False
        
        def fallback_personality_engine(personality):
            return {'modify_move': lambda move, **kwargs: move}
        
        self.get_personality_engine = fallback_personality_engine
    
    def load_optimal_sequences(self):
        """Load the optimal sequences from JSON file."""
        try:
            with open('best_sequences_quick_ref.json', 'r') as f:
                self.sequences = json.load(f)
            print("✅ Loaded optimal sequences successfully")
            print(f"   • 25-move sequence: {self.sequences['25_moves']['name']} (avg win rate: {self.sequences['25_moves']['avg_win_rate']:.1f}%)")
            print(f"   • 50-move sequence: {self.sequences['50_moves']['name']} (avg win rate: {self.sequences['50_moves']['avg_win_rate']:.1f}%)")
        except FileNotFoundError:
            print("❌ Optimal sequences not found. Creating default sequences...")
            self.sequences = {
                '25_moves': {
                    'name': 'anti_pattern_25',
                    'sequence': ['paper', 'stone', 'scissor'] * 8 + ['paper'],
                    'avg_win_rate': 33.3
                },
                '50_moves': {
                    'name': 'anti_pattern_50', 
                    'sequence': ['paper', 'stone', 'scissor'] * 16 + ['paper', 'stone'],
                    'avg_win_rate': 33.3
                }
            }
    
    def _generate_robot_combinations(self):
        """Generate all possible robot character combinations."""
        difficulties = ['random', 'frequency', 'markov', 'enhanced', 'lstm']
        strategies = ['balanced', 'to_win', 'not_to_lose']
        personalities = ['neutral', 'berserker', 'guardian', 'chameleon', 'professor', 'wildcard', 'mirror']
        
        combinations = []
        for difficulty in difficulties:
            for strategy in strategies:
                for personality in personalities:
                    combinations.append({
                        'difficulty': difficulty,
                        'strategy': strategy,
                        'personality': personality,
                        'name': f"{difficulty.title()} {strategy.replace('_', ' ').title()} {personality.title()}",
                        'id': f"{difficulty}_{strategy}_{personality}"
                    })
        
        return combinations
    
    def create_ai_player(self, robot_config):
        """Create an AI player with the specified configuration."""
        difficulty = robot_config['difficulty']
        strategy = robot_config['strategy']
        personality = robot_config['personality']
        
        # Create the base AI strategy
        if difficulty == 'random':
            base_ai = self.RandomStrategy()
        elif difficulty == 'frequency':
            base_ai = self.FrequencyStrategy()
        elif difficulty == 'markov':
            base_ai = self.MarkovStrategy()
        elif difficulty == 'enhanced':
            base_ai = self.EnhancedStrategy(order=2, recency_weight=0.8)
        elif difficulty == 'lstm':
            if self.LSTM_AVAILABLE:
                # Use LSTM if available
                try:
                    from lstm_web_integration import get_lstm_predictor
                    base_ai = get_lstm_predictor()
                except:
                    base_ai = self.EnhancedStrategy(order=3, recency_weight=0.9)
            else:
                # Fallback to enhanced strategy
                base_ai = self.EnhancedStrategy(order=3, recency_weight=0.9)
        else:
            base_ai = self.RandomStrategy()
        
        # Get personality engine
        try:
            personality_engine_instance = self.get_personality_engine()
            # The personality engine should have methods to apply personality effects
            personality_engine = {
                'modify_move': lambda move, **kwargs: self._apply_personality_via_engine(
                    move, personality, personality_engine_instance, **kwargs
                )
            }
        except:
            # Fallback personality engine
            personality_engine = {'modify_move': lambda move, **kwargs: move}
        
        return AIPlayer(base_ai, strategy, personality, personality_engine)
    
    def _apply_personality_via_engine(self, move, personality, engine, **kwargs):
        """Apply personality effects using the personality engine."""
        try:
            # Try to use the actual personality engine
            if hasattr(engine, 'apply_personality'):
                return engine.apply_personality(move, personality, **kwargs)
            elif hasattr(engine, 'get_personality_modifier'):
                modifier = engine.get_personality_modifier(personality)
                return modifier.get('move_override', move)
            else:
                return move
        except:
            return move
    
    def simulate_game(self, sequence, robot_config):
        """Simulate a complete game between human sequence and robot."""
        # Create AI player
        ai_player = self.create_ai_player(robot_config)
        
        # Game state tracking
        human_moves = []
        robot_moves = []
        results = []
        game_stats = {
            'human_wins': 0,
            'robot_wins': 0,
            'ties': 0,
            'total_moves': 0
        }
        
        # Simulate each move in the sequence
        for i, human_move in enumerate(sequence):
            # Get robot's move
            robot_move = ai_player.get_move(human_moves.copy(), robot_moves.copy(), results.copy())
            
            # Ensure robot_move is a string
            if isinstance(robot_move, tuple):
                robot_move = robot_move[0] if robot_move else 'paper'
            elif not isinstance(robot_move, str):
                robot_move = str(robot_move)
            
            # Determine winner
            result = self.determine_winner(human_move, robot_move)
            
            # Update game state
            human_moves.append(human_move)
            robot_moves.append(robot_move)
            results.append(result)
            
            # Update stats
            if result == 'human':
                game_stats['human_wins'] += 1
            elif result == 'robot':
                game_stats['robot_wins'] += 1
            else:
                game_stats['ties'] += 1
            game_stats['total_moves'] += 1
        
        # Calculate detailed metrics
        metrics = self.calculate_metrics(game_stats, robot_moves, results)
        
        return {
            'robot_config': robot_config,
            'game_stats': game_stats,
            'metrics': metrics,
            'human_moves': human_moves,
            'robot_moves': robot_moves,
            'results': results
        }
    
    def determine_winner(self, human_move, robot_move):
        """Determine the winner of a single move."""
        if human_move == robot_move:
            return 'tie'
        
        winning_combinations = {
            'paper': 'stone',      # Paper beats stone
            'stone': 'scissor',    # Stone beats scissor
            'scissor': 'paper'     # Scissor beats paper
        }
        
        if winning_combinations.get(human_move) == robot_move:
            return 'human'
        else:
            return 'robot'
    
    def calculate_metrics(self, game_stats, robot_moves, results):
        """Calculate detailed performance metrics."""
        total_moves = game_stats['total_moves']
        
        if total_moves == 0:
            return {
                'human_win_rate': 0,
                'robot_win_rate': 0,
                'tie_rate': 0,
                'entropy': 0,
                'predictability': 0,
                'adaptation_rate': 0,
                'move_distribution': {'paper': 0, 'stone': 0, 'scissor': 0}
            }
        
        # Win rates
        human_win_rate = (game_stats['human_wins'] / total_moves) * 100
        robot_win_rate = (game_stats['robot_wins'] / total_moves) * 100
        tie_rate = (game_stats['ties'] / total_moves) * 100
        
        # Move distribution and entropy
        move_counts = Counter(robot_moves)
        total_robot_moves = len(robot_moves)
        
        # Calculate entropy
        entropy = 0
        if total_robot_moves > 0:
            for move in ['paper', 'stone', 'scissor']:
                count = move_counts.get(move, 0)
                if count > 0:
                    p = count / total_robot_moves
                    import math
                    entropy -= p * math.log2(p)
        
        # Calculate predictability (pattern repetition)
        predictability = 0
        if len(robot_moves) >= 3:
            patterns = []
            for i in range(len(robot_moves) - 2):
                pattern = ''.join(robot_moves[i:i+3])
                patterns.append(pattern)
            
            if patterns:
                pattern_counts = Counter(patterns)
                max_pattern_count = max(pattern_counts.values())
                predictability = (max_pattern_count / len(patterns)) * 100
        
        # Calculate adaptation rate (performance change over time)
        adaptation_rate = 0
        if len(results) >= 10:
            mid_point = len(results) // 2
            first_half = results[:mid_point]
            second_half = results[mid_point:]
            
            first_half_robot_rate = (first_half.count('robot') / len(first_half)) * 100
            second_half_robot_rate = (second_half.count('robot') / len(second_half)) * 100
            
            adaptation_rate = abs(second_half_robot_rate - first_half_robot_rate)
        
        # Move distribution percentages
        move_distribution = {}
        for move in ['paper', 'stone', 'scissor']:
            count = move_counts.get(move, 0)
            move_distribution[move] = (count / total_robot_moves) * 100 if total_robot_moves > 0 else 0
        
        return {
            'human_win_rate': human_win_rate,
            'robot_win_rate': robot_win_rate,
            'tie_rate': tie_rate,
            'entropy': entropy,
            'predictability': predictability,
            'adaptation_rate': adaptation_rate,
            'move_distribution': move_distribution
        }
    
    def run_analysis(self):
        """Run the complete distinctiveness analysis."""
        print("🤖 Starting Robot Distinctiveness Analysis")
        print("=" * 60)
        print(f"🎯 Testing {len(self.robot_combinations)} robot combinations")
        print(f"📊 Using optimal sequences against all AI difficulty/strategy/personality combinations")
        print()
        
        # Test both sequence lengths
        for sequence_key in ['25_moves', '50_moves']:
            sequence_data = self.sequences[sequence_key]
            sequence = sequence_data['sequence']
            sequence_name = sequence_data['name']
            sequence_length = len(sequence)
            
            print(f"\n🎯 Testing {sequence_length}-move sequence: {sequence_name}")
            print(f"📈 Expected human win rate: {sequence_data['avg_win_rate']:.1f}%")
            print(f"🤖 Testing against {len(self.robot_combinations)} robot combinations...")
            
            sequence_results = []
            
            # Test each robot combination
            for i, robot_config in enumerate(self.robot_combinations):
                # Progress indicator
                if (i + 1) % 21 == 0:
                    print(f"📊 Progress: {i + 1}/{len(self.robot_combinations)} combinations tested")
                
                # Simulate the game
                result = self.simulate_game(sequence, robot_config)
                result['sequence_length'] = sequence_length
                result['sequence_name'] = sequence_name
                
                sequence_results.append(result)
                self.simulation_results.append(result)
            
            print(f"✅ Completed {sequence_length}-move sequence testing")
            
            # Analyze results for this sequence
            self.analyze_sequence_results(sequence_results, sequence_length)
        
        # Generate comprehensive analysis
        self.generate_comprehensive_analysis()
        
        # Save results
        self.save_results()
        
        print("\n🎉 Robot Distinctiveness Analysis Complete!")
    
    def analyze_sequence_results(self, sequence_results, sequence_length):
        """Analyze results for a specific sequence length."""
        print(f"\n📊 ANALYSIS FOR {sequence_length}-MOVE SEQUENCE")
        print("=" * 50)
        
        # Group results by components
        by_difficulty = defaultdict(list)
        by_strategy = defaultdict(list)
        by_personality = defaultdict(list)
        
        for result in sequence_results:
            config = result['robot_config']
            metrics = result['metrics']
            
            by_difficulty[config['difficulty']].append(metrics)
            by_strategy[config['strategy']].append(metrics)
            by_personality[config['personality']].append(metrics)
        
        # Analyze difficulty distinctiveness
        print("\n🎯 DIFFICULTY LEVEL PERFORMANCE:")
        difficulty_order = ['random', 'frequency', 'markov', 'enhanced', 'lstm']
        for difficulty in difficulty_order:
            if difficulty in by_difficulty:
                metrics_list = by_difficulty[difficulty]
                win_rates = [m['human_win_rate'] for m in metrics_list]
                entropies = [m['entropy'] for m in metrics_list]
                
                avg_win_rate = statistics.mean(win_rates)
                avg_entropy = statistics.mean(entropies)
                
                strength = (
                    "🔴 Very Strong" if avg_win_rate < 30 else
                    "🟡 Strong" if avg_win_rate < 40 else
                    "🟢 Moderate" if avg_win_rate < 50 else
                    "🔵 Weak"
                )
                
                print(f"  {difficulty.upper()}: {strength}")
                print(f"    Human Win Rate: {avg_win_rate:.1f}% | Entropy: {avg_entropy:.2f}")
        
        # Analyze strategy impact
        print("\n⚔️ STRATEGY IMPACT:")
        for strategy, metrics_list in by_strategy.items():
            win_rates = [m['human_win_rate'] for m in metrics_list]
            adaptation_rates = [m['adaptation_rate'] for m in metrics_list]
            
            avg_win_rate = statistics.mean(win_rates)
            avg_adaptation = statistics.mean(adaptation_rates)
            
            print(f"  {strategy.replace('_', ' ').upper()}:")
            print(f"    Avg Human Win Rate: {avg_win_rate:.1f}%")
            print(f"    Avg Adaptation Rate: {avg_adaptation:.1f}%")
        
        # Analyze personality distinctiveness
        print("\n🎭 PERSONALITY DISTINCTIVENESS:")
        personality_stats = []
        for personality, metrics_list in by_personality.items():
            win_rates = [m['human_win_rate'] for m in metrics_list]
            predictabilities = [m['predictability'] for m in metrics_list]
            
            avg_win_rate = statistics.mean(win_rates)
            variance = statistics.variance(win_rates) if len(win_rates) > 1 else 0
            avg_predictability = statistics.mean(predictabilities)
            
            personality_stats.append({
                'name': personality,
                'avg_win_rate': avg_win_rate,
                'variance': variance,
                'avg_predictability': avg_predictability
            })
        
        # Sort by distinctiveness (variance)
        personality_stats.sort(key=lambda x: x['variance'], reverse=True)
        
        for i, stats in enumerate(personality_stats):
            distinctiveness = (
                "⭐ Highly Distinctive" if stats['variance'] > 50 else
                "⭐ Moderately Distinctive" if stats['variance'] > 25 else
                "⚪ Low Distinctiveness"
            )
            
            print(f"  {i+1}. {stats['name'].upper()}: {distinctiveness}")
            print(f"     Performance: {stats['avg_win_rate']:.1f}% | Variance: {stats['variance']:.1f}")
    
    def generate_comprehensive_analysis(self):
        """Generate comprehensive analysis across all sequences."""
        print("\n📋 COMPREHENSIVE DISTINCTIVENESS ASSESSMENT")
        print("=" * 60)
        
        # Overall statistics
        all_metrics = [result['metrics'] for result in self.simulation_results]
        all_win_rates = [m['human_win_rate'] for m in all_metrics]
        all_entropies = [m['entropy'] for m in all_metrics]
        all_predictabilities = [m['predictability'] for m in all_metrics]
        
        print(f"📊 Overall Statistics:")
        print(f"   Total Simulations: {len(self.simulation_results)}")
        print(f"   Robot Combinations: {len(self.robot_combinations)}")
        print(f"   Human Win Rate Range: {min(all_win_rates):.1f}% - {max(all_win_rates):.1f}%")
        print(f"   Win Rate Spread: {max(all_win_rates) - min(all_win_rates):.1f}%")
        print(f"   Average Entropy: {statistics.mean(all_entropies):.2f}")
        print(f"   Entropy Range: {min(all_entropies):.2f} - {max(all_entropies):.2f}")
        
        # Overall assessment
        win_rate_spread = max(all_win_rates) - min(all_win_rates)
        entropy_spread = max(all_entropies) - min(all_entropies)
        
        if win_rate_spread > 30 and entropy_spread > 0.5:
            assessment = "🌟 EXCELLENT - High distinctiveness across all dimensions"
        elif win_rate_spread > 20 and entropy_spread > 0.3:
            assessment = "✅ GOOD - Clear differences between robot types"
        elif win_rate_spread > 10:
            assessment = "⚠️ MODERATE - Some differences but could be more distinctive"
        else:
            assessment = "❌ LOW - Robots are too similar, need more differentiation"
        
        print(f"\n🎯 OVERALL ASSESSMENT: {assessment}")
        
        # Find most and least distinctive robots
        print("\n🏆 MOST DISTINCTIVE ROBOTS:")
        robot_scores = []
        for result in self.simulation_results:
            metrics = result['metrics']
            config = result['robot_config']
            
            # Calculate distinctiveness score
            entropy_score = metrics['entropy'] * 25
            adaptation_score = metrics['adaptation_rate']
            uniqueness_score = abs(metrics['human_win_rate'] - 50)
            
            total_score = entropy_score + adaptation_score + uniqueness_score
            
            robot_scores.append({
                'config': config,
                'metrics': metrics,
                'score': total_score
            })
        
        robot_scores.sort(key=lambda x: x['score'], reverse=True)
        
        for i in range(min(5, len(robot_scores))):
            robot = robot_scores[i]
            print(f"  {i+1}. {robot['config']['name']}")
            print(f"     Distinctiveness Score: {robot['score']:.1f}")
            print(f"     Human Win Rate: {robot['metrics']['human_win_rate']:.1f}%")
            print(f"     Entropy: {robot['metrics']['entropy']:.2f}")
    
    def save_results(self):
        """Save detailed results to JSON file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Prepare summary data
        all_metrics = [result['metrics'] for result in self.simulation_results]
        all_win_rates = [m['human_win_rate'] for m in all_metrics]
        all_entropies = [m['entropy'] for m in all_metrics]
        
        results_data = {
            'analysis_info': {
                'timestamp': timestamp,
                'total_simulations': len(self.simulation_results),
                'robot_combinations': len(self.robot_combinations),
                'sequences_tested': list(self.sequences.keys())
            },
            'summary_statistics': {
                'win_rate_range': [min(all_win_rates), max(all_win_rates)],
                'win_rate_spread': max(all_win_rates) - min(all_win_rates),
                'average_entropy': statistics.mean(all_entropies),
                'entropy_range': [min(all_entropies), max(all_entropies)]
            },
            'sequences_used': self.sequences,
            'detailed_results': self.simulation_results
        }
        
        filename = f'robot_distinctiveness_results_{timestamp}.json'
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {filename}")


class AIPlayer:
    """Wrapper class for AI players with strategy and personality."""
    
    def __init__(self, base_strategy, game_strategy, personality, personality_engine):
        self.base_strategy = base_strategy
        self.game_strategy = game_strategy
        self.personality = personality
        self.personality_engine = personality_engine
        self.move_history = []
    
    def get_move(self, human_moves, robot_moves, results):
        """Get the next move for this AI player."""
        # Get base prediction from strategy
        try:
            base_move = self.base_strategy.predict(human_moves)
        except:
            # Fallback if strategy doesn't work
            base_move = random.choice(['paper', 'stone', 'scissor'])
        
        # Apply game strategy modifications
        if self.game_strategy == 'to_win' and len(results) >= 3:
            recent_results = results[-3:]
            robot_wins = recent_results.count('robot')
            if robot_wins < 2:  # If not winning enough, be more aggressive
                # Try to counter what human played most recently
                if human_moves:
                    counters = {'paper': 'scissor', 'stone': 'paper', 'scissor': 'stone'}
                    base_move = counters.get(human_moves[-1], base_move)
        
        elif self.game_strategy == 'not_to_lose' and len(results) >= 3:
            recent_results = results[-3:]
            robot_wins = recent_results.count('robot')
            if robot_wins >= 2:  # If winning, be more random/defensive
                base_move = random.choice(['paper', 'stone', 'scissor'])
        
        # Apply personality modifications
        try:
            final_move = self.personality_engine['modify_move'](
                base_move,
                human_history=human_moves,
                robot_history=robot_moves,
                game_state={'results': results}
            )
        except:
            # Fallback personality modifications
            final_move = self._apply_simple_personality(base_move, human_moves, results)
        
        self.move_history.append(final_move)
        return final_move
    
    def _apply_simple_personality(self, base_move, human_moves, results):
        """Apply simplified personality effects."""
        if self.personality == 'berserker':
            # More aggressive, favor stone
            if random.random() < 0.3:
                return 'stone'
        elif self.personality == 'guardian':
            # More defensive, favor paper
            if random.random() < 0.3:
                return 'paper'
        elif self.personality == 'chameleon':
            # Adaptive, copy human patterns
            if human_moves and random.random() < 0.4:
                return human_moves[-1]
        elif self.personality == 'wildcard':
            # Unpredictable
            if random.random() < 0.2:
                return random.choice(['paper', 'stone', 'scissor'])
        elif self.personality == 'mirror':
            # Mirror with delay
            if len(human_moves) >= 2 and random.random() < 0.3:
                return human_moves[-2]
        
        return base_move


def main():
    """Main function to run robot distinctiveness analysis."""
    print("🤖 Robot Distinctiveness Analysis")
    print("Testing whether your 105 AI combinations create truly distinctive behaviors")
    print("=" * 70)
    
    simulator = RobotDistinctivenessSimulator()
    simulator.run_analysis()


if __name__ == '__main__':
    main()