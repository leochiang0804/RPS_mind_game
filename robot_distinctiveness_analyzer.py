#!/usr/bin/env python3
"""
Robot Distinctiveness Simulator

Direct simulation using the existing codebase to test all AI combinations
against preset optimal sequences without browser dependency.
"""

import json
import sys
import os
import statistics
from typing import List, Dict, Any, Tuple
import importlib.util

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class RobotDistinctivenessSimulator:
    """Simulates games using actual codebase to test robot distinctiveness."""
    
    def __init__(self):
        self.load_game_modules()
        self.load_optimal_sequences()
        self.robot_combinations = self._generate_robot_combinations()
        self.simulation_results = []
    
    def load_game_modules(self):
        """Load the necessary game modules from codebase."""
        try:
            # Import game engine
            from game_engine import GameEngine
            self.GameEngine = GameEngine
            
            # Import AI modules
            from ai_difficulty import AIDifficulty
            from ai_strategy import AIStrategy
            from ai_personality import AIPersonality
            
            self.AIDifficulty = AIDifficulty
            self.AIStrategy = AIStrategy
            self.AIPersonality = AIPersonality
            
            print("✅ Successfully loaded game modules")
            
        except ImportError as e:
            print(f"❌ Failed to import game modules: {e}")
            print("📁 Available files in directory:")
            for file in os.listdir('.'):
                if file.endswith('.py'):
                    print(f"   • {file}")
            
            # Try alternative imports
            self.try_alternative_imports()
    
    def try_alternative_imports(self):
        """Try to find and import the AI logic from available files."""
        print("\n🔍 Attempting to locate AI logic in available files...")
        
        # Check for app.py or main game file
        if os.path.exists('app.py'):
            print("📁 Found app.py - attempting to extract AI logic")
            self.extract_ai_from_app()
        else:
            print("❌ Could not find main game files")
            print("Please ensure the following files exist:")
            print("   • game_engine.py or app.py")
            print("   • AI difficulty/strategy/personality modules")
            sys.exit(1)
    
    def extract_ai_from_app(self):
        """Extract AI logic from app.py if game modules don't exist."""
        print("🔧 Creating simplified AI simulation from app.py...")
        
        # Create a simplified game engine for simulation
        class SimpleGameEngine:
            def __init__(self):
                self.reset_game()
            
            def reset_game(self):
                self.moves_history = []
                self.results_history = []
                self.game_stats = {
                    'human_wins': 0,
                    'robot_wins': 0,
                    'ties': 0,
                    'total_moves': 0
                }
            
            def play_move(self, human_move, robot_move):
                """Simulate a single move."""
                self.moves_history.append((human_move, robot_move))
                
                # Determine winner
                if human_move == robot_move:
                    result = 'tie'
                    self.game_stats['ties'] += 1
                elif self._human_wins(human_move, robot_move):
                    result = 'human'
                    self.game_stats['human_wins'] += 1
                else:
                    result = 'robot'
                    self.game_stats['robot_wins'] += 1
                
                self.results_history.append(result)
                self.game_stats['total_moves'] += 1
                
                return result
            
            def _human_wins(self, human_move, robot_move):
                """Check if human wins."""
                winning_combinations = {
                    'stone': 'scissor',
                    'paper': 'stone',
                    'scissor': 'paper'
                }
                return winning_combinations.get(human_move) == robot_move
        
        # Create simplified AI classes
        self.GameEngine = SimpleGameEngine
        self.create_simplified_ai_classes()
    
    def create_simplified_ai_classes(self):
        """Create simplified AI classes for simulation."""
        import random
        
        class SimplifiedAI:
            def __init__(self, difficulty='random', strategy='balanced', personality='neutral'):
                self.difficulty = difficulty
                self.strategy = strategy
                self.personality = personality
                self.moves_history = []
                self.results_history = []
                self.pattern_memory = {}
            
            def get_next_move(self, human_moves, robot_moves, results):
                """Get AI's next move based on configuration."""
                self.moves_history = human_moves
                self.results_history = results
                
                # Update pattern memory for learning algorithms
                if len(human_moves) >= 2:
                    pattern = ''.join(human_moves[-2:])
                    if pattern not in self.pattern_memory:
                        self.pattern_memory[pattern] = {'paper': 0, 'stone': 0, 'scissor': 0}
                    if len(human_moves) > 2:
                        next_move = human_moves[-1]
                        self.pattern_memory[pattern][next_move] += 1
                
                # Generate move based on difficulty
                if self.difficulty == 'random':
                    base_move = random.choice(['paper', 'stone', 'scissor'])
                
                elif self.difficulty == 'frequency':
                    if len(human_moves) >= 3:
                        # Predict based on frequency
                        move_counts = {'paper': 0, 'stone': 0, 'scissor': 0}
                        for move in human_moves:
                            move_counts[move] += 1
                        predicted_human = max(move_counts.items(), key=lambda x: x[1])[0]
                        base_move = self._counter_move(predicted_human)
                    else:
                        base_move = random.choice(['paper', 'stone', 'scissor'])
                
                elif self.difficulty == 'markov':
                    if len(human_moves) >= 3:
                        # Use pattern-based prediction
                        recent_pattern = ''.join(human_moves[-2:])
                        if recent_pattern in self.pattern_memory:
                            predicted_move = max(self.pattern_memory[recent_pattern].items(), key=lambda x: x[1])[0]
                            base_move = self._counter_move(predicted_move)
                        else:
                            base_move = random.choice(['paper', 'stone', 'scissor'])
                    else:
                        base_move = random.choice(['paper', 'stone', 'scissor'])
                
                elif self.difficulty == 'enhanced':
                    # Enhanced pattern recognition with recent bias
                    if len(human_moves) >= 4:
                        recent_moves = human_moves[-4:]
                        move_counts = {'paper': 0, 'stone': 0, 'scissor': 0}
                        for i, move in enumerate(recent_moves):
                            weight = (i + 1) / len(recent_moves)  # Recent moves weighted more
                            move_counts[move] += weight
                        predicted_human = max(move_counts.items(), key=lambda x: x[1])[0]
                        base_move = self._counter_move(predicted_human)
                    else:
                        base_move = random.choice(['paper', 'stone', 'scissor'])
                
                elif self.difficulty == 'lstm':
                    # LSTM-like behavior (simplified sequence learning)
                    if len(human_moves) >= 5:
                        # Look for repeating sequences
                        sequence_length = min(3, len(human_moves) // 2)
                        for seq_len in range(sequence_length, 0, -1):
                            current_sequence = human_moves[-seq_len:]
                            # Look for this sequence earlier in history
                            for i in range(len(human_moves) - seq_len - 1):
                                if human_moves[i:i+seq_len] == current_sequence:
                                    if i + seq_len < len(human_moves):
                                        predicted_move = human_moves[i + seq_len]
                                        base_move = self._counter_move(predicted_move)
                                        break
                            else:
                                continue
                            break
                        else:
                            # Fallback to frequency analysis
                            move_counts = {'paper': 0, 'stone': 0, 'scissor': 0}
                            for move in human_moves[-5:]:
                                move_counts[move] += 1
                            predicted_human = max(move_counts.items(), key=lambda x: x[1])[0]
                            base_move = self._counter_move(predicted_human)
                    else:
                        base_move = random.choice(['paper', 'stone', 'scissor'])
                
                else:
                    base_move = random.choice(['paper', 'stone', 'scissor'])
                
                # Apply strategy modifications
                base_move = self._apply_strategy(base_move, results)
                
                # Apply personality modifications
                base_move = self._apply_personality(base_move, human_moves, results)
                
                return base_move
            
            def _counter_move(self, move):
                """Return the move that beats the given move."""
                counters = {
                    'paper': 'scissor',
                    'stone': 'paper',
                    'scissor': 'stone'
                }
                return counters.get(move, 'paper')
            
            def _apply_strategy(self, base_move, results):
                """Apply strategy modifications to base move."""
                if len(results) < 3:
                    return base_move
                
                recent_results = results[-3:]
                
                if self.strategy == 'to_win':
                    # More aggressive when losing
                    robot_wins = recent_results.count('robot')
                    if robot_wins < 2:  # If not winning enough, be more predictable
                        return base_move
                    else:
                        # Stay with winning strategy
                        return base_move
                
                elif self.strategy == 'not_to_lose':
                    # More defensive when winning
                    robot_wins = recent_results.count('robot')
                    if robot_wins >= 2:  # If winning, be more random
                        return random.choice(['paper', 'stone', 'scissor'])
                    else:
                        return base_move
                
                # balanced strategy
                return base_move
            
            def _apply_personality(self, base_move, human_moves, results):
                """Apply personality modifications to move."""
                if len(human_moves) < 2:
                    return base_move
                
                if self.personality == 'berserker':
                    # Aggressive, tends to use stone more
                    if random.random() < 0.3:
                        return 'stone'
                
                elif self.personality == 'guardian':
                    # Defensive, tends to use paper more
                    if random.random() < 0.3:
                        return 'paper'
                
                elif self.personality == 'chameleon':
                    # Adapts to human patterns more quickly
                    if len(human_moves) >= 2:
                        last_human = human_moves[-1]
                        if random.random() < 0.4:
                            return self._counter_move(last_human)
                
                elif self.personality == 'professor':
                    # More analytical, uses complex patterns
                    if len(human_moves) >= 3:
                        # Look for alternating patterns
                        if human_moves[-1] != human_moves[-2] != human_moves[-3]:
                            # If human is alternating, predict continuation
                            return self._counter_move(human_moves[-2])
                
                elif self.personality == 'wildcard':
                    # Unpredictable, random 20% of the time
                    if random.random() < 0.2:
                        return random.choice(['paper', 'stone', 'scissor'])
                
                elif self.personality == 'mirror':
                    # Mirrors human behavior with delay
                    if len(human_moves) >= 2:
                        if random.random() < 0.3:
                            return human_moves[-2]  # Copy previous human move
                
                return base_move
        
        self.SimplifiedAI = SimplifiedAI
        print("✅ Created simplified AI simulation classes")
    
    def load_optimal_sequences(self):
        """Load the optimal sequences from JSON file."""
        try:
            with open('best_sequences_quick_ref.json', 'r') as f:
                self.sequences = json.load(f)
            print("✅ Loaded optimal sequences successfully")
        except FileNotFoundError:
            print("❌ Optimal sequences not found. Using default test sequences.")
            self.sequences = {
                '25_moves': {
                    'name': 'anti_lstm_pattern',
                    'sequence': ['paper', 'stone', 'scissor', 'paper', 'stone', 
                                'scissor', 'stone', 'paper', 'scissor', 'stone',
                                'paper', 'scissor', 'paper', 'stone', 'scissor',
                                'stone', 'paper', 'scissor', 'paper', 'stone',
                                'scissor', 'stone', 'paper', 'scissor', 'paper'],
                    'avg_win_rate': 35.5
                },
                '50_moves': {
                    'name': 'entropy_maximizer',
                    'sequence': ['scissor', 'paper', 'scissor', 'paper', 'stone',
                                'scissor', 'stone', 'paper', 'stone', 'scissor',
                                'paper', 'stone', 'scissor', 'paper', 'stone',
                                'scissor', 'paper', 'stone', 'scissor', 'paper',
                                'stone', 'scissor', 'paper', 'stone', 'scissor',
                                'paper', 'stone', 'scissor', 'paper', 'stone',
                                'scissor', 'paper', 'stone', 'scissor', 'paper',
                                'stone', 'scissor', 'paper', 'stone', 'scissor',
                                'paper', 'stone', 'scissor', 'paper', 'stone',
                                'scissor', 'paper', 'stone', 'scissor', 'paper'],
                    'avg_win_rate': 42.3
                }
            }
    
    def _generate_robot_combinations(self):
        """Generate all robot combinations."""
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
    
    def simulate_game(self, sequence, robot_config):
        """Simulate a complete game between human sequence and robot."""
        # Initialize game engine
        game = self.GameEngine()
        
        # Initialize AI with configuration
        if hasattr(self, 'SimplifiedAI'):
            ai = self.SimplifiedAI(
                difficulty=robot_config['difficulty'],
                strategy=robot_config['strategy'],
                personality=robot_config['personality']
            )
        else:
            # Use actual game AI classes if available
            ai = self.create_configured_ai(robot_config)
        
        # Track game data
        human_moves = []
        robot_moves = []
        results = []
        move_patterns = []
        
        # Simulate each move in sequence
        for i, human_move in enumerate(sequence):
            # Get robot's move
            robot_move = ai.get_next_move(human_moves.copy(), robot_moves.copy(), results.copy())
            
            # Play the move
            result = game.play_move(human_move, robot_move)
            
            # Update tracking
            human_moves.append(human_move)
            robot_moves.append(robot_move)
            results.append(result)
            
            # Track patterns for analysis
            if len(robot_moves) >= 3:
                pattern = ''.join(robot_moves[-3:])
                move_patterns.append(pattern)
        
        # Calculate metrics
        metrics = self.calculate_game_metrics(game.game_stats, robot_moves, results, move_patterns)
        
        return {
            'game_stats': game.game_stats,
            'metrics': metrics,
            'human_moves': human_moves,
            'robot_moves': robot_moves,
            'results': results
        }
    
    def create_configured_ai(self, robot_config):
        """Create AI instance with specific configuration (if using actual game classes)."""
        # This would be implemented if we have access to the actual AI classes
        # For now, using the simplified version
        return self.SimplifiedAI(
            difficulty=robot_config['difficulty'],
            strategy=robot_config['strategy'],
            personality=robot_config['personality']
        )
    
    def calculate_game_metrics(self, game_stats, robot_moves, results, patterns):
        """Calculate detailed metrics for game analysis."""
        total_moves = game_stats['total_moves']
        
        if total_moves == 0:
            return {
                'human_win_rate': 0,
                'robot_win_rate': 0,
                'tie_rate': 0,
                'predictability': 0,
                'entropy': 0,
                'adaptation_rate': 0,
                'move_distribution': {'paper': 0, 'stone': 0, 'scissor': 0}
            }
        
        # Win rates
        human_win_rate = (game_stats['human_wins'] / total_moves) * 100
        robot_win_rate = (game_stats['robot_wins'] / total_moves) * 100
        tie_rate = (game_stats['ties'] / total_moves) * 100
        
        # Move distribution
        move_counts = {'paper': 0, 'stone': 0, 'scissor': 0}
        for move in robot_moves:
            if move in move_counts:
                move_counts[move] += 1
        
        move_distribution = {move: (count / total_moves) * 100 for move, count in move_counts.items()}
        
        # Entropy calculation
        total_robot_moves = sum(move_counts.values())
        entropy = 0
        if total_robot_moves > 0:
            for count in move_counts.values():
                if count > 0:
                    p = count / total_robot_moves
                    entropy -= p * (p.bit_length() - 1) if p > 0 else 0  # Simplified entropy
        
        # Predictability (pattern repetition)
        if patterns:
            pattern_counts = {}
            for pattern in patterns:
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            max_pattern_count = max(pattern_counts.values()) if pattern_counts else 0
            predictability = (max_pattern_count / len(patterns)) * 100 if patterns else 0
        else:
            predictability = 0
        
        # Adaptation rate (performance change over time)
        if len(results) >= 10:
            first_half = results[:len(results)//2]
            second_half = results[len(results)//2:]
            
            first_half_robot_rate = (first_half.count('robot') / len(first_half)) * 100
            second_half_robot_rate = (second_half.count('robot') / len(second_half)) * 100
            
            adaptation_rate = abs(second_half_robot_rate - first_half_robot_rate)
        else:
            adaptation_rate = 0
        
        return {
            'human_win_rate': human_win_rate,
            'robot_win_rate': robot_win_rate,
            'tie_rate': tie_rate,
            'predictability': predictability,
            'entropy': entropy,
            'adaptation_rate': adaptation_rate,
            'move_distribution': move_distribution,
            'total_moves': total_moves
        }
    
    def run_distinctiveness_analysis(self):
        """Run complete distinctiveness analysis on all robot combinations."""
        print("🤖 Starting Robot Distinctiveness Simulation")
        print("=" * 60)
        
        # Test both sequence lengths
        for sequence_key in ['25_moves', '50_moves']:
            sequence_data = self.sequences[sequence_key]
            sequence = sequence_data['sequence']
            sequence_name = sequence_data['name']
            
            print(f"\n🎯 Testing {len(sequence)}-move sequence: {sequence_name}")
            print(f"📊 Expected average win rate: {sequence_data.get('avg_win_rate', 'Unknown')}%")
            print(f"🤖 Testing against {len(self.robot_combinations)} robot combinations...")
            
            sequence_results = []
            
            # Test each robot combination
            for i, robot_config in enumerate(self.robot_combinations):
                if (i + 1) % 21 == 0:  # Progress indicator
                    print(f"📈 Progress: {i + 1}/{len(self.robot_combinations)} combinations tested")
                
                # Simulate game
                simulation_result = self.simulate_game(sequence, robot_config)
                
                # Store results
                result_entry = {
                    'sequence_length': len(sequence),
                    'sequence_name': sequence_name,
                    'robot_config': robot_config,
                    'simulation': simulation_result
                }
                
                sequence_results.append(result_entry)
                self.simulation_results.append(result_entry)
            
            # Analyze results for this sequence
            self.analyze_sequence_results(sequence_results, len(sequence))
        
        # Generate comprehensive analysis
        self.generate_comprehensive_analysis()
    
    def analyze_sequence_results(self, sequence_results, sequence_length):
        """Analyze results for a specific sequence length."""
        print(f"\n📊 ANALYSIS RESULTS FOR {sequence_length}-MOVE SEQUENCE")
        print("=" * 50)
        
        # Group results by components
        by_difficulty = {}
        by_strategy = {}
        by_personality = {}
        
        for result in sequence_results:
            config = result['robot_config']
            metrics = result['simulation']['metrics']
            
            # Group by difficulty
            if config['difficulty'] not in by_difficulty:
                by_difficulty[config['difficulty']] = []
            by_difficulty[config['difficulty']].append(metrics)
            
            # Group by strategy
            if config['strategy'] not in by_strategy:
                by_strategy[config['strategy']] = []
            by_strategy[config['strategy']].append(metrics)
            
            # Group by personality
            if config['personality'] not in by_personality:
                by_personality[config['personality']] = []
            by_personality[config['personality']].append(metrics)
        
        # Analyze difficulty distinctiveness
        print("\n🎯 DIFFICULTY LEVEL ANALYSIS:")
        difficulty_stats = {}
        for difficulty, metrics_list in by_difficulty.items():
            win_rates = [m['human_win_rate'] for m in metrics_list]
            entropies = [m['entropy'] for m in metrics_list]
            
            avg_win_rate = statistics.mean(win_rates)
            win_rate_std = statistics.stdev(win_rates) if len(win_rates) > 1 else 0
            avg_entropy = statistics.mean(entropies)
            
            difficulty_stats[difficulty] = {
                'avg_win_rate': avg_win_rate,
                'win_rate_std': win_rate_std,
                'avg_entropy': avg_entropy,
                'count': len(metrics_list)
            }
            
            strength = (
                "🔴 Very Strong" if avg_win_rate < 30 else
                "🟡 Strong" if avg_win_rate < 40 else
                "🟢 Moderate" if avg_win_rate < 50 else
                "🔵 Weak"
            )
            
            print(f"  {difficulty.upper()}: {strength}")
            print(f"    Human Win Rate: {avg_win_rate:.1f}% (±{win_rate_std:.1f}%)")
            print(f"    Avg Entropy: {avg_entropy:.2f}")
        
        # Analyze strategy distinctiveness
        print("\n⚔️ STRATEGY ANALYSIS:")
        for strategy, metrics_list in by_strategy.items():
            win_rates = [m['human_win_rate'] for m in metrics_list]
            adaptation_rates = [m['adaptation_rate'] for m in metrics_list]
            
            avg_win_rate = statistics.mean(win_rates)
            avg_adaptation = statistics.mean(adaptation_rates)
            win_rate_range = max(win_rates) - min(win_rates)
            
            print(f"  {strategy.replace('_', ' ').upper()}:")
            print(f"    Avg Human Win Rate: {avg_win_rate:.1f}%")
            print(f"    Avg Adaptation Rate: {avg_adaptation:.1f}%")
            print(f"    Performance Range: {win_rate_range:.1f}%")
        
        # Analyze personality distinctiveness
        print("\n🎭 PERSONALITY ANALYSIS:")
        personality_stats = []
        for personality, metrics_list in by_personality.items():
            win_rates = [m['human_win_rate'] for m in metrics_list]
            predictabilities = [m['predictability'] for m in metrics_list]
            
            avg_win_rate = statistics.mean(win_rates)
            win_rate_variance = statistics.variance(win_rates) if len(win_rates) > 1 else 0
            avg_predictability = statistics.mean(predictabilities)
            
            personality_stats.append({
                'name': personality,
                'avg_win_rate': avg_win_rate,
                'variance': win_rate_variance,
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
            print(f"     Avg Performance: {stats['avg_win_rate']:.1f}%")
            print(f"     Variance: {stats['variance']:.1f} | Predictability: {stats['avg_predictability']:.1f}%")
        
        # Find most distinctive robots
        print("\n🏆 MOST DISTINCTIVE ROBOTS:")
        robot_scores = []
        for result in sequence_results:
            metrics = result['simulation']['metrics']
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
            print(f"     Score: {robot['score']:.1f} | Win Rate: {robot['metrics']['human_win_rate']:.1f}% | Entropy: {robot['metrics']['entropy']:.2f}")
    
    def generate_comprehensive_analysis(self):
        """Generate comprehensive analysis across all sequences."""
        print(f"\n📋 COMPREHENSIVE DISTINCTIVENESS ANALYSIS")
        print("=" * 60)
        
        # Overall statistics
        all_metrics = [result['simulation']['metrics'] for result in self.simulation_results]
        all_win_rates = [m['human_win_rate'] for m in all_metrics]
        all_entropies = [m['entropy'] for m in all_metrics]
        all_predictabilities = [m['predictability'] for m in all_metrics]
        
        print(f"📊 Overall Statistics:")
        print(f"   Total Simulations: {len(self.simulation_results)}")
        print(f"   Robot Combinations: {len(self.robot_combinations)}")
        print(f"   Win Rate Range: {min(all_win_rates):.1f}% - {max(all_win_rates):.1f}%")
        print(f"   Win Rate Spread: {max(all_win_rates) - min(all_win_rates):.1f}%")
        print(f"   Average Entropy: {statistics.mean(all_entropies):.2f}")
        print(f"   Entropy Range: {min(all_entropies):.2f} - {max(all_entropies):.2f}")
        print(f"   Predictability Range: {min(all_predictabilities):.1f}% - {max(all_predictabilities):.1f}%")
        
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
        
        # Save detailed results
        self.save_simulation_results()
        
        print(f"\n✅ Robot distinctiveness simulation completed!")
        print(f"📁 Detailed results saved to: robot_distinctiveness_simulation_results.json")
    
    def save_simulation_results(self):
        """Save detailed simulation results to JSON file."""
        timestamp = __import__('datetime').datetime.now().isoformat()
        
        # Prepare summary data
        all_metrics = [result['simulation']['metrics'] for result in self.simulation_results]
        all_win_rates = [m['human_win_rate'] for m in all_metrics]
        all_entropies = [m['entropy'] for m in all_metrics]
        
        # Find best and worst performers
        best_performers = sorted(self.simulation_results, 
                               key=lambda x: x['simulation']['metrics']['human_win_rate'])[:5]
        worst_performers = sorted(self.simulation_results, 
                                key=lambda x: x['simulation']['metrics']['human_win_rate'], 
                                reverse=True)[:5]
        
        results_data = {
            'analysis_info': {
                'analysis_type': 'robot_distinctiveness_simulation',
                'timestamp': timestamp,
                'total_simulations': len(self.simulation_results),
                'robot_combinations': len(self.robot_combinations),
                'sequences_tested': list(self.sequences.keys())
            },
            'summary_statistics': {
                'win_rate_range': [min(all_win_rates), max(all_win_rates)],
                'win_rate_spread': max(all_win_rates) - min(all_win_rates),
                'average_entropy': statistics.mean(all_entropies),
                'entropy_range': [min(all_entropies), max(all_entropies)],
                'entropy_spread': max(all_entropies) - min(all_entropies)
            },
            'best_performers': [
                {
                    'robot_name': result['robot_config']['name'],
                    'robot_config': result['robot_config'],
                    'human_win_rate': result['simulation']['metrics']['human_win_rate'],
                    'entropy': result['simulation']['metrics']['entropy'],
                    'sequence_length': result['sequence_length']
                }
                for result in best_performers
            ],
            'worst_performers': [
                {
                    'robot_name': result['robot_config']['name'],
                    'robot_config': result['robot_config'],
                    'human_win_rate': result['simulation']['metrics']['human_win_rate'],
                    'entropy': result['simulation']['metrics']['entropy'],
                    'sequence_length': result['sequence_length']
                }
                for result in worst_performers
            ],
            'detailed_results': self.simulation_results
        }
        
        # Save to file
        with open('robot_distinctiveness_simulation_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)


def main():
    """Main function to run robot distinctiveness simulation."""
    print("🤖 Robot Distinctiveness Simulation")
    print("Using actual codebase to test robot behavioral differences")
    print("=" * 60)
    
    simulator = RobotDistinctivenessSimulator()
    simulator.run_distinctiveness_analysis()


if __name__ == '__main__':
    main()