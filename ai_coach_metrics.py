"""
AI Coach Metrics Aggregator
Comprehensive system to collect and prepare all game metrics for AI coach analysis
"""

import math
import time
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, deque
from coach_tips import CoachTipsGenerator
from change_point_detector import ChangePointDetector


class AICoachMetricsAggregator:
    """
    Comprehensive metrics aggregator for AI coach system
    Collects all available game data and formats it for LLM consumption
    """
    
    def __init__(self):
        self.basic_coach = CoachTipsGenerator()
        self.change_detector = ChangePointDetector()
        self.session_start_time = time.time()
        
    def _format_metric(self, value: float, decimals: int = 4) -> float:
        """Format metric to consistent decimal places"""
        if isinstance(value, (int, float)) and not math.isnan(value) and not math.isinf(value):
            return round(float(value), decimals)
        return 0.0
    
    def _calculate_win_rates(self, results: List[str]) -> Dict[str, float]:
        """Calculate win rates from game results"""
        if not results:
            return {'human': 0.0, 'ai': 0.0, 'tie': 0.0}
        
        total = len(results)
        wins = results.count('win')
        losses = results.count('lose') 
        ties = results.count('tie')
        
        return {
            'human': self._format_metric(wins / total, 3),
            'ai': self._format_metric(losses / total, 3),
            'tie': self._format_metric(ties / total, 3)
        }
        
    def aggregate_comprehensive_metrics(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate all available metrics from the game state
        Returns comprehensive context for AI coach analysis
        """
        
        # Core game metrics
        core_metrics = self._extract_core_game_metrics(game_state)
        
        # Pattern analysis metrics
        pattern_metrics = self._extract_pattern_metrics(game_state)
        
        # Performance metrics
        performance_metrics = self._extract_performance_metrics(game_state)
        
        # AI behavior metrics
        ai_metrics = self._extract_ai_behavior_metrics(game_state)
        
        # Temporal metrics
        temporal_metrics = self._extract_temporal_metrics(game_state)
        
        # Advanced analytics
        advanced_metrics = self._extract_advanced_analytics(game_state)
        
        # Psychological indicators
        psychological_metrics = self._extract_psychological_indicators(game_state)
        
        # Strategic context
        strategic_context = self._extract_strategic_context(game_state)
        
        # Model performance tracking
        model_performance = self._extract_model_performance_metrics(game_state)
        
        return {
            'core_game': core_metrics,
            'patterns': pattern_metrics,
            'performance': performance_metrics,
            'ai_behavior': ai_metrics,
            'temporal': temporal_metrics,
            'advanced': advanced_metrics,
            'psychological': psychological_metrics,
            'strategic': strategic_context,
            'model_performance': model_performance,
            'meta': {
                'aggregation_timestamp': time.time(),
                'data_quality_score': self._calculate_data_quality(game_state),
                'confidence_level': self._calculate_confidence_level(game_state),
                'total_metrics_count': self._count_total_metrics()
            }
        }
    
    def _extract_core_game_metrics(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract basic game state metrics"""
        
        # Handle both webapp format (human_moves, robot_moves) and historical format
        human_history = game_state.get('human_moves', game_state.get('human_history', []))
        robot_history = game_state.get('robot_moves', game_state.get('robot_history', []))
        result_history = game_state.get('results', game_state.get('result_history', []))
        
        # Calculate win rates if we have results
        win_rates = self._calculate_win_rates(result_history)
        
        return {
            'current_round': game_state.get('round', len(human_history)),
            'total_moves': len(human_history),
            'human_moves': human_history,
            'robot_moves': robot_history,
            'results': result_history,
            'win_rates': win_rates,
            'recent_moves': {
                'human_last_5': human_history[-5:] if len(human_history) >= 5 else human_history,
                'human_last_10': human_history[-10:] if len(human_history) >= 10 else human_history,
                'robot_last_5': robot_history[-5:] if len(robot_history) >= 5 else robot_history,
                'robot_last_10': robot_history[-10:] if len(robot_history) >= 10 else robot_history,
            },
            'game_settings': {
                'difficulty': game_state.get('current_difficulty', game_state.get('difficulty', 'unknown')),
                'strategy_preference': game_state.get('current_strategy', game_state.get('strategy_preference', 'unknown')),
                'personality': game_state.get('personality', 'unknown'),
                'multiplayer': game_state.get('multiplayer', False)
            },
            'win_rates': {
                'human': self._calculate_win_rate(result_history, 'human'),
                'robot': self._calculate_win_rate(result_history, 'robot'),
                'tie': self._calculate_win_rate(result_history, 'tie')
            }
        }
    
    def _extract_pattern_metrics(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract pattern analysis metrics"""
        
        # Handle both webapp format and historical format
        human_history = game_state.get('human_moves', game_state.get('human_history', []))
        
        if len(human_history) < 2:
            # Provide basic analysis even with minimal data
            return {
                'pattern_type': 'insufficient_data_for_patterns',
                'complexity_score': 0.0,
                'entropy': 0.0,
                'predictability': 1.0,  # With no pattern, next move is unpredictable
                'recent_pattern': 'none',
                'total_moves': len(human_history),
                'move_distribution': self._get_move_distribution(human_history),
                'status': 'need_more_moves'
            }
        
        # Calculate comprehensive pattern analysis
        entropy = self._calculate_entropy(human_history)
        predictability = self._calculate_predictability_score(human_history)
        move_dist = self._get_move_distribution(human_history)
        
        return {
            'pattern_type': self._determine_pattern_type(human_history),
            'complexity_score': self._calculate_pattern_strength(human_history),
            'entropy': entropy,
            'predictability': predictability,
            'recent_pattern': self._analyze_recent_pattern(human_history),
            'total_moves': len(human_history),
            'move_distribution': move_dist,
            'randomness_score': self._calculate_randomness_score(human_history),
            'repetition_count': self._count_repetitions(human_history),
            'alternation_rate': self._calculate_alternation_rate(human_history),
            'status': 'analyzed'
        }
    
    def _get_move_distribution(self, moves: List[str]) -> Dict[str, float]:
        """Calculate distribution of moves"""
        if not moves:
            return {'rock': 0.0, 'paper': 0.0, 'scissors': 0.0}
        
        total = len(moves)
        counter = Counter(moves)
        
        return {
            'rock': self._format_metric(counter.get('rock', 0) / total, 3),
            'paper': self._format_metric(counter.get('paper', 0) / total, 3),
            'scissors': self._format_metric(counter.get('scissors', 0) / total, 3)
        }
    
    def _determine_pattern_type(self, moves: List[str]) -> str:
        """Determine the type of pattern in moves"""
        if len(moves) < 3:
            return 'too_few_moves'
        
        # Check for repetitions
        if len(set(moves)) == 1:
            return 'single_move_repetition'
        
        # Check for alternating pattern
        alternations = sum(1 for i in range(len(moves)-1) if moves[i] != moves[i+1])
        if alternations / (len(moves)-1) > 0.8:
            return 'high_variation'
        elif alternations / (len(moves)-1) < 0.3:
            return 'low_variation'
        else:
            return 'mixed_pattern'
    
    def _analyze_recent_pattern(self, moves: List[str]) -> str:
        """Analyze the most recent pattern"""
        if len(moves) < 3:
            return 'insufficient_data'
        
        recent = moves[-3:]
        if len(set(recent)) == 1:
            return f'repeating_{recent[0]}'
        elif len(set(recent)) == 3:
            return 'all_different'
        else:
            return 'mixed'
    
    def _count_repetitions(self, moves: List[str]) -> int:
        """Count consecutive repetitions"""
        if len(moves) < 2:
            return 0
        
        repetitions = 0
        for i in range(len(moves) - 1):
            if moves[i] == moves[i+1]:
                repetitions += 1
        
        return repetitions
    
    def _calculate_alternation_rate(self, moves: List[str]) -> float:
        """Calculate how often moves alternate"""
        if len(moves) < 2:
            return 0.0
        
        alternations = sum(1 for i in range(len(moves)-1) if moves[i] != moves[i+1])
        return self._format_metric(alternations / (len(moves)-1), 3)
    
    def _calculate_pattern_strength(self, moves: List[str]) -> float:
        """Calculate how strong the patterns are"""
        if len(moves) < 3:
            return 0.0
        
        # Look for consecutive patterns
        patterns = 0
        for i in range(len(moves) - 2):
            if moves[i] == moves[i+1] == moves[i+2]:
                patterns += 1
        
        return self._format_metric(patterns / max(1, len(moves) - 2), 3)
    
    def _calculate_randomness_score(self, moves: List[str]) -> float:
        """Calculate how random the moves appear"""
        if len(moves) < 2:
            return 1.0
        
        # Check for alternating patterns, repetitions, etc.
        alternations = 0
        repetitions = 0
        
        for i in range(len(moves) - 1):
            if moves[i] != moves[i+1]:
                alternations += 1
            else:
                repetitions += 1
        
        # Higher alternation = more randomness
        total_pairs = len(moves) - 1
        randomness = alternations / total_pairs if total_pairs > 0 else 0.5
        
        return self._format_metric(randomness, 3)
    
    def _extract_performance_metrics(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract performance-related metrics"""
        
        result_history = game_state.get('result_history', [])
        
        if not result_history:
            return {'no_results': True}
        
        return {
            'overall_performance': {
                'total_games': len(result_history),
                'win_rate': self._calculate_win_rate(result_history, 'human'),
                'loss_rate': self._calculate_win_rate(result_history, 'robot'),
                'tie_rate': self._calculate_win_rate(result_history, 'tie')
            },
            'recent_performance': {
                'last_5_games': self._analyze_recent_performance(result_history, 5),
                'last_10_games': self._analyze_recent_performance(result_history, 10),
                'trend': self._calculate_performance_trend(result_history)
            },
            'streaks': {
                'current_streak': self._get_current_streak(result_history),
                'longest_win_streak': self._get_longest_streak(result_history, 'human'),
                'longest_loss_streak': self._get_longest_streak(result_history, 'robot')
            },
            'momentum': {
                'momentum_score': self._calculate_momentum(result_history),
                'momentum_direction': self._get_momentum_direction(result_history)
            }
        }
    
    def _extract_ai_behavior_metrics(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract AI behavior and prediction metrics"""
        
        return {
            'model_accuracy': game_state.get('accuracy', {}),
            'prediction_history': game_state.get('model_predictions_history', {}),
            'confidence_history': game_state.get('model_confidence_history', {}),
            'current_strategy': game_state.get('current_strategy', 'unknown'),
            'ai_adaptation': self._analyze_ai_adaptation(game_state),
            'prediction_patterns': self._analyze_prediction_patterns(game_state)
        }
    
    def _extract_temporal_metrics(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract timing and temporal pattern metrics"""
        
        return {
            'session_duration': time.time() - self.session_start_time,
            'game_phase': self._determine_game_phase(game_state.get('round', 0)),
            'rounds_per_minute': self._calculate_rounds_per_minute(game_state),
            'change_point_timing': self._analyze_change_point_timing(game_state.get('change_points', []))
        }
    
    def _extract_advanced_analytics(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract advanced analytical metrics"""
        
        human_history = game_state.get('human_history', [])
        
        if len(human_history) < 5:
            return {'insufficient_data_for_advanced': True}
        
        return {
            'complexity_metrics': {
                'decision_complexity': self._calculate_decision_complexity(human_history),
                'strategy_consistency': self._calculate_strategy_consistency(human_history),
                'adaptation_rate': self._calculate_adaptation_rate(game_state)
            },
            'information_theory': {
                'entropy': self._calculate_entropy(human_history),
                'mutual_information': self._calculate_mutual_information(game_state),
                'compression_ratio': self._calculate_compression_ratio(human_history)
            },
            'game_theory_metrics': {
                'nash_equilibrium_distance': self._calculate_nash_distance(human_history),
                'exploitability': self._calculate_exploitability(game_state),
                'counter_strategy_effectiveness': self._analyze_counter_strategies(game_state)
            }
        }
    
    def _extract_psychological_indicators(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract psychological pattern indicators"""
        
        human_history = game_state.get('human_history', [])
        result_history = game_state.get('result_history', [])
        
        return {
            'decision_making_style': {
                'impulsiveness_indicator': self._assess_impulsiveness(human_history),
                'consistency_score': self._assess_consistency(human_history),
                'risk_tolerance': self._assess_risk_tolerance(human_history, result_history)
            },
            'emotional_indicators': {
                'frustration_signals': self._detect_frustration_patterns(game_state),
                'confidence_patterns': self._analyze_confidence_patterns(game_state),
                'tilt_detection': self._detect_tilt_behavior(game_state)
            },
            'cognitive_patterns': {
                'pattern_awareness': self._assess_pattern_awareness(game_state),
                'meta_cognitive_indicators': self._assess_meta_cognition(game_state),
                'learning_indicators': self._detect_learning_patterns(game_state)
            }
        }
    
    def _extract_strategic_context(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract strategic context and recommendations"""
        
        return {
            'current_strategy_assessment': self._assess_current_strategy(game_state),
            'strategic_opportunities': self._identify_strategic_opportunities(game_state),
            'weaknesses': self._identify_strategic_weaknesses(game_state),
            'adaptation_suggestions': self._generate_adaptation_suggestions(game_state),
            'educational_focus': self._determine_educational_focus(game_state)
        }
    
    # Helper methods for calculations
    
    def _calculate_win_rate(self, results: List[str], outcome: str) -> float:
        """Calculate win rate for specific outcome"""
        if not results:
            return 0.0
        return results.count(outcome) / len(results)
    
    def _calculate_move_distribution(self, moves: List[str]) -> Dict[str, float]:
        """Calculate distribution of moves"""
        if not moves:
            return {'paper': 0.0, 'scissor': 0.0, 'stone': 0.0}
        
        total = len(moves)
        return {
            'paper': moves.count('paper') / total,
            'scissor': moves.count('scissor') / total,
            'stone': moves.count('stone') / total
        }
    
    def _calculate_entropy(self, moves: List[str]) -> float:
        """Calculate Shannon entropy of move sequence"""
        if not moves:
            return 0.0
        
        distribution = self._calculate_move_distribution(moves)
        entropy = 0.0
        
        for prob in distribution.values():
            if prob > 0:
                entropy -= prob * math.log2(prob)
        
        return self._format_metric(entropy, 4)
    
    def _calculate_predictability_score(self, moves: List[str]) -> float:
        """Calculate predictability score using existing coach method"""
        if len(moves) < 3:
            return 0.0
        score = self.basic_coach._calculate_predictability(moves)
        return self._format_metric(score, 4)
    
    def _detect_sequence_patterns(self, moves: List[str]) -> Dict[str, Any]:
        """Detect common sequence patterns"""
        if len(moves) < 3:
            return {'insufficient_data': True}
        
        patterns = {
            'bigrams': self._count_bigrams(moves),
            'trigrams': self._count_trigrams(moves),
            'common_sequences': self._find_common_sequences(moves)
        }
        
        return patterns
    
    def _count_bigrams(self, moves: List[str]) -> Dict[str, int]:
        """Count bigram patterns"""
        bigrams = {}
        for i in range(len(moves) - 1):
            bigram = f"{moves[i]}->{moves[i+1]}"
            bigrams[bigram] = bigrams.get(bigram, 0) + 1
        return bigrams
    
    def _count_trigrams(self, moves: List[str]) -> Dict[str, int]:
        """Count trigram patterns"""
        trigrams = {}
        for i in range(len(moves) - 2):
            trigram = f"{moves[i]}->{moves[i+1]}->{moves[i+2]}"
            trigrams[trigram] = trigrams.get(trigram, 0) + 1
        return trigrams
    
    def _find_common_sequences(self, moves: List[str], min_length: int = 2) -> List[Dict[str, Any]]:
        """Find commonly repeated sequences"""
        sequences = []
        
        for length in range(min_length, min(6, len(moves) // 2)):
            sequence_counts = {}
            
            for i in range(len(moves) - length + 1):
                seq = tuple(moves[i:i+length])
                sequence_counts[seq] = sequence_counts.get(seq, 0) + 1
            
            # Find sequences that appear more than once
            for seq, count in sequence_counts.items():
                if count > 1:
                    sequences.append({
                        'sequence': list(seq),
                        'count': count,
                        'length': length,
                        'frequency': count / (len(moves) - length + 1)
                    })
        
        return sorted(sequences, key=lambda x: x['frequency'], reverse=True)[:10]
    
    def _analyze_recent_performance(self, results: List[str], window: int) -> Dict[str, Any]:
        """Analyze performance in recent window"""
        if len(results) < window:
            recent = results
        else:
            recent = results[-window:]
        
        if not recent:
            return {'no_data': True}
        
        return {
            'win_rate': self._calculate_win_rate(recent, 'human'),
            'games_played': len(recent),
            'wins': recent.count('human'),
            'losses': recent.count('robot'),
            'ties': recent.count('tie')
        }
    
    def _calculate_performance_trend(self, results: List[str]) -> str:
        """Calculate performance trend direction"""
        if len(results) < 10:
            return 'insufficient_data'
        
        # Compare first half vs second half
        mid = len(results) // 2
        first_half_wr = self._calculate_win_rate(results[:mid], 'human')
        second_half_wr = self._calculate_win_rate(results[mid:], 'human')
        
        diff = second_half_wr - first_half_wr
        
        if diff > 0.1:
            return 'improving'
        elif diff < -0.1:
            return 'declining'
        else:
            return 'stable'
    
    def _get_current_streak(self, results: List[str]) -> Dict[str, Any]:
        """Get current win/loss streak"""
        if not results:
            return {'type': 'none', 'length': 0}
        
        current = results[-1]
        length = 1
        
        for i in range(len(results) - 2, -1, -1):
            if results[i] == current:
                length += 1
            else:
                break
        
        return {'type': current, 'length': length}
    
    def _get_longest_streak(self, results: List[str], outcome: str) -> int:
        """Get longest streak of specific outcome"""
        if not results:
            return 0
        
        max_streak = 0
        current_streak = 0
        
        for result in results:
            if result == outcome:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def _calculate_momentum(self, results: List[str]) -> float:
        """Calculate momentum score based on recent results"""
        if len(results) < 5:
            return 0.0
        
        # Weight recent results more heavily
        weights = [0.4, 0.3, 0.2, 0.1]  # Most recent gets highest weight
        recent = results[-4:]
        
        momentum = 0.0
        for i, result in enumerate(recent):
            if result == 'human':
                momentum += weights[-(i+1)]
            elif result == 'robot':
                momentum -= weights[-(i+1)]
            # ties are neutral (0)
        
        return momentum
    
    def _get_momentum_direction(self, results: List[str]) -> str:
        """Get momentum direction"""
        momentum = self._calculate_momentum(results)
        
        if momentum > 0.2:
            return 'positive'
        elif momentum < -0.2:
            return 'negative'
        else:
            return 'neutral'
    
    def _determine_game_phase(self, round_num: int) -> str:
        """Determine current phase of the game"""
        if round_num < 10:
            return 'early'
        elif round_num < 30:
            return 'mid'
        else:
            return 'late'
    
    def _calculate_rounds_per_minute(self, game_state: Dict[str, Any]) -> float:
        """Calculate rounds per minute"""
        duration = time.time() - self.session_start_time
        rounds = game_state.get('round', 0)
        
        if duration < 60:  # Less than a minute
            return 0.0
        
        return rounds / (duration / 60)
    
    def _calculate_data_quality(self, game_state: Dict[str, Any]) -> float:
        """Calculate data quality score for metrics"""
        round_num = game_state.get('round', 0)
        
        # Quality improves with more data
        if round_num < 5:
            return 0.2
        elif round_num < 15:
            return 0.5
        elif round_num < 30:
            return 0.8
        else:
            return 1.0
    
    def _calculate_confidence_level(self, game_state: Dict[str, Any]) -> float:
        """Calculate confidence level in analysis"""
        quality = self._calculate_data_quality(game_state)
        consistency = self._assess_data_consistency(game_state)
        return (quality + consistency) / 2
    
    def _assess_data_consistency(self, game_state: Dict[str, Any]) -> float:
        """Assess consistency of game data"""
        # Check if data structures are consistent
        human_len = len(game_state.get('human_history', []))
        robot_len = len(game_state.get('robot_history', []))
        result_len = len(game_state.get('result_history', []))
        
        if human_len == robot_len == result_len:
            return 1.0
        else:
            return 0.5  # Some inconsistency detected
    
    # Placeholder methods for advanced analytics (to be implemented)
    def _analyze_repetitions(self, moves: List[str]) -> Dict[str, Any]:
        """Analyze repetition patterns"""
        if len(moves) < 3:
            return {
                'repetition_score': 0.0,
                'longest_streak': 0,
                'repetition_frequency': 0.0
            }
        
        # Count consecutive repeats
        streaks = []
        current_streak = 1
        
        for i in range(1, len(moves)):
            if moves[i] == moves[i-1]:
                current_streak += 1
            else:
                streaks.append(current_streak)
                current_streak = 1
        streaks.append(current_streak)
        
        longest_streak = max(streaks)
        repetition_frequency = sum(1 for s in streaks if s > 1) / len(streaks)
        repetition_score = min(longest_streak / len(moves), 1.0)
        
        return {
            'repetition_score': self._format_metric(repetition_score),
            'longest_streak': longest_streak,
            'repetition_frequency': self._format_metric(repetition_frequency)
        }
    
    def _detect_cycling_patterns(self, moves: List[str]) -> Dict[str, Any]:
        """Detect cycling patterns"""
        if len(moves) < 6:
            return {
                'cycling_score': 0.0,
                'detected_cycles': [],
                'dominant_cycle_length': 0
            }
        
        cycles = []
        # Check for cycles of length 2-5
        for cycle_length in range(2, min(6, len(moves)//2 + 1)):
            for start in range(len(moves) - cycle_length * 2 + 1):
                pattern = moves[start:start + cycle_length]
                next_pattern = moves[start + cycle_length:start + cycle_length * 2]
                if pattern == next_pattern:
                    cycles.append({
                        'pattern': pattern,
                        'length': cycle_length,
                        'position': start
                    })
        
        cycling_score = len(cycles) / max(1, len(moves) // 3)
        dominant_cycle_length = max((c['length'] for c in cycles), default=0)
        
        return {
            'cycling_score': self._format_metric(min(cycling_score, 1.0)),
            'detected_cycles': cycles[:3],  # Top 3 cycles
            'dominant_cycle_length': dominant_cycle_length
        }
    
    def _analyze_ai_adaptation(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze AI adaptation patterns"""
        human_moves = game_state.get('human_moves', [])
        robot_moves = game_state.get('robot_moves', [])
        results = game_state.get('results', [])
        
        if len(human_moves) < 5:
            return {
                'adaptation_rate': 0.5,
                'prediction_accuracy': 0.33,
                'strategy_changes': 0
            }
        
        # Calculate AI prediction accuracy (how often AI counters human move)
        correct_predictions = 0
        for i, (human, robot) in enumerate(zip(human_moves, robot_moves)):
            if ((human == 'stone' and robot == 'paper') or
                (human == 'paper' and robot == 'scissor') or
                (human == 'scissor' and robot == 'stone')):
                correct_predictions += 1
        
        prediction_accuracy = correct_predictions / len(human_moves)
        
        # Estimate strategy changes by looking at AI move diversity over time
        recent_moves = robot_moves[-10:] if len(robot_moves) >= 10 else robot_moves
        early_moves = robot_moves[:10] if len(robot_moves) >= 10 else []
        
        strategy_changes = 0
        if len(early_moves) >= 3 and len(recent_moves) >= 3:
            early_diversity = len(set(early_moves)) / len(early_moves)
            recent_diversity = len(set(recent_moves)) / len(recent_moves)
            strategy_changes = abs(recent_diversity - early_diversity)
        
        # Adaptation rate based on improving performance over time
        if len(results) >= 10:
            early_wins = results[:len(results)//2].count('robot')
            later_wins = results[len(results)//2:].count('robot')
            early_rate = early_wins / (len(results)//2)
            later_rate = later_wins / (len(results) - len(results)//2)
            adaptation_rate = (later_rate - early_rate + 1) / 2  # Normalize to 0-1
        else:
            adaptation_rate = 0.5
        
        return {
            'adaptation_rate': self._format_metric(max(0, min(1, adaptation_rate))),
            'prediction_accuracy': self._format_metric(prediction_accuracy),
            'strategy_changes': self._format_metric(strategy_changes)
        }
    
    def _analyze_prediction_patterns(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze prediction patterns"""
        human_moves = game_state.get('human_moves', [])
        robot_moves = game_state.get('robot_moves', [])
        
        if len(human_moves) < 3:
            return {
                'predictability_score': 0.5,
                'pattern_strength': 0.0,
                'prediction_confidence': 0.33
            }
        
        # Analyze human predictability from AI perspective
        correct_counters = 0
        pattern_matches = 0
        
        for i in range(1, len(human_moves)):
            # Check if AI successfully predicted and countered
            prev_human = human_moves[i-1]
            current_human = human_moves[i]
            ai_move = robot_moves[i] if i < len(robot_moves) else 'stone'
            
            # Check if AI countered correctly
            if ((current_human == 'stone' and ai_move == 'paper') or
                (current_human == 'paper' and ai_move == 'scissor') or
                (current_human == 'scissor' and ai_move == 'stone')):
                correct_counters += 1
            
            # Check for simple pattern following
            if i >= 2 and human_moves[i-2] == current_human:
                pattern_matches += 1
        
        predictability_score = correct_counters / max(1, len(human_moves) - 1)
        pattern_strength = pattern_matches / max(1, len(human_moves) - 2)
        prediction_confidence = (predictability_score + pattern_strength) / 2
        
        return {
            'predictability_score': self._format_metric(predictability_score),
            'pattern_strength': self._format_metric(pattern_strength),
            'prediction_confidence': self._format_metric(prediction_confidence)
        }
    
    def _analyze_change_point_timing(self, change_points: List[Dict]) -> Dict[str, Any]:
        """Analyze timing of change points"""
        return {'change_point_count': len(change_points)}
    
    def _calculate_decision_complexity(self, moves: List[str]) -> float:
        """Calculate decision complexity score"""
        if len(moves) < 5:
            return self._format_metric(0.0)
        
        # Calculate based on entropy and pattern diversity
        entropy = self._calculate_entropy(moves)
        
        # Check for pattern breaks vs consistency
        bigrams = self._count_bigrams(moves)
        unique_patterns = len(bigrams)
        max_possible_patterns = 9  # 3 moves * 3 moves
        
        pattern_diversity = unique_patterns / max_possible_patterns
        complexity = (entropy / 1.585) * 0.7 + pattern_diversity * 0.3
        
        return self._format_metric(complexity)
    
    def _calculate_strategy_consistency(self, moves: List[str]) -> float:
        """Calculate strategy consistency"""
        if len(moves) < 5:
            return self._format_metric(0.5)
            
        # Measure consistency through variance in move distribution over time
        window_size = 5
        distributions = []
        
        for i in range(len(moves) - window_size + 1):
            window = moves[i:i + window_size]
            dist = self._calculate_move_distribution(window)
            distributions.append([dist['paper'], dist['scissor'], dist['stone']])
        
        # Calculate variance across windows
        if len(distributions) < 2:
            return self._format_metric(0.5)
            
        variances = []
        for move_idx in range(3):
            values = [dist[move_idx] for dist in distributions]
            if len(values) > 1:
                mean_val = sum(values) / len(values)
                variance = sum((x - mean_val) ** 2 for x in values) / len(values)
                variances.append(variance)
        
        avg_variance = sum(variances) / len(variances) if variances else 0
        consistency = 1.0 - min(avg_variance * 3, 1.0)  # Normalize
        
        return self._format_metric(consistency)
    
    def _calculate_adaptation_rate(self, game_state: Dict[str, Any]) -> float:
        """Calculate adaptation rate"""
        moves = game_state.get('human_history', [])
        results = game_state.get('result_history', [])
        
        if len(moves) < 10 or len(results) < 10:
            return self._format_metric(0.5)
        
        # Look at strategy changes after losses
        adaptations = 0
        loss_opportunities = 0
        
        for i in range(1, min(len(moves), len(results))):
            if results[i-1] == 'robot':  # Previous move was a loss
                loss_opportunities += 1
                # Check if player changed strategy
                if moves[i] != moves[i-1]:
                    adaptations += 1
        
        adaptation_rate = adaptations / loss_opportunities if loss_opportunities > 0 else 0.5
        return self._format_metric(adaptation_rate)
    
    def _calculate_mutual_information(self, game_state: Dict[str, Any]) -> float:
        """Calculate mutual information between consecutive moves"""
        moves = game_state.get('human_history', [])
        
        if len(moves) < 3:
            return self._format_metric(0.0)
        
        # Calculate joint probability of consecutive moves
        bigrams = self._count_bigrams(moves)
        total_bigrams = len(moves) - 1
        
        # Individual move probabilities
        move_dist = self._calculate_move_distribution(moves)
        
        # Mutual information calculation
        mutual_info = 0.0
        for bigram, count in bigrams.items():
            if '->' in bigram:
                move1, move2 = bigram.split('->')
                p_joint = count / total_bigrams
                p_move1 = move_dist.get(move1, 0)
                p_move2 = move_dist.get(move2, 0)
                
                if p_joint > 0 and p_move1 > 0 and p_move2 > 0:
                    mutual_info += p_joint * math.log2(p_joint / (p_move1 * p_move2))
        
        return self._format_metric(mutual_info)
    
    def _calculate_compression_ratio(self, moves: List[str]) -> float:
        """Calculate compression ratio (measure of pattern complexity)"""
        if len(moves) < 5:
            return self._format_metric(1.0)
        
        # Simple run-length encoding simulation
        compressed_length = 1
        current_move = moves[0]
        run_length = 1
        
        for move in moves[1:]:
            if move == current_move:
                run_length += 1
            else:
                compressed_length += 1  # New run
                current_move = move
                run_length = 1
        
        compression_ratio = compressed_length / len(moves)
        return self._format_metric(compression_ratio)
    
    def _calculate_nash_distance(self, moves: List[str]) -> float:
        """Calculate distance from Nash equilibrium (1/3, 1/3, 1/3)"""
        if not moves:
            return self._format_metric(0.333)  # Default distance
        
        distribution = self._calculate_move_distribution(moves)
        nash_prob = 1.0 / 3.0
        
        # Calculate Euclidean distance from Nash equilibrium
        distance = math.sqrt(
            (distribution['paper'] - nash_prob) ** 2 +
            (distribution['scissor'] - nash_prob) ** 2 +
            (distribution['stone'] - nash_prob) ** 2
        )
        
        return self._format_metric(distance)
    
    def _calculate_exploitability(self, game_state: Dict[str, Any]) -> float:
        """Calculate exploitability score"""
        moves = game_state.get('human_history', [])
        
        if len(moves) < 5:
            return self._format_metric(0.0)
        
        # High predictability = high exploitability
        predictability = self._calculate_predictability_score(moves)
        
        # Pattern repetition increases exploitability
        patterns = self._detect_sequence_patterns(moves)
        if patterns.get('insufficient_data'):
            pattern_score = 0.0
        else:
            bigrams = patterns.get('bigrams', {})
            max_frequency = max(bigrams.values()) if bigrams else 0
            total_bigrams = sum(bigrams.values()) if bigrams else 1
            pattern_score = max_frequency / total_bigrams
        
        exploitability = (predictability * 0.7) + (pattern_score * 0.3)
        return self._format_metric(exploitability)
    
    def _analyze_counter_strategies(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze counter-strategy effectiveness"""
        human_moves = game_state.get('human_moves', [])
        robot_moves = game_state.get('robot_moves', [])
        results = game_state.get('results', [])
        
        if len(human_moves) < 3:
            return {
                'counter_effectiveness': 0.33,
                'successful_counters': 0,
                'optimal_strategy': 'random'
            }
        
        # Analyze what would have been optimal counter-strategies
        successful_counters = 0
        total_opportunities = 0
        
        # Counter mapping
        counters = {'stone': 'paper', 'paper': 'scissor', 'scissor': 'stone'}
        
        for i, human_move in enumerate(human_moves):
            if i < len(robot_moves):
                optimal_counter = counters.get(human_move, 'stone')
                robot_move = robot_moves[i]
                
                if robot_move == optimal_counter:
                    successful_counters += 1
                total_opportunities += 1
        
        counter_effectiveness = successful_counters / max(1, total_opportunities)
        
        # Suggest optimal strategy based on human patterns
        move_counts = {'stone': 0, 'paper': 0, 'scissor': 0}
        for move in human_moves:
            if move in move_counts:
                move_counts[move] += 1
        
        most_common_human = max(move_counts, key=lambda x: move_counts[x])
        optimal_strategy = counters.get(most_common_human, 'random')
        
        return {
            'counter_effectiveness': self._format_metric(counter_effectiveness),
            'successful_counters': successful_counters,
            'optimal_strategy': optimal_strategy
        }
    
    def _assess_impulsiveness(self, moves: List[str]) -> float:
        """Assess impulsiveness indicator"""
        if len(moves) < 5:
            return 0.5
        
        # Calculate quick reversals and inconsistent patterns
        quick_changes = 0
        for i in range(2, len(moves)):
            # Count quick reversals (A -> B -> A pattern)
            if moves[i] == moves[i-2] and moves[i] != moves[i-1]:
                quick_changes += 1
        
        # Calculate variety within short windows
        window_variety = 0
        window_size = 3
        for i in range(len(moves) - window_size + 1):
            window = moves[i:i + window_size]
            if len(set(window)) == 3:  # All different moves in 3-move window
                window_variety += 1
        
        impulsiveness = (quick_changes / max(1, len(moves) - 2)) * 0.6 + \
                       (window_variety / max(1, len(moves) - window_size + 1)) * 0.4
        
        return self._format_metric(min(1.0, impulsiveness))
    
    def _assess_consistency(self, moves: List[str]) -> float:
        """Assess consistency score"""
        if len(moves) < 3:
            return 0.5
        
        # Calculate frequency distribution stability
        move_counts = {'stone': 0, 'paper': 0, 'scissor': 0}
        for move in moves:
            if move in move_counts:
                move_counts[move] += 1
        
        total_moves = len(moves)
        frequencies = [count / total_moves for count in move_counts.values()]
        
        # Perfect consistency would be equal distribution (0.33, 0.33, 0.33)
        target_freq = 1.0 / 3.0
        variance = sum((freq - target_freq) ** 2 for freq in frequencies) / 3
        
        # Lower variance = higher consistency, inverted and normalized
        consistency = max(0, 1 - (variance * 9))  # Scale factor to normalize
        
        return self._format_metric(consistency)
    
    def _assess_risk_tolerance(self, moves: List[str], results: List[str]) -> float:
        """Assess risk tolerance"""
        if len(moves) < 3 or len(results) < 3:
            return 0.5
        
        # Analyze behavior after losses
        risk_taking_after_loss = 0
        loss_situations = 0
        
        for i in range(1, min(len(moves), len(results))):
            if results[i-1] == 'lose':  # Human lost previous round
                loss_situations += 1
                
                # Check if human changed strategy (higher risk)
                if i >= 2 and moves[i] != moves[i-1]:
                    risk_taking_after_loss += 1
        
        # Calculate risk tolerance
        if loss_situations > 0:
            risk_tolerance = risk_taking_after_loss / loss_situations
        else:
            # Assess overall variety as risk indicator
            move_variety = len(set(moves)) / 3.0  # Normalized by max possible variety
            risk_tolerance = move_variety
        
        return self._format_metric(risk_tolerance)
    
    def _detect_frustration_patterns(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Detect frustration patterns based on gameplay behavior"""
        human_moves = game_state.get('human_moves', game_state.get('human_history', []))
        results = game_state.get('results', game_state.get('result_history', []))
        
        if len(human_moves) < 5:
            return {'status': 'insufficient_data', 'rounds_needed': 5 - len(human_moves)}
        
        # Detect frustration indicators
        frustration_indicators = {}
        
        # Pattern 1: Increased repetition after losses
        recent_losses = [i for i, result in enumerate(results[-10:]) if result in ['lose', 'Loss']]
        if recent_losses:
            moves_after_losses = []
            for loss_idx in recent_losses:
                if loss_idx + 1 < len(human_moves):
                    moves_after_losses.append(human_moves[loss_idx + 1])
            
            if moves_after_losses:
                # Count repetition after losses
                repetition_rate = len(moves_after_losses) - len(set(moves_after_losses))
                frustration_indicators['post_loss_repetition'] = self._format_metric(repetition_rate / len(moves_after_losses))
        
        # Pattern 2: Rapid strategy changes (erratic behavior)
        strategy_changes = 0
        for i in range(1, min(len(human_moves), 10)):
            if human_moves[-i] != human_moves[-(i+1)]:
                strategy_changes += 1
        
        frustration_indicators['recent_erratic_changes'] = self._format_metric(strategy_changes / min(len(human_moves), 10))
        
        # Pattern 3: Breaking established patterns suddenly
        if len(human_moves) >= 8:
            early_pattern = Counter(human_moves[-8:-4])
            recent_pattern = Counter(human_moves[-4:])
            
            pattern_disruption = 0
            for move in early_pattern:
                early_freq = early_pattern[move] / 4
                recent_freq = recent_pattern.get(move, 0) / 4
                pattern_disruption += abs(early_freq - recent_freq)
            
            frustration_indicators['pattern_disruption'] = self._format_metric(pattern_disruption)
        
        # Overall frustration score
        if frustration_indicators:
            avg_frustration = sum(frustration_indicators.values()) / len(frustration_indicators)
            frustration_level = 'low' if avg_frustration < 0.3 else 'medium' if avg_frustration < 0.6 else 'high'
        else:
            avg_frustration = 0.0
            frustration_level = 'unknown'
        
        return {
            'overall_frustration_score': self._format_metric(avg_frustration),
            'frustration_level': frustration_level,
            'indicators': frustration_indicators,
            'analysis_confidence': self._format_metric(min(len(human_moves) / 10, 1.0))
        }
    
    def _analyze_confidence_patterns(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze confidence patterns based on decision timing and consistency"""
        human_moves = game_state.get('human_moves', game_state.get('human_history', []))
        results = game_state.get('results', game_state.get('result_history', []))
        
        if len(human_moves) < 3:
            return {'status': 'insufficient_data', 'rounds_needed': 3 - len(human_moves)}
        
        confidence_indicators = {}
        
        # Pattern 1: Consistency after wins vs losses
        win_indices = [i for i, result in enumerate(results) if result in ['win', 'Win']]
        loss_indices = [i for i, result in enumerate(results) if result in ['lose', 'Loss']]
        
        if win_indices and len(win_indices) >= 2:
            # Moves after wins - confident players tend to stick with winning strategies
            moves_after_wins = [human_moves[i+1] for i in win_indices if i+1 < len(human_moves)]
            if moves_after_wins:
                win_consistency = len(moves_after_wins) - len(set(moves_after_wins))
                confidence_indicators['post_win_consistency'] = self._format_metric(1 - (win_consistency / len(moves_after_wins)))
        
        if loss_indices and len(loss_indices) >= 2:
            # Moves after losses - confident players adapt more systematically
            moves_after_losses = [human_moves[i+1] for i in loss_indices if i+1 < len(human_moves)]
            if moves_after_losses:
                loss_adaptation = len(set(moves_after_losses)) / len(moves_after_losses)
                confidence_indicators['post_loss_adaptation'] = self._format_metric(loss_adaptation)
        
        # Pattern 2: Move distribution balance (confident players use all moves)
        move_distribution = Counter(human_moves)
        total_moves = len(human_moves)
        entropy = 0
        for count in move_distribution.values():
            p = count / total_moves
            if p > 0:
                entropy -= p * math.log2(p)
        
        max_entropy = math.log2(min(3, len(set(human_moves))))
        if max_entropy > 0:
            confidence_indicators['move_diversity'] = self._format_metric(entropy / max_entropy)
        
        # Pattern 3: Recovery from losing streaks
        current_streak = 0
        streak_type = None
        for result in reversed(results):
            if streak_type is None:
                streak_type = result
                current_streak = 1
            elif result == streak_type:
                current_streak += 1
            else:
                break
        
        if streak_type in ['lose', 'Loss'] and current_streak >= 3:
            confidence_indicators['losing_streak_resilience'] = self._format_metric(1.0 / current_streak)
        elif streak_type in ['win', 'Win']:
            confidence_indicators['winning_streak_confidence'] = self._format_metric(min(current_streak / 5, 1.0))
        
        # Overall confidence assessment
        if confidence_indicators:
            avg_confidence = sum(confidence_indicators.values()) / len(confidence_indicators)
            confidence_level = 'low' if avg_confidence < 0.4 else 'medium' if avg_confidence < 0.7 else 'high'
        else:
            avg_confidence = 0.5
            confidence_level = 'unknown'
        
        return {
            'overall_confidence_score': self._format_metric(avg_confidence),
            'confidence_level': confidence_level,
            'indicators': confidence_indicators,
            'current_streak': current_streak,
            'streak_type': streak_type,
            'analysis_confidence': self._format_metric(min(len(human_moves) / 15, 1.0))
        }
    
    def _detect_tilt_behavior(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Detect tilt behavior - emotional decision making that deviates from optimal play"""
        human_moves = game_state.get('human_moves', game_state.get('human_history', []))
        results = game_state.get('results', game_state.get('result_history', []))
        
        if len(human_moves) < 6:
            return {'status': 'insufficient_data', 'rounds_needed': 6 - len(human_moves)}
        
        tilt_indicators = {}
        
        # Pattern 1: Immediate reaction to losses (emotional decisions)
        immediate_responses = []
        for i in range(len(results) - 1):
            if results[i] in ['lose', 'Loss'] and i + 1 < len(human_moves):
                # Check if move changes immediately after loss
                if i > 0 and human_moves[i] != human_moves[i + 1]:
                    immediate_responses.append(1)  # Changed move
                else:
                    immediate_responses.append(0)  # Kept move
        
        if immediate_responses:
            tilt_indicators['immediate_loss_reaction'] = self._format_metric(sum(immediate_responses) / len(immediate_responses))
        
        # Pattern 2: Abandoning working strategies after single loss
        strategy_abandonment = 0
        for i in range(2, len(results) - 1):
            # If last 2 were wins with same move, but changed after one loss
            if (results[i-2:i] == ['win', 'win'] and 
                human_moves[i-2] == human_moves[i-1] and 
                results[i] in ['lose', 'Loss'] and 
                i + 1 < len(human_moves) and
                human_moves[i+1] != human_moves[i]):
                strategy_abandonment += 1
        
        total_opportunities = max(1, len(results) - 3)
        tilt_indicators['strategy_abandonment'] = self._format_metric(strategy_abandonment / total_opportunities)
        
        # Pattern 3: Extreme randomness after losses (panic mode)
        recent_losses = [i for i, result in enumerate(results[-8:]) if result in ['lose', 'Loss']]
        if recent_losses:
            moves_during_losses = human_moves[-8:]
            loss_period_entropy = 0
            
            if len(moves_during_losses) >= 3:
                move_counts = Counter(moves_during_losses)
                total = len(moves_during_losses)
                for count in move_counts.values():
                    p = count / total
                    if p > 0:
                        loss_period_entropy -= p * math.log2(p)
                
                # Higher entropy during losses might indicate tilt
                max_entropy = math.log2(3)
                tilt_indicators['loss_period_chaos'] = self._format_metric(loss_period_entropy / max_entropy)
        
        # Pattern 4: Repetitive behavior under pressure (freeze response)
        if len(human_moves) >= 10:
            recent_moves = human_moves[-6:]
            repetition_rate = 0
            for i in range(1, len(recent_moves)):
                if recent_moves[i] == recent_moves[i-1]:
                    repetition_rate += 1
            
            tilt_indicators['repetition_under_pressure'] = self._format_metric(repetition_rate / (len(recent_moves) - 1))
        
        # Overall tilt assessment
        if tilt_indicators:
            avg_tilt = sum(tilt_indicators.values()) / len(tilt_indicators)
            tilt_level = 'none' if avg_tilt < 0.3 else 'mild' if avg_tilt < 0.6 else 'severe'
        else:
            avg_tilt = 0.0
            tilt_level = 'unknown'
        
        return {
            'overall_tilt_score': self._format_metric(avg_tilt),
            'tilt_level': tilt_level,
            'indicators': tilt_indicators,
            'risk_factors': len([v for v in tilt_indicators.values() if v > 0.5]),
            'analysis_confidence': self._format_metric(min(len(human_moves) / 12, 1.0))
        }
    
    def _assess_pattern_awareness(self, game_state: Dict[str, Any]) -> float:
        """Assess player's awareness of their own patterns"""
        human_moves = game_state.get('human_moves', game_state.get('human_history', []))
        
        if len(human_moves) < 8:
            return 0.0  # Not enough data
        
        # Check if player breaks their own patterns
        pattern_breaks = 0
        total_patterns = 0
        
        # Look for 2-3 move patterns and see if player breaks them
        for pattern_length in [2, 3]:
            if len(human_moves) < pattern_length * 3:
                continue
                
            for i in range(len(human_moves) - pattern_length * 2):
                # Get a pattern
                pattern = human_moves[i:i + pattern_length]
                
                # Look ahead to see if pattern continues
                next_expected = pattern[0]  # First move of pattern repetition
                actual_next_idx = i + pattern_length
                
                if actual_next_idx < len(human_moves):
                    total_patterns += 1
                    if human_moves[actual_next_idx] != next_expected:
                        pattern_breaks += 1
        
        if total_patterns == 0:
            return 0.5
            
        awareness_score = pattern_breaks / total_patterns
        return self._format_metric(awareness_score)
    
    def _assess_meta_cognition(self, game_state: Dict[str, Any]) -> float:
        """Assess meta-cognitive indicators - thinking about thinking"""
        human_moves = game_state.get('human_moves', game_state.get('human_history', []))
        results = game_state.get('results', game_state.get('result_history', []))
        
        if len(human_moves) < 10:
            return 0.0
        
        meta_indicators = []
        
        # Indicator 1: Adaptation after multiple losses with same strategy
        adaptation_score = 0
        for i in range(3, len(results)):
            # Check if lost 2+ times with same move, then changed
            if (results[i-3:i-1] == ['lose', 'lose'] and 
                human_moves[i-3] == human_moves[i-2] and
                i < len(human_moves) and
                human_moves[i] != human_moves[i-1]):
                adaptation_score += 1
        
        if len(results) > 5:
            meta_indicators.append(adaptation_score / (len(results) - 5))
        
        # Indicator 2: Counter-strategy deployment (using rock against frequent scissors, etc.)
        if len(human_moves) >= 8:
            recent_ai_moves = game_state.get('robot_moves', game_state.get('robot_history', []))[-8:]
            recent_human_moves = human_moves[-8:]
            
            if recent_ai_moves:
                ai_most_common = Counter(recent_ai_moves).most_common(1)[0][0]
                # Check if human is using the counter-move more frequently
                counter_moves = {'stone': 'paper', 'paper': 'scissor', 'scissor': 'stone',
                               'rock': 'paper', 'R': 'P', 'P': 'S', 'S': 'R'}
                
                expected_counter = counter_moves.get(ai_most_common.lower(), None)
                if expected_counter:
                    human_counter_count = sum(1 for move in recent_human_moves 
                                           if move.lower().startswith(expected_counter.lower()))
                    counter_usage = human_counter_count / len(recent_human_moves)
                    meta_indicators.append(counter_usage)
        
        # Indicator 3: Variance in strategy (shows deliberate experimentation)
        if len(human_moves) >= 12:
            windows = [human_moves[i:i+4] for i in range(0, len(human_moves)-3, 4)]
            if len(windows) >= 3:
                window_entropies = []
                for window in windows:
                    counts = Counter(window)
                    entropy = 0
                    for count in counts.values():
                        p = count / len(window)
                        if p > 0:
                            entropy -= p * math.log2(p)
                    window_entropies.append(entropy)
                
                # High variance in entropy suggests deliberate strategy changes
                if len(window_entropies) > 1:
                    mean_entropy = sum(window_entropies) / len(window_entropies)
                    variance = sum((e - mean_entropy) ** 2 for e in window_entropies) / len(window_entropies)
                    meta_indicators.append(min(variance, 1.0))
        
        if not meta_indicators:
            return 0.5
        
        meta_score = sum(meta_indicators) / len(meta_indicators)
        return self._format_metric(meta_score)
    
    def _detect_learning_patterns(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Detect learning patterns and skill development"""
        human_moves = game_state.get('human_moves', game_state.get('human_history', []))
        results = game_state.get('results', game_state.get('result_history', []))
        
        if len(human_moves) < 10:
            return {'status': 'insufficient_data', 'rounds_needed': 10 - len(human_moves)}
        
        learning_indicators = {}
        
        # Pattern 1: Improvement in win rate over time
        if len(results) >= 10:
            early_results = results[:len(results)//2]
            late_results = results[len(results)//2:]
            
            early_win_rate = early_results.count('win') / len(early_results)
            late_win_rate = late_results.count('win') / len(late_results)
            
            learning_indicators['win_rate_improvement'] = self._format_metric(late_win_rate - early_win_rate)
        
        # Pattern 2: Decreasing predictability over time
        if len(human_moves) >= 16:
            early_moves = human_moves[:len(human_moves)//2]
            late_moves = human_moves[len(human_moves)//2:]
            
            def calculate_predictability(moves):
                if len(moves) < 3:
                    return 0.5
                most_common_count = Counter(moves).most_common(1)[0][1]
                return most_common_count / len(moves)
            
            early_predictability = calculate_predictability(early_moves)
            late_predictability = calculate_predictability(late_moves)
            
            learning_indicators['unpredictability_growth'] = self._format_metric(early_predictability - late_predictability)
        
        # Pattern 3: Pattern recognition and breaking
        pattern_recognition_score = 0
        if len(human_moves) >= 12:
            # Look for instances where player breaks their own 3-move patterns
            for i in range(len(human_moves) - 5):
                pattern = human_moves[i:i+3]
                next_expected = pattern[0]  # If pattern repeats
                if i + 3 < len(human_moves) and human_moves[i+3] != next_expected:
                    pattern_recognition_score += 1
            
            learning_indicators['pattern_breaking_skill'] = self._format_metric(
                pattern_recognition_score / max(1, len(human_moves) - 5)
            )
        
        # Pattern 4: Adaptation to AI strategy changes
        ai_strategy = game_state.get('current_strategy', 'unknown')
        robot_moves = game_state.get('robot_moves', game_state.get('robot_history', []))
        
        if len(robot_moves) >= 8 and len(human_moves) >= 8:
            # Check if human adapted to AI's most common move
            ai_recent = robot_moves[-8:]
            human_recent = human_moves[-8:]
            
            ai_most_common = Counter(ai_recent).most_common(1)[0][0]
            
            # Count appropriate counter-moves
            counters = {'stone': ['paper', 'P'], 'rock': ['paper', 'P'], 'R': ['paper', 'P'],
                       'paper': ['scissor', 'S'], 'P': ['scissor', 'S'],
                       'scissor': ['stone', 'rock', 'R'], 'S': ['stone', 'rock', 'R']}
            
            appropriate_counters = counters.get(ai_most_common, [])
            counter_count = sum(1 for move in human_recent if move in appropriate_counters)
            
            learning_indicators['ai_adaptation_rate'] = self._format_metric(counter_count / len(human_recent))
        
        # Overall learning assessment
        if learning_indicators:
            # Weight positive learning indicators
            positive_indicators = [v for v in learning_indicators.values() if v > 0]
            avg_learning = sum(positive_indicators) / len(positive_indicators) if positive_indicators else 0
            
            learning_stage = ('beginner' if avg_learning < 0.3 else 
                            'developing' if avg_learning < 0.6 else 'advanced')
        else:
            avg_learning = 0.0
            learning_stage = 'unknown'
        
        return {
            'overall_learning_score': self._format_metric(avg_learning),
            'learning_stage': learning_stage,
            'indicators': learning_indicators,
            'data_points_analyzed': len(human_moves),
            'confidence_level': self._format_metric(min(len(human_moves) / 20, 1.0))
        }
    
    def _assess_current_strategy(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Assess current human strategy and its effectiveness"""
        human_moves = game_state.get('human_moves', game_state.get('human_history', []))
        results = game_state.get('results', game_state.get('result_history', []))
        
        if len(human_moves) < 5:
            return {'status': 'insufficient_data', 'rounds_needed': 5 - len(human_moves)}
        
        strategy_analysis = {}
        
        # Analyze recent move distribution (strategy type)
        recent_moves = human_moves[-8:] if len(human_moves) >= 8 else human_moves
        move_counts = Counter(recent_moves)
        total_recent = len(recent_moves)
        
        # Determine strategy type
        if len(set(recent_moves)) == 1:
            strategy_type = "single_move"
            strategy_analysis['strategy_type'] = 'Single Move Repetition'
        elif max(move_counts.values()) / total_recent >= 0.6:
            strategy_type = "heavily_biased"
            most_used = move_counts.most_common(1)[0][0]
            strategy_analysis['strategy_type'] = f'Heavily Biased ({most_used})'
        elif len(set(recent_moves)) == len(recent_moves):
            strategy_type = "fully_random"
            strategy_analysis['strategy_type'] = 'Fully Random'
        else:
            strategy_type = "mixed"
            strategy_analysis['strategy_type'] = 'Mixed Strategy'
        
        # Calculate strategy effectiveness
        if len(results) >= len(recent_moves):
            recent_results = results[-len(recent_moves):]
            win_rate = recent_results.count('win') / len(recent_results)
            strategy_analysis['recent_effectiveness'] = self._format_metric(win_rate)
            
            effectiveness_rating = ('poor' if win_rate < 0.3 else 
                                   'fair' if win_rate < 0.5 else 
                                   'good' if win_rate < 0.7 else 'excellent')
            strategy_analysis['effectiveness_rating'] = effectiveness_rating
        
        # Analyze pattern consistency
        pattern_consistency = 0
        if len(human_moves) >= 6:
            for i in range(len(human_moves) - 2):
                pattern = human_moves[i:i+2]
                if i + 2 < len(human_moves) - 1:
                    next_actual = human_moves[i+2]
                    # Check if this pattern appeared before and what followed
                    for j in range(i):
                        if human_moves[j:j+2] == pattern and j+2 < len(human_moves):
                            if human_moves[j+2] == next_actual:
                                pattern_consistency += 1
                            break
        
        if len(human_moves) > 6:
            strategy_analysis['pattern_consistency'] = self._format_metric(
                pattern_consistency / max(1, len(human_moves) - 6)
            )
        
        # Analyze adaptation rate
        strategy_changes = 0
        if len(human_moves) >= 8:
            early_dist = Counter(human_moves[:4])
            late_dist = Counter(human_moves[-4:])
            
            # Calculate distribution change
            change_score = 0
            for move in ['stone', 'paper', 'scissor', 'rock', 'R', 'P', 'S']:
                early_freq = early_dist.get(move, 0) / 4
                late_freq = late_dist.get(move, 0) / 4
                change_score += abs(early_freq - late_freq)
            
            strategy_analysis['adaptation_rate'] = self._format_metric(change_score / 2)
        
        return {
            'current_strategy': strategy_analysis,
            'ai_opponent_strategy': game_state.get('current_strategy', 'unknown'),
            'strategy_match_score': self._calculate_strategy_match_score(game_state),
            'recommended_adjustments': self._suggest_strategy_adjustments(strategy_analysis, game_state)
        }
    
    def _calculate_strategy_match_score(self, game_state: Dict[str, Any]) -> float:
        """Calculate how well human strategy matches against AI strategy"""
        ai_strategy = game_state.get('current_strategy', 'unknown')
        human_moves = game_state.get('human_moves', game_state.get('human_history', []))
        results = game_state.get('results', game_state.get('result_history', []))
        
        if len(results) < 5:
            return 0.5
        
        recent_win_rate = results[-5:].count('win') / 5
        
        # Adjust score based on AI difficulty
        difficulty_multipliers = {
            'random': 1.0,
            'frequency': 0.9,
            'markov': 0.8,
            'enhanced': 0.7,
            'lstm': 0.6
        }
        
        multiplier = difficulty_multipliers.get(ai_strategy, 0.8)
        match_score = recent_win_rate * multiplier
        
        return self._format_metric(match_score)
    
    def _suggest_strategy_adjustments(self, strategy_analysis: Dict[str, Any], game_state: Dict[str, Any]) -> List[str]:
        """Suggest strategy adjustments based on current performance"""
        suggestions = []
        
        effectiveness = strategy_analysis.get('recent_effectiveness', 0.5)
        strategy_type = strategy_analysis.get('strategy_type', 'unknown')
        
        if effectiveness < 0.4:
            if 'Single Move' in strategy_type:
                suggestions.append("Break the single-move pattern - it's too predictable")
            elif 'Heavily Biased' in strategy_type:
                suggestions.append("Reduce bias towards your favorite move")
            else:
                suggestions.append("Current approach isn't working - try a different strategy")
        
        if strategy_analysis.get('pattern_consistency', 0) > 0.7:
            suggestions.append("Your patterns are too consistent - add more randomness")
        
        ai_strategy = game_state.get('current_strategy', 'unknown')
        if ai_strategy == 'frequency':
            suggestions.append("AI is using frequency analysis - vary your move distribution")
        elif ai_strategy == 'markov':
            suggestions.append("AI is using pattern matching - break your sequences")
        elif ai_strategy == 'lstm':
            suggestions.append("AI is using neural networks - increase unpredictability")
        
        return suggestions[:3]  # Return top 3 suggestions
    
    def _identify_strategic_opportunities(self, game_state: Dict[str, Any]) -> List[str]:
        """Identify strategic opportunities based on current game state"""
        opportunities = []
        
        human_moves = game_state.get('human_moves', game_state.get('human_history', []))
        robot_moves = game_state.get('robot_moves', game_state.get('robot_history', []))
        results = game_state.get('results', game_state.get('result_history', []))
        
        if len(human_moves) < 5:
            return ["Play more rounds to identify strategic opportunities"]
        
        # Opportunity 1: Exploit AI patterns
        if len(robot_moves) >= 6:
            ai_recent = robot_moves[-6:]
            ai_counts = Counter(ai_recent)
            if ai_counts.most_common(1)[0][1] >= 4:  # AI is showing bias
                biased_move = ai_counts.most_common(1)[0][0]
                counter_move = {'stone': 'paper', 'rock': 'paper', 'R': 'paper',
                               'paper': 'scissor', 'P': 'scissor',
                               'scissor': 'stone', 'S': 'stone'}.get(biased_move, 'unknown')
                opportunities.append(f"AI is biased toward {biased_move} - use {counter_move} more often")
        
        # Opportunity 2: Breaking own patterns
        if len(human_moves) >= 8:
            human_recent = human_moves[-6:]
            human_counts = Counter(human_recent)
            if human_counts.most_common(1)[0][1] >= 4:
                opportunities.append("You're showing move bias - mix up your strategy")
        
        # Opportunity 3: Winning streak extension
        if len(results) >= 3 and results[-3:].count('win') >= 2:
            opportunities.append("You're on a winning streak - maintain current approach but add slight variations")
        
        # Opportunity 4: Loss recovery
        if len(results) >= 3 and results[-3:].count('lose') >= 2:
            opportunities.append("Break losing pattern - try opposite of your recent moves")
        
        # Opportunity 5: Entropy optimization
        if len(human_moves) >= 10:
            move_dist = Counter(human_moves[-10:])
            entropy = 0
            for count in move_dist.values():
                p = count / 10
                if p > 0:
                    entropy -= p * math.log2(p)
            
            max_entropy = math.log2(3)
            entropy_ratio = entropy / max_entropy
            
            if entropy_ratio < 0.8:
                opportunities.append("Increase randomness - you're too predictable")
            elif entropy_ratio > 0.95:
                opportunities.append("You're very unpredictable - maintain this randomness")
        
        return opportunities[:4]  # Return top 4 opportunities
    
    def _identify_strategic_weaknesses(self, game_state: Dict[str, Any]) -> List[str]:
        """Identify strategic weaknesses in current play"""
        weaknesses = []
        
        human_moves = game_state.get('human_moves', game_state.get('human_history', []))
        results = game_state.get('results', game_state.get('result_history', []))
        
        if len(human_moves) < 5:
            return ["Need more gameplay data to identify weaknesses"]
        
        # Weakness 1: Poor win rate
        if len(results) >= 8:
            recent_win_rate = results[-8:].count('win') / 8
            if recent_win_rate < 0.3:
                weaknesses.append("Low win rate - current strategy isn't effective")
        
        # Weakness 2: High predictability
        if len(human_moves) >= 8:
            recent_moves = human_moves[-8:]
            most_common_count = Counter(recent_moves).most_common(1)[0][1]
            if most_common_count >= 5:
                weaknesses.append("Too predictable - overusing certain moves")
        
        # Weakness 3: Poor adaptation after losses
        loss_adaptation_score = 0
        total_losses = 0
        for i, result in enumerate(results[:-1]):
            if result in ['lose', 'Loss']:
                total_losses += 1
                if i + 1 < len(human_moves) and i < len(human_moves):
                    # Check if player changed strategy after loss
                    if human_moves[i] != human_moves[i + 1]:
                        loss_adaptation_score += 1
        
        if total_losses >= 3:
            adaptation_rate = loss_adaptation_score / total_losses
            if adaptation_rate < 0.3:
                weaknesses.append("Poor adaptation after losses - not changing strategy when needed")
        
        # Weakness 4: Emotional decision making
        if len(results) >= 10:
            # Look for erratic behavior after losses
            erratic_count = 0
            for i in range(1, len(results) - 1):
                if results[i] in ['lose', 'Loss']:
                    # Check if next few moves are very different
                    if (i + 2 < len(human_moves) and 
                        len(set(human_moves[i+1:i+3])) == 2):  # Changed moves rapidly
                        erratic_count += 1
            
            if erratic_count >= 2:
                weaknesses.append("Emotional reactions to losses - making hasty strategy changes")
        
        # Weakness 5: Lack of counter-strategy
        robot_moves = game_state.get('robot_moves', game_state.get('robot_history', []))
        if len(robot_moves) >= 8 and len(human_moves) >= 8:
            ai_recent = robot_moves[-8:]
            human_recent = human_moves[-8:]
            
            ai_most_common = Counter(ai_recent).most_common(1)[0][0]
            
            # Check if human is not using effective counters
            counter_moves = {'stone': 'paper', 'rock': 'paper', 'R': 'paper',
                           'paper': 'scissor', 'P': 'scissor', 
                           'scissor': 'stone', 'S': 'stone'}
            
            expected_counter = counter_moves.get(ai_most_common, None)
            if expected_counter:
                counter_usage = sum(1 for move in human_recent if move.lower().startswith(expected_counter.lower()))
                if counter_usage < 2:  # Less than 25% counter usage
                    weaknesses.append(f"Not countering AI's {ai_most_common} bias effectively")
        
        return weaknesses[:4]  # Return top 4 weaknesses
    
    def _generate_adaptation_suggestions(self, game_state: Dict[str, Any]) -> List[str]:
        """Generate specific adaptation suggestions"""
        suggestions = []
        
        ai_strategy = game_state.get('current_strategy', 'unknown')
        human_moves = game_state.get('human_moves', game_state.get('human_history', []))
        results = game_state.get('results', game_state.get('result_history', []))
        
        if len(human_moves) < 3:
            return ["Play more rounds to get personalized adaptation suggestions"]
        
        # AI-specific adaptations
        if ai_strategy == 'frequency':
            suggestions.append("Against frequency AI: Balance your move distribution (aim for 33% each)")
        elif ai_strategy == 'markov':
            suggestions.append("Against Markov AI: Break patterns by avoiding predictable sequences")
        elif ai_strategy == 'enhanced':
            suggestions.append("Against enhanced AI: Use counter-exploitation and mixed strategies")
        elif ai_strategy == 'lstm':
            suggestions.append("Against LSTM AI: Maximize entropy and avoid any learnable patterns")
        
        # Performance-based adaptations
        if len(results) >= 5:
            recent_performance = results[-5:].count('win') / 5
            if recent_performance < 0.4:
                suggestions.append("Low recent performance: Try inverse of your recent strategy")
            elif recent_performance > 0.6:
                suggestions.append("Good performance: Maintain strategy but add small variations")
        
        # Pattern-based adaptations
        if len(human_moves) >= 6:
            recent_moves = human_moves[-6:]
            unique_moves = len(set(recent_moves))
            
            if unique_moves == 1:
                suggestions.append("Break single-move pattern: Introduce other moves gradually")
            elif unique_moves == 2:
                suggestions.append("Add third move type: Complete your strategic toolkit")
            else:
                suggestions.append("Good variety: Focus on timing and sequence optimization")
        
        # Streak-based adaptations
        if len(results) >= 3:
            current_streak = 1
            streak_type = results[-1]
            for i in range(len(results) - 2, -1, -1):
                if results[i] == streak_type:
                    current_streak += 1
                else:
                    break
            
            if streak_type in ['lose', 'Loss'] and current_streak >= 3:
                suggestions.append("Long losing streak: Complete strategy reset recommended")
            elif streak_type in ['win', 'Win'] and current_streak >= 3:
                suggestions.append("Winning streak: Add subtle variations to maintain momentum")
        
        return suggestions[:4]  # Return top 4 suggestions
    
    def _determine_educational_focus(self, game_state: Dict[str, Any]) -> List[str]:
        """Determine educational focus areas for improvement"""
        focus_areas = []
        
        human_moves = game_state.get('human_moves', game_state.get('human_history', []))
        results = game_state.get('results', game_state.get('result_history', []))
        
        if len(human_moves) < 5:
            return ["Game Theory Basics", "Random Strategy", "Move Distribution"]
        
        # Focus area 1: Entropy and randomness
        if len(human_moves) >= 8:
            move_counts = Counter(human_moves[-8:])
            max_usage = max(move_counts.values()) / 8
            if max_usage > 0.6:
                focus_areas.append("Entropy Optimization - Learn to balance randomness")
        
        # Focus area 2: Pattern recognition
        pattern_score = self._assess_pattern_awareness(game_state)
        if pattern_score < 0.4:
            focus_areas.append("Pattern Recognition - Learn to identify and break patterns")
        
        # Focus area 3: Game theory
        if len(results) >= 8:
            win_rate = results[-8:].count('win') / 8
            if win_rate < 0.35:
                focus_areas.append("Nash Equilibrium - Study optimal mixed strategies")
        
        # Focus area 4: AI-specific strategies
        ai_strategy = game_state.get('current_strategy', 'unknown')
        if ai_strategy != 'unknown' and ai_strategy != 'random':
            focus_areas.append(f"Counter-AI Strategies - Learn to beat {ai_strategy} algorithms")
        
        # Focus area 5: Psychological aspects
        frustration_data = self._detect_frustration_patterns(game_state)
        if (isinstance(frustration_data, dict) and 
            frustration_data.get('overall_frustration_score', 0) > 0.6):
            focus_areas.append("Emotional Control - Manage tilt and frustration")
        
        # Focus area 6: Advanced concepts
        if len(human_moves) >= 20:
            # For experienced players
            focus_areas.append("Information Theory - Advanced entropy concepts")
            focus_areas.append("Meta-Gaming - Thinking about opponent's thinking")
        
        # Always include fundamentals for new players
        if len(human_moves) < 10:
            focus_areas.insert(0, "RPS Fundamentals - Basic strategy principles")
        
        return focus_areas[:5]  # Return top 5 focus areas
    
    def _extract_model_performance_metrics(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract model performance metrics for LLM coaching insights"""
        human_moves = game_state.get('human_moves', game_state.get('human_history', []))
        robot_moves = game_state.get('robot_moves', game_state.get('robot_history', []))
        results = game_state.get('results', game_state.get('result_history', []))
        current_ai = game_state.get('current_strategy', 'unknown')
        
        if len(human_moves) < 3:
            return {'status': 'insufficient_data', 'rounds_needed': 3 - len(human_moves)}
        
        model_metrics = {}
        
        # Current AI model performance
        model_metrics['current_ai_model'] = {
            'name': current_ai,
            'difficulty_level': self._get_difficulty_level(current_ai),
            'win_rate_against_human': self._calculate_ai_win_rate(results),
            'prediction_accuracy': self._calculate_ai_prediction_accuracy(game_state),
            'adaptation_speed': self._calculate_ai_adaptation_speed(game_state)
        }
        
        # Performance against different model types (for strategic advice)
        model_performance_comparison = {}
        
        # Historical performance if available
        historical_performance = getattr(self, 'historical_performance', None)
        if historical_performance:
            model_performance_comparison = historical_performance
        else:
            # Estimate performance based on current patterns
            model_performance_comparison = self._estimate_performance_against_models(game_state)
        
        model_metrics['performance_vs_models'] = model_performance_comparison
        
        # Strategic recommendations based on model performance
        model_metrics['model_specific_advice'] = self._generate_model_specific_advice(current_ai, game_state)
        
        # Vulnerability analysis
        model_metrics['human_vulnerabilities'] = self._analyze_human_vulnerabilities_per_model(game_state)
        
        # Strengths analysis
        model_metrics['human_strengths'] = self._analyze_human_strengths_per_model(game_state)
        
        return model_metrics
    
    def _get_difficulty_level(self, ai_strategy: str) -> str:
        """Get difficulty level for AI strategy"""
        difficulty_map = {
            'random': 'Beginner',
            'frequency': 'Easy',
            'markov': 'Medium',
            'enhanced': 'Hard',
            'lstm': 'Expert'
        }
        return difficulty_map.get(ai_strategy, 'Unknown')
    
    def _calculate_ai_win_rate(self, results: List[str]) -> float:
        """Calculate AI's win rate against human"""
        if not results:
            return 0.0
        
        ai_wins = results.count('lose') + results.count('Loss')  # Human losses = AI wins
        return self._format_metric(ai_wins / len(results))
    
    def _calculate_ai_prediction_accuracy(self, game_state: Dict[str, Any]) -> float:
        """Calculate AI's prediction accuracy if available"""
        # This would require access to AI predictions vs actual human moves
        # For now, estimate based on AI strategy and human predictability
        current_ai = game_state.get('current_strategy', 'unknown')
        human_moves = game_state.get('human_moves', game_state.get('human_history', []))
        
        if len(human_moves) < 5:
            return 0.5
        
        # Estimate based on human predictability and AI sophistication
        move_counts = Counter(human_moves[-8:] if len(human_moves) >= 8 else human_moves)
        max_frequency = max(move_counts.values()) / len(human_moves[-8:] if len(human_moves) >= 8 else human_moves)
        
        # More sophisticated AIs can exploit predictability better
        ai_sophistication = {
            'random': 0.33,
            'frequency': 0.4 + (max_frequency - 0.33) * 0.5,
            'markov': 0.45 + (max_frequency - 0.33) * 0.7,
            'enhanced': 0.5 + (max_frequency - 0.33) * 0.8,
            'lstm': 0.55 + (max_frequency - 0.33) * 0.9
        }
        
        return self._format_metric(ai_sophistication.get(current_ai, 0.5))
    
    def _calculate_ai_adaptation_speed(self, game_state: Dict[str, Any]) -> str:
        """Calculate how quickly AI adapts to human patterns"""
        current_ai = game_state.get('current_strategy', 'unknown')
        
        adaptation_speeds = {
            'random': 'None (no adaptation)',
            'frequency': 'Slow (global frequency tracking)',
            'markov': 'Medium (pattern recognition)',
            'enhanced': 'Fast (advanced ML)',
            'lstm': 'Very Fast (neural learning)'
        }
        
        return adaptation_speeds.get(current_ai, 'Unknown')
    
    def _estimate_performance_against_models(self, game_state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Estimate human performance against different AI models"""
        human_moves = game_state.get('human_moves', game_state.get('human_history', []))
        results = game_state.get('results', game_state.get('result_history', []))
        
        if len(human_moves) < 5:
            return {
                'insufficient_data': {
                    'status': 'insufficient_data_for_estimation',
                    'rounds_needed': 5 - len(human_moves)
                }
            }
        
        # Analyze human characteristics
        move_entropy = self._calculate_move_entropy(human_moves)
        pattern_consistency = self._calculate_pattern_consistency(human_moves)
        adaptation_rate = self._calculate_human_adaptation_rate(game_state)
        
        # Estimate performance against each model type
        performance_estimates = {}
        
        # Against Random AI
        performance_estimates['random'] = {
            'estimated_win_rate': 0.33,  # Always 33% against pure random
            'confidence': 'high',
            'reasoning': 'Random AI has no strategy to exploit patterns'
        }
        
        # Against Frequency AI
        if move_entropy > 1.4:  # Good randomness
            freq_win_rate = 0.45
            freq_confidence = 'medium'
            freq_reasoning = 'Good move balance should perform well against frequency analysis'
        else:
            freq_win_rate = 0.25
            freq_confidence = 'high'
            freq_reasoning = 'Predictable moves will be exploited by frequency analysis'
        
        performance_estimates['frequency'] = {
            'estimated_win_rate': self._format_metric(freq_win_rate),
            'confidence': freq_confidence,
            'reasoning': freq_reasoning
        }
        
        # Against Markov AI
        if pattern_consistency < 0.4:  # Low pattern consistency = good against Markov
            markov_win_rate = 0.4
            markov_reasoning = 'Unpredictable sequences should confuse pattern matching'
        else:
            markov_win_rate = 0.2
            markov_reasoning = 'Consistent patterns will be learned by Markov chains'
        
        performance_estimates['markov'] = {
            'estimated_win_rate': self._format_metric(markov_win_rate),
            'confidence': 'medium',
            'reasoning': markov_reasoning
        }
        
        # Against Enhanced AI
        adaptability_score = adaptation_rate if adaptation_rate else 0.5
        if move_entropy > 1.3 and adaptability_score > 0.6:
            enhanced_win_rate = 0.35
            enhanced_reasoning = 'High entropy and adaptability should help against enhanced AI'
        else:
            enhanced_win_rate = 0.15
            enhanced_reasoning = 'Enhanced AI can exploit most human patterns'
        
        performance_estimates['enhanced'] = {
            'estimated_win_rate': self._format_metric(enhanced_win_rate),
            'confidence': 'low',
            'reasoning': enhanced_reasoning
        }
        
        # Against LSTM AI
        if move_entropy > 1.5 and pattern_consistency < 0.3:
            lstm_win_rate = 0.3
            lstm_reasoning = 'Maximum entropy and minimal patterns might confuse neural networks'
        else:
            lstm_win_rate = 0.1
            lstm_reasoning = 'LSTM networks excel at learning human behavioral patterns'
        
        performance_estimates['lstm'] = {
            'estimated_win_rate': self._format_metric(lstm_win_rate),
            'confidence': 'low',
            'reasoning': lstm_reasoning
        }
        
        return performance_estimates
    
    def _calculate_human_adaptation_rate(self, game_state: Dict[str, Any]) -> float:
        """Calculate how well human adapts to changing conditions"""
        results = game_state.get('results', game_state.get('result_history', []))
        human_moves = game_state.get('human_moves', game_state.get('human_history', []))
        
        if len(results) < 8:
            return 0.5
        
        # Count strategy changes after losses
        adaptations = 0
        loss_opportunities = 0
        
        for i in range(len(results) - 2):
            if results[i] in ['lose', 'Loss']:
                loss_opportunities += 1
                # Check if strategy changed in next 2 moves
                if (i + 2 < len(human_moves) and 
                    human_moves[i + 1] != human_moves[i]):
                    adaptations += 1
        
        if loss_opportunities == 0:
            return 0.5
        
        return self._format_metric(adaptations / loss_opportunities)
    
    def _generate_model_specific_advice(self, current_ai: str, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate advice specific to the current AI model"""
        advice = {}
        
        if current_ai == 'random':
            advice = {
                'primary_strategy': 'Play any strategy - random AI cannot be beaten consistently',
                'focus_areas': ['Practice different techniques', 'Experiment with patterns'],
                'win_rate_expectation': '33% (pure chance)',
                'difficulty_rating': 'Beginner'
            }
        
        elif current_ai == 'frequency':
            advice = {
                'primary_strategy': 'Balance your move distribution (aim for 33.3% each move)',
                'focus_areas': ['Move balance', 'Avoid favorite moves', 'Track your own frequency'],
                'win_rate_expectation': '40-50% with good balance',
                'difficulty_rating': 'Easy'
            }
        
        elif current_ai == 'markov':
            advice = {
                'primary_strategy': 'Break patterns and avoid predictable sequences',
                'focus_areas': ['Sequence randomization', 'Pattern awareness', 'Anti-pattern techniques'],
                'win_rate_expectation': '35-45% with good pattern breaking',
                'difficulty_rating': 'Medium'
            }
        
        elif current_ai == 'enhanced':
            advice = {
                'primary_strategy': 'Combine frequency balance with pattern breaking',
                'focus_areas': ['Advanced entropy', 'Meta-strategy', 'Counter-exploitation'],
                'win_rate_expectation': '30-40% with optimal play',
                'difficulty_rating': 'Hard'
            }
        
        elif current_ai == 'lstm':
            advice = {
                'primary_strategy': 'Maximum entropy and complete unpredictability',
                'focus_areas': ['Perfect randomness', 'Anti-learning techniques', 'Information theory'],
                'win_rate_expectation': '25-35% with expert play',
                'difficulty_rating': 'Expert'
            }
        
        else:
            advice = {
                'primary_strategy': 'Unknown AI - use balanced approach',
                'focus_areas': ['Observe AI behavior', 'Maintain flexibility'],
                'win_rate_expectation': 'Unknown',
                'difficulty_rating': 'Unknown'
            }
        
        return advice
    
    def _analyze_human_vulnerabilities_per_model(self, game_state: Dict[str, Any]) -> Dict[str, List[str]]:
        """Analyze human vulnerabilities against different AI models"""
        human_moves = game_state.get('human_moves', game_state.get('human_history', []))
        
        if len(human_moves) < 5:
            return {'insufficient_data': ['Need at least 5 moves for vulnerability analysis']}
        
        vulnerabilities = {}
        
        # Calculate key metrics
        move_entropy = self._calculate_move_entropy(human_moves)
        pattern_consistency = self._calculate_pattern_consistency(human_moves)
        move_distribution = Counter(human_moves)
        total_moves = len(human_moves)
        
        # Vulnerabilities against Frequency AI
        freq_vulns = []
        for move, count in move_distribution.items():
            frequency = count / total_moves
            if frequency > 0.4:
                freq_vulns.append(f"Overusing {move} ({frequency:.1%})")
        if not freq_vulns:
            freq_vulns.append("Good move balance - no major frequency vulnerabilities")
        vulnerabilities['frequency'] = freq_vulns
        
        # Vulnerabilities against Markov AI
        markov_vulns = []
        if pattern_consistency > 0.6:
            markov_vulns.append("Too consistent in patterns - sequences are predictable")
        if len(set(human_moves)) < 3:
            markov_vulns.append("Limited move variety makes patterns easier to learn")
        if not markov_vulns:
            markov_vulns.append("Good pattern variation - sequences are unpredictable")
        vulnerabilities['markov'] = markov_vulns
        
        # Vulnerabilities against Enhanced AI
        enhanced_vulns = []
        if move_entropy < 1.2:
            enhanced_vulns.append("Low entropy - move distribution is exploitable")
        if pattern_consistency > 0.5:
            enhanced_vulns.append("Pattern consistency allows for exploitation")
        if not enhanced_vulns:
            enhanced_vulns.append("High entropy and pattern breaking - well defended")
        vulnerabilities['enhanced'] = enhanced_vulns
        
        # Vulnerabilities against LSTM AI
        lstm_vulns = []
        if move_entropy < 1.4:
            lstm_vulns.append("Insufficient randomness - neural networks can learn patterns")
        if any(count / total_moves > 0.35 for count in move_distribution.values()):
            lstm_vulns.append("Move bias detected - neural network will exploit this")
        
        # Check for any behavioral patterns
        if len(human_moves) >= 10:
            recent_moves = human_moves[-10:]
            if len(set(recent_moves)) < 3:
                lstm_vulns.append("Limited recent variety - LSTM will predict this")
        
        if not lstm_vulns:
            lstm_vulns.append("Near-optimal entropy - well defended against neural learning")
        vulnerabilities['lstm'] = lstm_vulns
        
        return vulnerabilities
    
    def _analyze_human_strengths_per_model(self, game_state: Dict[str, Any]) -> Dict[str, List[str]]:
        """Analyze human strengths against different AI models"""
        human_moves = game_state.get('human_moves', game_state.get('human_history', []))
        results = game_state.get('results', game_state.get('result_history', []))
        
        if len(human_moves) < 5:
            return {'insufficient_data': ['Need at least 5 moves for strength analysis']}
        
        strengths = {}
        
        # Calculate key metrics
        move_entropy = self._calculate_move_entropy(human_moves)
        pattern_consistency = self._calculate_pattern_consistency(human_moves)
        adaptation_rate = self._calculate_human_adaptation_rate(game_state)
        
        # Strengths against Frequency AI
        freq_strengths = []
        move_distribution = Counter(human_moves)
        balance_score = 1 - max(move_distribution.values()) / len(human_moves)
        if balance_score > 0.6:
            freq_strengths.append("Excellent move balance - frequency analysis cannot exploit")
        if move_entropy > 1.3:
            freq_strengths.append("High entropy confuses frequency-based predictions")
        if not freq_strengths:
            freq_strengths.append("Room for improvement in move balance")
        strengths['frequency'] = freq_strengths
        
        # Strengths against Markov AI
        markov_strengths = []
        if pattern_consistency < 0.4:
            markov_strengths.append("Excellent pattern breaking - sequences are unpredictable")
        if len(set(human_moves)) == 3:
            markov_strengths.append("Full move variety prevents pattern learning")
        if adaptation_rate and adaptation_rate > 0.6:
            markov_strengths.append("Good adaptation prevents Markov exploitation")
        if not markov_strengths:
            markov_strengths.append("Patterns may be too predictable for Markov AI")
        strengths['markov'] = markov_strengths
        
        # Strengths against Enhanced AI
        enhanced_strengths = []
        if move_entropy > 1.4:
            enhanced_strengths.append("High entropy - excellent defense against ML models")
        if pattern_consistency < 0.3:
            enhanced_strengths.append("Minimal patterns - enhanced AI cannot find exploits")
        if adaptation_rate and adaptation_rate > 0.7:
            enhanced_strengths.append("Superior adaptation - can counter AI learning")
        if not enhanced_strengths:
            enhanced_strengths.append("Enhanced AI can exploit current patterns")
        strengths['enhanced'] = enhanced_strengths
        
        # Strengths against LSTM AI
        lstm_strengths = []
        if move_entropy > 1.5:
            lstm_strengths.append("Near-maximum entropy - optimal defense against neural networks")
        if pattern_consistency < 0.2:
            lstm_strengths.append("Minimal behavioral patterns - LSTM cannot learn effectively")
        
        # Check for anti-learning behavior
        if len(human_moves) >= 15:
            recent_entropy = self._calculate_move_entropy(human_moves[-10:])
            if recent_entropy > 1.4:
                lstm_strengths.append("Maintaining high recent entropy - prevents LSTM adaptation")
        
        if not lstm_strengths:
            lstm_strengths.append("LSTM networks can exploit current behavioral patterns")
        strengths['lstm'] = lstm_strengths
        
        return strengths
    
    def _calculate_move_entropy(self, moves: List[str]) -> float:
        """Calculate Shannon entropy of move distribution"""
        if not moves:
            return 0.0
        
        move_counts = Counter(moves)
        total = len(moves)
        entropy = 0
        
        for count in move_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        
        return self._format_metric(entropy)
    
    def _calculate_pattern_consistency(self, moves: List[str]) -> float:
        """Calculate how consistent patterns are in the move sequence"""
        if len(moves) < 6:
            return 0.0
        
        pattern_matches = 0
        total_patterns = 0
        
        # Look for 2-move patterns
        for i in range(len(moves) - 3):
            pattern = moves[i:i+2]
            expected_next = pattern[0]  # If pattern repeats
            
            # Look for this pattern later
            for j in range(i+2, len(moves)-1):
                if moves[j:j+2] == pattern:
                    total_patterns += 1
                    if j+2 < len(moves) and moves[j+2] == expected_next:
                        pattern_matches += 1
                    break
        
        if total_patterns == 0:
            return 0.0
        
        return self._format_metric(pattern_matches / total_patterns)
    
    def _count_total_metrics(self) -> int:
        """Count total number of metrics being tracked"""
        return 85  # Updated count with all new metrics
    
    def get_realtime_metrics(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get metrics suitable for real-time coaching during gameplay
        Fast computation, focused on immediate strategic advice
        """
        human_moves = game_state.get('human_moves', game_state.get('human_history', []))
        robot_moves = game_state.get('robot_moves', game_state.get('robot_history', []))
        results = game_state.get('results', game_state.get('result_history', []))
        
        if len(human_moves) < 2:
            return {'status': 'insufficient_data', 'rounds_needed': 2 - len(human_moves)}
        
        realtime_metrics = {}
        
        # Core game state (always available)
        realtime_metrics['current_state'] = {
            'total_rounds': len(human_moves),
            'current_streak': self._calculate_current_streak(results),
            'recent_performance': self._calculate_win_rates(results[-5:] if len(results) >= 5 else results),
            'current_ai': game_state.get('current_strategy', 'unknown')
        }
        
        # Quick move analysis (last 3-5 moves)
        recent_moves = human_moves[-5:] if len(human_moves) >= 5 else human_moves
        realtime_metrics['recent_patterns'] = {
            'last_move': human_moves[-1] if human_moves else None,
            'move_frequency': dict(Counter(recent_moves)),
            'is_repeating': len(set(recent_moves[-3:])) == 1 if len(recent_moves) >= 3 else False,
            'pattern_detected': self._detect_simple_pattern(recent_moves)
        }
        
        # Quick strategic advice
        realtime_metrics['immediate_advice'] = self._generate_immediate_advice(game_state)
        
        # Performance against current AI (fast calculation)
        current_ai = game_state.get('current_strategy', 'unknown')
        realtime_metrics['vs_current_ai'] = {
            'ai_type': current_ai,
            'difficulty': self._get_difficulty_level(current_ai),
            'recommended_focus': self._get_quick_ai_advice(current_ai, recent_moves)
        }
        
        # Simple entropy calculation (for immediate randomness feedback)
        realtime_metrics['randomness_check'] = {
            'recent_entropy': self._calculate_move_entropy(recent_moves),
            'entropy_score': 'Good' if self._calculate_move_entropy(recent_moves) > 1.2 else 'Needs Improvement',
            'balance_score': self._calculate_quick_balance_score(recent_moves)
        }
        
        return realtime_metrics
    
    def get_postgame_metrics(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get comprehensive metrics for post-game behavioral analysis
        Detailed computation, focused on learning and improvement
        """
        human_moves = game_state.get('human_moves', game_state.get('human_history', []))
        
        if len(human_moves) < 5:
            return {'status': 'insufficient_data', 'rounds_needed': 5 - len(human_moves)}
        
        # Get full comprehensive metrics for deep analysis
        comprehensive_metrics = self.aggregate_comprehensive_metrics(game_state)
        
        # Add post-game specific analysis
        postgame_analysis = {}
        
        # Behavioral evolution over time
        postgame_analysis['behavioral_evolution'] = self._analyze_behavioral_evolution(game_state)
        
        # Learning patterns and adaptation
        postgame_analysis['learning_analysis'] = self._analyze_learning_patterns(game_state)
        
        # Strategic mistakes and missed opportunities
        postgame_analysis['strategic_review'] = self._analyze_strategic_mistakes(game_state)
        
        # Psychological profile development
        postgame_analysis['psychological_development'] = self._analyze_psychological_development(game_state)
        
        # Performance comparison across sessions
        postgame_analysis['session_comparison'] = self._analyze_session_performance(game_state)
        
        # Advanced pattern recognition
        postgame_analysis['advanced_patterns'] = self._analyze_advanced_patterns(game_state)
        
        # Future improvement roadmap
        postgame_analysis['improvement_roadmap'] = self._generate_improvement_roadmap(game_state)
        
        return {
            'comprehensive_data': comprehensive_metrics,
            'postgame_analysis': postgame_analysis,
            'analysis_type': 'post_game',
            'analysis_depth': 'comprehensive'
        }
    
    def _calculate_current_streak(self, results: List[str]) -> Dict[str, Any]:
        """Calculate current win/loss streak"""
        if not results:
            return {'type': 'none', 'length': 0}
        
        current_result = results[-1]
        streak_length = 1
        
        for i in range(len(results) - 2, -1, -1):
            if results[i] == current_result:
                streak_length += 1
            else:
                break
        
        return {
            'type': current_result,
            'length': streak_length,
            'is_significant': streak_length >= 3
        }
    
    def _detect_simple_pattern(self, moves: List[str]) -> Dict[str, Any]:
        """Detect simple patterns for real-time feedback"""
        if len(moves) < 3:
            return {'detected': False, 'type': 'insufficient_data'}
        
        # Check for alternating pattern
        if len(moves) >= 4:
            if moves[-1] == moves[-3] and moves[-2] == moves[-4]:
                return {'detected': True, 'type': 'alternating', 'confidence': 'high'}
        
        # Check for repetition
        if len(set(moves[-3:])) == 1:
            return {'detected': True, 'type': 'repetition', 'confidence': 'high'}
        
        # Check for cycle
        if len(moves) >= 6:
            if moves[-3:] == moves[-6:-3]:
                return {'detected': True, 'type': 'cycle', 'confidence': 'medium'}
        
        return {'detected': False, 'type': 'random'}
    
    def _generate_immediate_advice(self, game_state: Dict[str, Any]) -> List[str]:
        """Generate quick advice for real-time coaching"""
        advice = []
        
        human_moves = game_state.get('human_moves', game_state.get('human_history', []))
        results = game_state.get('results', game_state.get('result_history', []))
        
        if len(human_moves) < 3:
            return ["Play more rounds to get personalized advice"]
        
        # Check recent moves
        recent_moves = human_moves[-3:]
        
        # Repetition warning
        if len(set(recent_moves)) == 1:
            advice.append(f"⚠️ Avoid repeating {recent_moves[0]} - AI will exploit this!")
        
        # Alternating pattern warning
        elif len(human_moves) >= 4 and human_moves[-1] == human_moves[-3]:
            advice.append("⚠️ Breaking alternating patterns - mix it up more!")
        
        # Performance-based advice
        if len(results) >= 3:
            recent_results = results[-3:]
            if recent_results.count('lose') >= 2:
                advice.append("💡 Try a completely different strategy - AI is predicting you")
            elif recent_results.count('win') >= 2:
                advice.append("✅ Good strategy! But stay unpredictable")
        
        # Entropy advice
        entropy = self._calculate_move_entropy(recent_moves)
        if entropy < 1.0:
            advice.append("🎲 Increase randomness - be more unpredictable!")
        
        return advice[:3]  # Limit to 3 quick tips
    
    def _get_quick_ai_advice(self, ai_type: str, recent_moves: List[str]) -> str:
        """Quick advice specific to AI type"""
        advice_map = {
            'random': "No specific strategy needed - pure chance",
            'frequency': "Balance your move distribution equally",
            'markov': "Break patterns and avoid sequences",
            'enhanced': "Maximum randomness and pattern breaking",
            'lstm': "Perfect entropy - be completely unpredictable"
        }
        return advice_map.get(ai_type, "Stay flexible and observe AI behavior")
    
    def _calculate_quick_balance_score(self, moves: List[str]) -> str:
        """Quick balance assessment"""
        if len(moves) < 3:
            return "Insufficient data"
        
        counts = Counter(moves)
        max_freq = max(counts.values()) / len(moves)
        
        if max_freq <= 0.4:
            return "Good balance"
        elif max_freq <= 0.6:
            return "Slight bias"
        else:
            return "Heavily biased"
    
    def _analyze_behavioral_evolution(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how behavior changed over the session"""
        human_moves = game_state.get('human_moves', game_state.get('human_history', []))
        
        if len(human_moves) < 10:
            return {'status': 'insufficient_data'}
        
        # Analyze evolution in chunks
        chunk_size = len(human_moves) // 3
        early_chunk = human_moves[:chunk_size]
        middle_chunk = human_moves[chunk_size:2*chunk_size]
        late_chunk = human_moves[-chunk_size:]
        
        evolution = {
            'entropy_evolution': {
                'early': self._calculate_move_entropy(early_chunk),
                'middle': self._calculate_move_entropy(middle_chunk),
                'late': self._calculate_move_entropy(late_chunk)
            },
            'pattern_evolution': {
                'early': self._calculate_pattern_consistency(early_chunk),
                'middle': self._calculate_pattern_consistency(middle_chunk),
                'late': self._calculate_pattern_consistency(late_chunk)
            },
            'trend_direction': 'improving' if self._calculate_move_entropy(late_chunk) > self._calculate_move_entropy(early_chunk) else 'declining'
        }
        
        return evolution
    
    def _analyze_learning_patterns(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze learning and adaptation patterns"""
        results = game_state.get('results', game_state.get('result_history', []))
        human_moves = game_state.get('human_moves', game_state.get('human_history', []))
        
        learning_analysis = {
            'adaptation_speed': self._calculate_human_adaptation_rate(game_state),
            'mistake_repetition': self._analyze_mistake_repetition(game_state),
            'strategy_experimentation': self._analyze_strategy_experimentation(human_moves),
            'feedback_responsiveness': self._analyze_feedback_responsiveness(game_state)
        }
        
        return learning_analysis
    
    def _analyze_strategic_mistakes(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze strategic mistakes and missed opportunities"""
        return {
            'frequency_mistakes': self._identify_frequency_mistakes(game_state),
            'pattern_mistakes': self._identify_pattern_mistakes(game_state),
            'missed_opportunities': self._identify_missed_opportunities(game_state),
            'exploitation_points': self._identify_exploitation_points(game_state)
        }
    
    def _analyze_psychological_development(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze psychological development throughout session"""
        return {
            'confidence_progression': self._analyze_confidence_progression(game_state),
            'frustration_management': self._analyze_frustration_management(game_state),
            'risk_tolerance_evolution': self._analyze_risk_tolerance_evolution(game_state),
            'tilt_recovery': self._analyze_tilt_recovery(game_state)
        }
    
    def _analyze_session_performance(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Compare performance across different parts of session"""
        results = game_state.get('results', game_state.get('result_history', []))
        
        if len(results) < 10:
            return {'status': 'insufficient_data'}
        
        # Split into quarters
        quarter_size = len(results) // 4
        quarters = [
            results[:quarter_size],
            results[quarter_size:2*quarter_size],
            results[2*quarter_size:3*quarter_size],
            results[3*quarter_size:]
        ]
        
        performance_comparison = {}
        for i, quarter in enumerate(quarters, 1):
            if quarter:
                win_rate = quarter.count('win') / len(quarter)
                performance_comparison[f'quarter_{i}'] = {
                    'win_rate': self._format_metric(win_rate),
                    'total_games': len(quarter)
                }
        
        return performance_comparison
    
    def _analyze_advanced_patterns(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced pattern recognition for post-game analysis"""
        return {
            'markov_chain_analysis': self._analyze_markov_patterns(game_state),
            'frequency_domain_analysis': self._analyze_frequency_patterns(game_state),
            'complexity_analysis': self._analyze_move_complexity(game_state),
            'predictability_analysis': self._analyze_predictability_metrics(game_state)
        }
    
    def _generate_improvement_roadmap(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive improvement roadmap"""
        return {
            'immediate_focus': self._identify_immediate_improvements(game_state),
            'medium_term_goals': self._identify_medium_term_goals(game_state),
            'advanced_techniques': self._identify_advanced_techniques(game_state),
            'practice_recommendations': self._generate_practice_recommendations(game_state)
        }
    
    # Placeholder implementations for detailed analysis methods
    def _analyze_mistake_repetition(self, game_state: Dict[str, Any]) -> float:
        """Analyze how often mistakes are repeated"""
        return 0.3  # Placeholder
    
    def _analyze_strategy_experimentation(self, moves: List[str]) -> float:
        """Analyze how much strategy experimentation occurs"""
        return 0.7  # Placeholder
    
    def _analyze_feedback_responsiveness(self, game_state: Dict[str, Any]) -> float:
        """Analyze responsiveness to feedback"""
        return 0.6  # Placeholder
    
    def _identify_frequency_mistakes(self, game_state: Dict[str, Any]) -> List[str]:
        """Identify frequency-related mistakes"""
        return ["Overusing rock", "Underusing scissors"]  # Placeholder
    
    def _identify_pattern_mistakes(self, game_state: Dict[str, Any]) -> List[str]:
        """Identify pattern-related mistakes"""
        return ["Alternating too regularly", "Predictable sequences"]  # Placeholder
    
    def _identify_missed_opportunities(self, game_state: Dict[str, Any]) -> List[str]:
        """Identify missed strategic opportunities"""
        return ["Could have exploited AI pattern", "Missed counter-strategy"]  # Placeholder
    
    def _identify_exploitation_points(self, game_state: Dict[str, Any]) -> List[str]:
        """Identify points where player was exploited"""
        return ["Round 5-8: Pattern exploitation", "Round 12-15: Frequency bias"]  # Placeholder
    
    def _analyze_confidence_progression(self, game_state: Dict[str, Any]) -> Dict[str, float]:
        """Analyze confidence progression"""
        return {"start": 0.6, "middle": 0.4, "end": 0.7}  # Placeholder
    
    def _analyze_frustration_management(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze frustration management"""
        return {"tilt_episodes": 2, "recovery_speed": "fast"}  # Placeholder
    
    def _analyze_risk_tolerance_evolution(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze risk tolerance evolution"""
        return {"trend": "increasing", "stability": "moderate"}  # Placeholder
    
    def _analyze_tilt_recovery(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze tilt recovery patterns"""
        return {"recovery_rounds": 3, "effectiveness": "good"}  # Placeholder
    
    def _analyze_markov_patterns(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Markov chain patterns"""
        return {"order_detected": 2, "predictability": 0.4}  # Placeholder
    
    def _analyze_frequency_patterns(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze frequency domain patterns"""
        return {"dominant_frequency": "rock", "bias_strength": 0.3}  # Placeholder
    
    def _analyze_move_complexity(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze move sequence complexity"""
        return {"complexity_score": 0.65, "trend": "increasing"}  # Placeholder
    
    def _analyze_predictability_metrics(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze predictability metrics"""
        return {"short_term": 0.4, "long_term": 0.3, "trend": "improving"}  # Placeholder
    
    def _identify_immediate_improvements(self, game_state: Dict[str, Any]) -> List[str]:
        """Identify immediate improvement areas"""
        return ["Reduce rock bias", "Break alternating patterns"]  # Placeholder
    
    def _identify_medium_term_goals(self, game_state: Dict[str, Any]) -> List[str]:
        """Identify medium-term goals"""
        return ["Master entropy management", "Develop anti-Markov techniques"]  # Placeholder
    
    def _identify_advanced_techniques(self, game_state: Dict[str, Any]) -> List[str]:
        """Identify advanced techniques to learn"""
        return ["Information theory application", "Counter-exploitation strategies"]  # Placeholder
    
    def _generate_practice_recommendations(self, game_state: Dict[str, Any]) -> List[str]:
        """Generate specific practice recommendations"""
        return ["Practice against frequency AI", "Study entropy maximization"]  # Placeholder


# Global instance for use across the application
ai_metrics_aggregator = AICoachMetricsAggregator()

def get_metrics_aggregator():
    """Get global metrics aggregator instance"""
    return ai_metrics_aggregator