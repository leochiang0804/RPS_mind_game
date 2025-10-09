"""
Centralized Game Context Builder
===============================

This module provides a single source of truth for building game context data
used across all AI coach endpoints, analytics, and UI components.

The goal is to eliminate scattered data construction and ensure consistency
across the entire application.

Now routes all AI interactions through the adaptive opponent engine designed for richer difficulty, strategy, and personality behaviours.
"""

import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field


@dataclass
class GameContextConfig:
    """Configuration for game context building"""
    context_type: str = 'full'  # 'realtime', 'comprehensive', 'analytics', 'full'
    include_ai_behavior: bool = True
    include_advanced_metrics: bool = True
    include_temporal_data: bool = True
    validate_schema: bool = True


# 42-Opponent System Integration
_current_opponent_params = None
_ai_system_metadata = {}

_PERSONALITY_ALIAS_MAP = {
    'aggressive': 'berserker',
    'defensive': 'guardian',
    'unpredictable': 'wildcard',
    'cautious': 'professor',
    'confident': 'mirror',
    'chameleon': 'chameleon',
    # Neutral intentionally omitted
}


def set_opponent_parameters(difficulty: str, strategy: str, personality: str) -> bool:
    """
    Set current opponent parameters for the adaptive opponent engine.
    
    Args:
        difficulty: 'rookie', 'challenger', 'master', or 'grandmaster'
        strategy: 'to_win' or 'not_to_lose'
        personality: 'neutral', 'aggressive', 'defensive', etc.
        
    Returns:
        True if parameters were set successfully
    """
    global _current_opponent_params, _ai_system_metadata
    
    try:
        # Import here to avoid circular imports
        from rps_ai_system import get_ai_system
        
        ai_system = get_ai_system()

        # Avoid resetting the opponent if configuration hasn't changed
        if (
            _current_opponent_params
            and _current_opponent_params.get('difficulty') == difficulty
            and _current_opponent_params.get('strategy') == strategy
            and _current_opponent_params.get('personality') == personality
        ):
            return True

        success = ai_system.set_opponent(difficulty, strategy, personality)
        
        if success:
            opponent_info = ai_system.get_opponent_info()
            _current_opponent_params = {
                'difficulty': opponent_info.get('difficulty', difficulty),
                'strategy': opponent_info.get('strategy', strategy),
                'personality': opponent_info.get('personality', personality),
                'opponent_info': opponent_info,
                'set_time': time.time(),
                'status': 'active',
            }
            
            _ai_system_metadata = {
                'system_type': 'adaptive_opponent_engine',
                'version': '2.0',
                'last_reset': time.time()
            }

            # Sync advanced personality engine (if available) with mapped name
            try:
                from personality_engine import get_personality_engine
                personality_engine = get_personality_engine()
                alias = _PERSONALITY_ALIAS_MAP.get(personality)
                if alias:
                    try:
                        personality_engine.set_personality(alias)
                    except ValueError as alias_error:
                        print(f"Warning: personality alias '{alias}' unavailable ({alias_error})")
                elif personality == 'neutral':
                    # Reset to a neutral baseline by clearing current personality
                    personality_engine.current_personality = None
            except Exception as exc:
                print(f"Warning: Unable to sync personality engine: {exc}")
            
        return success
        
    except Exception as e:
        print(f"Error setting opponent parameters: {e}")
        return False


def get_current_opponent_info() -> Dict[str, Any]:
    """Get current opponent information."""
    global _current_opponent_params
    
    if _current_opponent_params is None:
        return {
            'status': 'not_set',
            'difficulty': 'unknown',
            'strategy': 'unknown',
            'personality': 'unknown'
        }
        
    return _current_opponent_params.copy()


def get_ai_prediction(session_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get AI prediction using the adaptive opponent engine.
    
    Args:
        session_data: Current session data with move history
        
    Returns:
        Dictionary with prediction data
    """
    global _current_opponent_params
    
    try:
        # Import here to avoid circular imports
        from rps_ai_system import get_ai_system
        
        ai_system = get_ai_system()
        
        # Get move history from session
        human_moves = session_data.get('human_moves', [])
        
        # Convert outcome history from results
        results = session_data.get('results', [])
        outcome_history = []
        for result in results:
            if result == 'human':
                outcome_history.append('win')
            elif result == 'robot':
                outcome_history.append('loss')
            else:
                outcome_history.append('tie')
        
        # Get prediction
        if len(human_moves) > 0:
            probabilities, ai_move, metadata = ai_system.predict_next_move(
                human_moves, outcome_history
            )
        else:
            probabilities, ai_move, metadata = ai_system.predict_next_move()
        
        # Round probabilities to 4 decimals
        rounded_probabilities = [round(float(p), 4) for p in probabilities.tolist()]
        rounded_confidence = round(float(metadata.get('confidence', 0.33)), 4)
            
        return {
            'ai_move': ai_move,
            'human_probabilities': rounded_probabilities,
            'confidence': rounded_confidence,
            'metadata': metadata,
            'system_type': 'adaptive_opponent_engine'
        }
        
    except Exception as e:
        print(f"Error getting AI prediction: {e}")
        # Fallback to random
        import random
        return {
            'ai_move': random.choice(['rock', 'paper', 'scissors']),
            'human_probabilities': [0.3333, 0.3333, 0.3334],  # Rounded to 4 decimals, sums to 1
            'confidence': 0.3333,  # Rounded to 4 decimals
            'metadata': {'error': str(e)},
            'system_type': 'fallback_random'
        }


def update_ai_with_result(human_move: str, ai_move: str) -> Dict[str, Any]:
    """
    Update AI system with game result.
    
    Args:
        human_move: Human's move
        ai_move: AI's move
        
    Returns:
        Update metadata
    """
    try:
        # Import here to avoid circular imports
        from rps_ai_system import get_ai_system
        
        ai_system = get_ai_system()
        return ai_system.update_with_human_move(human_move, ai_move)
        
    except Exception as e:
        print(f"Error updating AI system: {e}")
        return {'error': str(e)}


def get_ai_performance_summary() -> Dict[str, Any]:
    """Get AI performance summary for current session."""
    try:
        # Import here to avoid circular imports
        from rps_ai_system import get_ai_system
        
        ai_system = get_ai_system()
        return ai_system.get_performance_summary()
        
    except Exception as e:
        print(f"Error getting AI performance summary: {e}")
        return {'error': str(e), 'status': 'unavailable'}


def reset_ai_system() -> None:
    """Reset AI system for new game."""
    global _ai_system_metadata
    
    try:
        # Import here to avoid circular imports
        from rps_ai_system import get_ai_system
        
        ai_system = get_ai_system()
        ai_system.reset_game_state()
        
        _ai_system_metadata['last_reset'] = time.time()
        
    except Exception as e:
        print(f"Error resetting AI system: {e}")


class GameContextBuilder:
    """
    Centralized game context builder with validation and optimization
    
    This class consolidates all the scattered game data construction logic
    from various endpoints into a single, consistent, and testable implementation.
    """
    
    def __init__(self):
        self.build_count = 0  # For performance monitoring
    
    @staticmethod
    def _round_metric(value, decimals=4):
        """
        Round metric values to specified decimal places to save memory.
        
        Args:
            value: The numeric value to round
            decimals: Number of decimal places (default: 4)
            
        Returns:
            Rounded value or original value if not numeric
        """
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return round(float(value), decimals)
        return value
    
    @staticmethod
    def _round_metrics_in_dict(data, decimals=4):
        """
        Recursively round all numeric values in a dictionary to specified decimal places.
        
        Args:
            data: Dictionary or list to process
            decimals: Number of decimal places (default: 4)
            
        Returns:
            Data with all metrics rounded
        """
        if isinstance(data, dict):
            return {key: GameContextBuilder._round_metrics_in_dict(value, decimals) 
                   for key, value in data.items()}
        elif isinstance(data, list):
            return [GameContextBuilder._round_metrics_in_dict(item, decimals) 
                   for item in data]
        else:
            return GameContextBuilder._round_metric(data, decimals)
    
    @staticmethod
    def calculate_metrics(human_moves, robot_moves, results, confidence_history=None, personality_modified_confidence_history=None):
        """
        Unified metric calculation - single source of truth for all game metrics.
        This ensures consistency between what's displayed and what's saved.
        
        Args:
            human_moves: List of human moves
            robot_moves: List of robot moves  
            results: List of game results
            confidence_history: List of AI confidence scores for each round
            personality_modified_confidence_history: List of personality-modified confidence scores for each round
        """
        from collections import Counter
        
        # Win/tie counts
        human_wins = results.count('human')
        robot_wins = results.count('robot')
        ties = results.count('tie')
        total_rounds = len(results) if results else 0
        human_win_rate = GameContextBuilder._round_metric(human_wins / total_rounds * 100 if total_rounds else 0.0)
        robot_win_rate = GameContextBuilder._round_metric(robot_wins / total_rounds * 100 if total_rounds else 0.0)
        tie_rate = GameContextBuilder._round_metric(ties / total_rounds * 100 if total_rounds else 0.0)

        # Longest streaks
        def longest_streak(target):
            max_streak = streak = 0
            for r in results:
                if r == target:
                    streak += 1
                    max_streak = max(max_streak, streak)
                else:
                    streak = 0
            return max_streak
        longest_human_streak = longest_streak('human')
        longest_robot_streak = longest_streak('robot')

        # Most common move
        most_common_move = Counter(human_moves).most_common(1)[0][0] if human_moves else None

        # Recent win rate (last 10 rounds)
        recent_results = results[-10:]
        recent_human_wins = recent_results.count('human')
        recent_win_rate = GameContextBuilder._round_metric(recent_human_wins / len(recent_results) * 100 if recent_results else 0.0)

        # Score differential
        score_differential = human_wins - robot_wins

        # AI confidence tracking
        confidence_score_history = list(confidence_history or [])
        if total_rounds and len(confidence_score_history) > total_rounds:
            confidence_score_history = confidence_score_history[-total_rounds:]
        
        # Round all confidence values to 4 decimals
        confidence_score_history = [GameContextBuilder._round_metric(c) for c in confidence_score_history]
        
        # Calculate confidence statistics
        if confidence_score_history:
            current_confidence_score = int(confidence_score_history[-1] * 100)  # Convert to percentage as int
            max_confidence_score = int(max(confidence_score_history) * 100)
            min_confidence_score = int(min(confidence_score_history) * 100)
            avg_confidence_score = int(sum(confidence_score_history) / len(confidence_score_history) * 100)
        else:
            current_confidence_score = 33  # Default confidence as percentage
            max_confidence_score = 33
            min_confidence_score = 33
            avg_confidence_score = 33

        # Personality-modified confidence tracking
        personality_modified_confidence_score_history = list(personality_modified_confidence_history or [])
        if total_rounds and len(personality_modified_confidence_score_history) > total_rounds:
            personality_modified_confidence_score_history = personality_modified_confidence_score_history[-total_rounds:]
        
        # Round all personality-modified confidence values to 4 decimals
        personality_modified_confidence_score_history = [GameContextBuilder._round_metric(c) for c in personality_modified_confidence_score_history]
        
        # Calculate personality-modified confidence statistics
        if personality_modified_confidence_score_history:
            current_confidence_score_modified_by_personality = int(personality_modified_confidence_score_history[-1] * 100)
            max_confidence_score_modified_by_personality = int(max(personality_modified_confidence_score_history) * 100)
            min_confidence_score_modified_by_personality = int(min(personality_modified_confidence_score_history) * 100)
            avg_confidence_score_modified_by_personality = int(sum(personality_modified_confidence_score_history) / len(personality_modified_confidence_score_history) * 100)
        else:
            # For random/frequency models or when personality modification isn't applicable
            current_confidence_score_modified_by_personality = current_confidence_score
            max_confidence_score_modified_by_personality = max_confidence_score
            min_confidence_score_modified_by_personality = min_confidence_score
            avg_confidence_score_modified_by_personality = avg_confidence_score

        # Predictability score (move variance)
        def calculate_move_variance(move_history):
            if not move_history or len(move_history) < 3:
                return 0.0
            move_counts = {'paper': 0, 'rock': 0, 'scissors': 0}
            for move in move_history:
                normalized = move.lower() if move else ''
                if normalized in move_counts:
                    move_counts[normalized] += 1
            total = len(move_history)
            expected_freq = total / 3
            variance = sum((count - expected_freq) ** 2 for count in move_counts.values()) / 3
            return GameContextBuilder._round_metric(min(100, (variance / expected_freq) * 50))
        predictability_score = calculate_move_variance(human_moves)

        # Recent momentum (last 10 moves)
        recent_moves = human_moves[-10:]
        recent_bias_type = None
        recent_bias_percent = None
        if recent_moves:
            move_counts = Counter([m for m in recent_moves if m])
            if move_counts:
                bias_type, bias_count = move_counts.most_common(1)[0]
                percent = GameContextBuilder._round_metric((bias_count / len(recent_moves)) * 100)
                recent_bias_type = bias_type
                recent_bias_percent = percent

        # SBC-specific emotional and strategic indicators
        sbc_metrics = GameContextBuilder._calculate_sbc_metrics(
            human_moves, robot_moves, results, confidence_score_history
        )

        # Round all metrics in the final return dictionary
        return GameContextBuilder._round_metrics_in_dict({
            'full_game_snapshot': {
                'human_moves': human_moves,
                'robot_moves': robot_moves,
                'results': results,
            },
            'recent_momentum': {
                'last_10': recent_moves,
                'recent_bias_type': recent_bias_type,
                'recent_bias_percent': recent_bias_percent,
            },
            'predictability_score': predictability_score,
            'human_wins': human_wins,
            'robot_wins': robot_wins,
            'ties': ties,
            'human_win_rate': human_win_rate,
            'robot_win_rate': robot_win_rate,
            'tie_rate': tie_rate,
            'longest_human_streak': longest_human_streak,
            'longest_robot_streak': longest_robot_streak,
            'most_common_move': most_common_move,
            'recent_win_rate': recent_win_rate,
            'score_differential': score_differential,
            'current_confidence_score': current_confidence_score,
            'max_confidence_score': max_confidence_score,
            'min_confidence_score': min_confidence_score,
            'avg_confidence_score': avg_confidence_score,
            'confidence_score_history': confidence_score_history,
            # Personality-modified confidence metrics
            'current_confidence_score_modified_by_personality': current_confidence_score_modified_by_personality,
            'max_confidence_score_modified_by_personality': max_confidence_score_modified_by_personality,
            'min_confidence_score_modified_by_personality': min_confidence_score_modified_by_personality,
            'avg_confidence_score_modified_by_personality': avg_confidence_score_modified_by_personality,
            'confidence_score_modified_by_personality_history': personality_modified_confidence_score_history,
            # SBC-specific metrics for banter and coaching
            'sbc_metrics': sbc_metrics,
        })
    
    @staticmethod
    def _calculate_sbc_metrics(human_moves, robot_moves, results, confidence_history):
        """
        Calculate SBC-specific metrics for enhanced banter and coaching.
        
        Args:
            human_moves: List of human moves
            robot_moves: List of robot moves
            results: List of game results  
            confidence_history: List of AI confidence scores
            
        Returns:
            Dictionary with SBC-specific emotional and strategic indicators
        """
        total_rounds = len(results) if results else 0
        
        # Emotional Context Indicators
        frustration_level = GameContextBuilder._calculate_frustration_level(results)
        momentum_state = GameContextBuilder._calculate_momentum_state(results)
        learning_trend = GameContextBuilder._calculate_learning_trend(results, confidence_history)
        
        # Strategic Analysis
        pattern_exploitation_success = GameContextBuilder._calculate_pattern_exploitation(
            human_moves, robot_moves, results
        )
        adaptation_speed = GameContextBuilder._calculate_adaptation_speed(results)
        consistency_score = GameContextBuilder._calculate_consistency_score(human_moves)
        
        # Performance Context
        performance_tier = GameContextBuilder._calculate_performance_tier(results)
        challenge_readiness = GameContextBuilder._calculate_challenge_readiness(
            results, confidence_history
        )
        
        # Round all SBC metrics before returning
        return GameContextBuilder._round_metrics_in_dict({
            'emotional_context': {
                'frustration_level': frustration_level,  # 0-1 scale
                'momentum_state': momentum_state,  # 'hot', 'cold', 'neutral'
                'learning_trend': learning_trend,  # 'improving', 'declining', 'stable'
            },
            'strategic_analysis': {
                'pattern_exploitation_success': pattern_exploitation_success,  # 0-1 scale
                'adaptation_speed': adaptation_speed,  # 'fast', 'medium', 'slow'
                'consistency_score': consistency_score,  # 0-1 scale
            },
            'performance_context': {
                'performance_tier': performance_tier,  # 'beginner', 'intermediate', 'advanced'
                'challenge_readiness': challenge_readiness,  # 0-1 scale
                'total_rounds': total_rounds,
            }
        })
    
    @staticmethod
    def _calculate_frustration_level(results):
        """Calculate player frustration based on recent losses and streaks"""
        if not results or len(results) < 3:
            return 0.0
            
        recent_results = results[-10:]  # Last 10 rounds
        recent_losses = recent_results.count('robot')
        loss_rate = recent_losses / len(recent_results)
        
        # Check for consecutive losses (increases frustration)
        consecutive_losses = 0
        for result in reversed(recent_results):
            if result == 'robot':
                consecutive_losses += 1
            else:
                break
                
        # Frustration increases with loss rate and consecutive losses
        base_frustration = min(1.0, loss_rate * 1.2)
        streak_frustration = min(0.5, consecutive_losses * 0.1)
        
        return GameContextBuilder._round_metric(min(1.0, base_frustration + streak_frustration))
    
    @staticmethod
    def _calculate_momentum_state(results):
        """Determine current momentum state"""
        if not results or len(results) < 5:
            return 'neutral'
            
        recent_results = results[-5:]  # Last 5 rounds
        human_wins = recent_results.count('human')
        robot_wins = recent_results.count('robot')
        
        if human_wins >= 4:
            return 'hot'
        elif robot_wins >= 4:
            return 'cold'
        elif human_wins >= 3:
            return 'warming'
        elif robot_wins >= 3:
            return 'cooling'
        else:
            return 'neutral'
    
    @staticmethod
    def _calculate_learning_trend(results, confidence_history):
        """Analyze if player is improving, declining, or stable"""
        if not results or len(results) < 10:
            return 'stable'
            
        # Split results into first half and second half
        mid_point = len(results) // 2
        first_half = results[:mid_point]
        second_half = results[mid_point:]
        
        if len(first_half) < 3 or len(second_half) < 3:
            return 'stable'
            
        first_half_win_rate = first_half.count('human') / len(first_half)
        second_half_win_rate = second_half.count('human') / len(second_half)
        
        improvement = second_half_win_rate - first_half_win_rate
        
        if improvement > 0.15:  # 15% improvement
            return 'improving'
        elif improvement < -0.15:  # 15% decline
            return 'declining'
        else:
            return 'stable'
    
    @staticmethod
    def _calculate_pattern_exploitation(human_moves, robot_moves, results):
        """Calculate how well AI is exploiting human patterns"""
        if not results or len(results) < 5:
            return 0.5
            
        # Look at recent performance (last 10 rounds)
        recent_results = results[-10:]
        robot_wins = recent_results.count('robot')
        
        # Success rate indicates pattern exploitation effectiveness
        return GameContextBuilder._round_metric(robot_wins / len(recent_results))
    
    @staticmethod
    def _calculate_adaptation_speed(results):
        """Determine how quickly player adapts to AI strategies"""
        if not results or len(results) < 10:
            return 'medium'
            
        # Check win rate progression in chunks of 5
        chunks = []
        for i in range(0, len(results), 5):
            chunk = results[i:i+5]
            if len(chunk) >= 3:  # Minimum chunk size
                win_rate = chunk.count('human') / len(chunk)
                chunks.append(win_rate)
                
        if len(chunks) < 2:
            return 'medium'
            
        # Calculate variance in win rates (high variance = fast adaptation)
        variance = sum((rate - sum(chunks)/len(chunks))**2 for rate in chunks) / len(chunks)
        
        if variance > 0.15:
            return 'fast'
        elif variance < 0.05:
            return 'slow'
        else:
            return 'medium'
    
    @staticmethod
    def _calculate_consistency_score(human_moves):
        """Calculate how consistent the player's move patterns are"""
        if not human_moves or len(human_moves) < 5:
            return 0.5
            
        from collections import Counter
        
        # Calculate move distribution
        move_counts = Counter(human_moves)
        total_moves = len(human_moves)
        
        # Perfect randomness would be 1/3 for each move
        expected_freq = total_moves / 3
        
        # Calculate deviation from perfect randomness
        deviations = []
        for move in ['rock', 'paper', 'scissors']:
            actual_freq = move_counts.get(move, 0)
            deviation = abs(actual_freq - expected_freq) / expected_freq
            deviations.append(deviation)
            
        # Average deviation (lower = more consistent/random)
        avg_deviation = sum(deviations) / len(deviations)
        
        # Convert to consistency score (0 = random, 1 = very predictable)
        return GameContextBuilder._round_metric(min(1.0, avg_deviation))
    
    @staticmethod
    def _calculate_performance_tier(results):
        """Categorize player skill level"""
        if not results or len(results) < 10:
            return 'beginner'
            
        human_win_rate = results.count('human') / len(results)
        
        if human_win_rate >= 0.55:
            return 'advanced'
        elif human_win_rate >= 0.45:
            return 'intermediate'
        else:
            return 'beginner'
    
    @staticmethod
    def _calculate_challenge_readiness(results, confidence_history):
        """Determine if player is ready for increased difficulty"""
        if not results or len(results) < 15:
            return 0.3  # Low readiness for new players
            
        # Recent performance
        recent_results = results[-10:]
        recent_win_rate = recent_results.count('human') / len(recent_results)
        
        # Consistency (stable performance indicates readiness)
        if len(results) >= 20:
            mid_results = results[-20:-10]
            mid_win_rate = mid_results.count('human') / len(mid_results)
            stability = 1.0 - abs(recent_win_rate - mid_win_rate)
        else:
            stability = 0.5
            
        # AI confidence trend (if AI is less confident, player might be ready)
        confidence_trend = 0.5
        if confidence_history and len(confidence_history) >= 10:
            recent_confidence = sum(confidence_history[-5:]) / 5
            earlier_confidence = sum(confidence_history[-10:-5]) / 5
            if recent_confidence < earlier_confidence:
                confidence_trend = 0.7  # AI is less confident = player improving
            
        # Combine factors
        readiness = (recent_win_rate * 0.4 + stability * 0.3 + confidence_trend * 0.3)
        return GameContextBuilder._round_metric(min(1.0, readiness))
        
    def build_game_context(
        self,
        session: Dict[str, Any],
        overrides: Optional[Dict[str, Any]] = None,
        config: Optional[GameContextConfig] = None
    ) -> Dict[str, Any]:
        """
        Build comprehensive game context from session data and optional overrides.
        Returns a dict with four main areas: opponent_info, game_status, analytics_metrics, endgame_metrics.
        """
        self.build_count += 1
        if config is None:
            config = GameContextConfig()

        raw_game_length = session.get('game_length', '25 Moves')
        game_length_int = None
        if isinstance(raw_game_length, str):
            import re
            match = re.search(r'(\d+)', raw_game_length)
            if match:
                game_length_int = int(match.group(1))
        elif isinstance(raw_game_length, int):
            game_length_int = raw_game_length
        # Get personality traits from personality engine
        personality_traits = {}
        personality_name = session.get('personality', 'unknown')
        if personality_name not in ('unknown', 'neutral'):
            from personality_engine import get_personality_engine
            engine = get_personality_engine()
            alias = _PERSONALITY_ALIAS_MAP.get(personality_name, personality_name)
            if alias:  # Only call if alias is not None
                personality_info = engine.get_personality_info(alias)
                if personality_info and 'traits' in personality_info:
                    personality_traits = personality_info['traits']
        
        # Get current opponent information from adaptive opponent engine
        opponent_info_42 = get_current_opponent_info()
        opponent_details = opponent_info_42.get('opponent_info', {})
        
        ai_difficulty = opponent_details.get('difficulty', session.get('ai_difficulty', 'unknown'))
        ai_strategy = opponent_details.get('strategy', session.get('strategy_preference', 'unknown'))
        ai_personality = opponent_details.get('personality', personality_name)

        opponent_info = {
            'ai_difficulty': ai_difficulty,
            'ai_strategy': ai_strategy,
            'ai_personality': ai_personality,
            # Store as integer if possible
            'game_length': game_length_int,
            # Personality traits (7 core traits) - rounded to 4 decimals
            'personality_aggression': GameContextBuilder._round_metric(personality_traits.get('aggression', 0.0)),
            'personality_defensiveness': GameContextBuilder._round_metric(personality_traits.get('defensiveness', 0.0)),
            'personality_adaptability': GameContextBuilder._round_metric(personality_traits.get('adaptability', 0.0)),
            'personality_predictability': GameContextBuilder._round_metric(personality_traits.get('predictability', 0.0)),
            'personality_risk_tolerance': GameContextBuilder._round_metric(personality_traits.get('risk_tolerance', 0.0)),
            'personality_memory_span': GameContextBuilder._round_metric(personality_traits.get('memory_span', 0.0)),
            'personality_confidence_sensitivity': GameContextBuilder._round_metric(personality_traits.get('confidence_sensitivity', 0.0)),
            # Adaptive opponent engine information
            'opponent_system': {
                'type': 'adaptive_opponent_engine',
                'status': opponent_info_42.get('status', 'not_set'),
                'opponent_id': opponent_details.get('opponent_id', 'unknown'),
                'description': opponent_details.get('description', 'Unknown opponent'),
                'difficulty_notes': opponent_details.get('difficulty_notes', ''),
                'strategy_notes': opponent_details.get('strategy_notes', ''),
                'personality_notes': opponent_details.get('personality_notes', '')
            }
        }

        # Game Status
        round_num = session.get('round_count', 0)
        game_length = opponent_info['game_length']
        try:
            game_length_int = int(game_length)
            valid_game_length = game_length_int > 0
        except (TypeError, ValueError):
            game_length_int = None
            valid_game_length = False
        
        in_game = valid_game_length and round_num < game_length_int
        end_game = valid_game_length and game_length_int is not None and round_num >= game_length_int and round_num > 0 and not in_game
        
        in_game = valid_game_length and round_num < game_length_int
        end_game = valid_game_length and game_length_int is not None and round_num >= game_length_int and round_num > 0 and not in_game

        # Centralized metrics calculation - single source of truth
        human_moves = session.get('human_moves', [])
        robot_moves = session.get('robot_moves', [])
        results = session.get('results', [])
        confidence_history = session.get('confidence_score_history', [])
        personality_modified_confidence_history = session.get('confidence_score_modified_by_personality_history', [])
        metrics = GameContextBuilder.calculate_metrics(human_moves, robot_moves, results, confidence_history, personality_modified_confidence_history)

        # Move histories and predictions (append per round)
        game_status = {
            'in_game': in_game,
            'end_game': end_game,
            'round_number': round_num,
            'metrics': metrics,
        }

        # Analytics Metrics (placeholder)
        analytics_metrics = session.get('analytics_metrics', {})

        # End-Game Metrics (placeholder)
        endgame_metrics = session.get('endgame_metrics', {})

        # Centralized context structure
        context = {
            'opponent_info': opponent_info,
            'game_status': game_status,
            'analytics_metrics': analytics_metrics,
            'endgame_metrics': endgame_metrics,
        }

        return context
    
    def _build_base_context(
        self, 
        session: Dict[str, Any], 
        overrides: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build base game context with core gameplay data"""
        
        # Extract core session data with defaults
        human_moves = session.get('human_moves', [])
        robot_moves = session.get('robot_moves', [])
        results = session.get('results', [])
        
        # Apply overrides if provided
        if overrides:
            human_moves = overrides.get('human_moves', human_moves)
            robot_moves = overrides.get('robot_moves', robot_moves)
            results = overrides.get('results', results)
        
        # Derived fields
        current_round = len(human_moves)
        
        # Strategy and difficulty with fallback logic
        human_strategy_label = self._get_strategy_label(session, overrides)
        ai_difficulty = self._get_ai_difficulty(session, overrides)
        
        return {
            # Core gameplay data
            'human_moves': human_moves,
            'robot_moves': robot_moves,
            'results': results,
            'round': current_round,
            'current_round': current_round,  # Alias for compatibility
            
            # Strategy and difficulty
            'current_strategy': human_strategy_label,
            'human_strategy_label': human_strategy_label,
            'ai_difficulty': ai_difficulty,
            'current_difficulty': ai_difficulty,  # Alias for compatibility
            'difficulty': ai_difficulty,  # Legacy alias
            
            # Player preferences
            'strategy_preference': session.get('strategy_preference', 'to_win'),
            'personality': session.get('personality', 'neutral'),
            'confidence': session.get('confidence', 0.5),
            
            # Game mode
            'multiplayer': session.get('multiplayer', False),
        }
    
    def _build_ai_behavior_context(
        self, 
        session: Dict[str, Any], 
        overrides: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build AI behavior context with model performance data"""
        
        context = {}
        
        # AI behavior session data (the fix from our previous work)
        ai_fields = [
            'accuracy',
            'model_predictions_history', 
            'model_confidence_history',
            'correct_predictions',
            'total_predictions'
        ]
        
        for field in ai_fields:
            value = session.get(field, {})
            if overrides and field in overrides:
                value = overrides[field]
            context[field] = value
            
        return context
    
    def _build_advanced_context(
        self, 
        session: Dict[str, Any], 
        overrides: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build advanced analytics context"""
        
        context = {}
        
        # Advanced analytics fields
        advanced_fields = [
            'change_points',
            'human_history',  # May be different from human_moves
            'robot_history',  # May be different from robot_moves
        ]
        
        for field in advanced_fields:
            value = session.get(field, [])
            if overrides and field in overrides:
                value = overrides[field]
            context[field] = value
            
        return context
    
    def _build_temporal_context(
        self, 
        session: Dict[str, Any], 
        overrides: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build temporal and session context"""
        
        context = {}
        
        # Temporal fields
        temporal_fields = [
            'round_count',
            'session_start_time',
            'game_start_time'
        ]
        
        for field in temporal_fields:
            value = session.get(field)
            if overrides and field in overrides:
                value = overrides[field]
            if value is not None:
                context[field] = value
                
        return context
    
    def _get_strategy_label(
        self, 
        session: Dict[str, Any], 
        overrides: Optional[Dict[str, Any]]
    ) -> str:
        """Get human strategy label with fallback logic"""
        
        if overrides and 'human_strategy_label' in overrides:
            return overrides['human_strategy_label']
            
        # Try multiple session keys in order of preference
        strategy_keys = [
            'human_strategy_label',
            'current_strategy', 
            'strategy'
        ]
        
        for key in strategy_keys:
            value = session.get(key)
            if value and value != 'unknown':
                return value
                
        return 'unknown'
    
    def _get_ai_difficulty(
        self, 
        session: Dict[str, Any], 
        overrides: Optional[Dict[str, Any]]
    ) -> str:
        """Get AI difficulty with fallback logic"""
        
        if overrides and 'ai_difficulty' in overrides:
            return overrides['ai_difficulty']
            
        # Try multiple session keys in order of preference
        difficulty_keys = [
            'ai_difficulty',
            'difficulty',
            'current_difficulty'
        ]
        
        for key in difficulty_keys:
            value = session.get(key)
            if value:
                return value
                
        return 'medium'  # Default difficulty
    
    def _validate_context(
        self, 
        context: Dict[str, Any], 
        config: GameContextConfig
    ) -> None:
        """Validate the built context for completeness and correctness"""
        
        # Required fields check
        required_fields = [
            'human_moves', 'robot_moves', 'results', 'round',
            'human_strategy_label', 'ai_difficulty'
        ]
        
        missing_fields = [field for field in required_fields if field not in context]
        if missing_fields:
            raise ValueError(f"Missing required context fields: {missing_fields}")
            
        # Data consistency checks
        human_moves = context['human_moves']
        robot_moves = context['robot_moves'] 
        results = context['results']
        
        if len(human_moves) != len(robot_moves):
            raise ValueError(f"Move count mismatch: human={len(human_moves)}, robot={len(robot_moves)}")
            
        if len(results) > len(human_moves):
            raise ValueError(f"Too many results: {len(results)} > {len(human_moves)}")
            
        # Round consistency
        if context['round'] != len(human_moves):
            raise ValueError(f"Round mismatch: round={context['round']}, moves={len(human_moves)}")


# Global instance for easy access
game_context_builder = GameContextBuilder()


def build_game_context(
    session: Dict[str, Any], 
    overrides: Optional[Dict[str, Any]] = None,
    context_type: str = 'full'
) -> Dict[str, Any]:
    """
    Convenience function for building game context.
    
    This is the primary entry point that should be used throughout the application.
    
    Args:
        session: Flask session dictionary
        overrides: Optional overrides for request-specific data
        context_type: Type of context to build ('realtime', 'comprehensive', 'analytics', 'full')
        
    Returns:
        Complete game context dictionary
    """
    
    # Configure based on context type
    config = GameContextConfig(context_type=context_type)
    
    if context_type == 'realtime':
        # Optimize for real-time performance
        config.include_advanced_metrics = False
        config.include_temporal_data = False
        config.validate_schema = False
    elif context_type == 'analytics':
        # Focus on core data for analytics
        config.include_ai_behavior = True
        config.include_advanced_metrics = True
    
    return game_context_builder.build_game_context(session, overrides, config)


def get_builder_stats() -> Dict[str, Any]:
    """Get performance statistics for monitoring"""
    return {
        'build_count': game_context_builder.build_count,
        'last_build_time': time.time()
    }


def reset_builder_stats() -> None:
    """Reset performance statistics"""
    game_context_builder.build_count = 0


# For testing and debugging
def validate_context_against_baseline(context: Dict[str, Any], baseline_path: str) -> bool:
    """
    Validate a context against a baseline snapshot for regression testing.
    
    This will be used in our regression test harness.
    """
    import json
    
    try:
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)
            
        # Compare key fields (ignoring timestamps and other variable data)
        comparison_fields = [
            'human_moves', 'robot_moves', 'results', 'round',
            'human_strategy_label', 'ai_difficulty', 'accuracy'
        ]
        
        for field in comparison_fields:
            if field in baseline.get('final_context', {}):
                baseline_value = baseline['final_context'][field]
                context_value = context.get(field)
                
                if baseline_value != context_value:
                    print(f"Mismatch in {field}: baseline={baseline_value}, context={context_value}")
                    return False
                    
        return True
        
    except Exception as e:
        print(f"Validation error: {e}")
        return False
