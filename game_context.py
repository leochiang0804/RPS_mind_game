"""
Centralized Game Context Builder
===============================

This module provides a single source of truth for building game context data
used across all AI coach endpoints, analytics, and UI components.

The goal is to eliminate scattered data construction and ensure consistency
across the entire application.
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


class GameContextBuilder:
    """
    Centralized game context builder with validation and optimization
    
    This class consolidates all the scattered game data construction logic
    from various endpoints into a single, consistent, and testable implementation.
    """
    
    def __init__(self):
        self.build_count = 0  # For performance monitoring
    
    @staticmethod
    def calculate_metrics(human_moves, robot_moves, results):
        """
        Unified metric calculation - single source of truth for all game metrics.
        This ensures consistency between what's displayed and what's saved.
        """
        from collections import Counter
        
        # Win/tie counts
        human_wins = results.count('human')
        robot_wins = results.count('robot')
        ties = results.count('tie')
        total_rounds = len(results) if results else 0
        human_win_rate = human_wins / total_rounds * 100 if total_rounds else 0.0
        robot_win_rate = robot_wins / total_rounds * 100 if total_rounds else 0.0
        tie_rate = ties / total_rounds * 100 if total_rounds else 0.0

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
        recent_win_rate = recent_human_wins / len(recent_results) * 100 if recent_results else 0.0

        # Score differential
        score_differential = human_wins - robot_wins

        # AI confidence (placeholder, can be enhanced with model-specific logic)
        ai_confidence = None

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
            return min(100, (variance / expected_freq) * 50)
        predictability_score = calculate_move_variance(human_moves)

        # Recent momentum (last 10 moves)
        recent_moves = human_moves[-10:]
        recent_bias_type = None
        recent_bias_percent = None
        if recent_moves:
            move_counts = Counter([m for m in recent_moves if m])
            if move_counts:
                bias_type, bias_count = move_counts.most_common(1)[0]
                percent = (bias_count / len(recent_moves)) * 100
                recent_bias_type = bias_type
                recent_bias_percent = percent

        return {
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
            'AI_confidence': ai_confidence,
        }
        
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
        opponent_info = {
            'ai_difficulty': session.get('ai_difficulty', 'unknown'),
            'ai_strategy': session.get('strategy_preference', 'unknown'),
            'ai_personality': session.get('personality', 'unknown'),
            # Store as integer if possible
            'game_length': game_length_int,
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
        metrics = GameContextBuilder.calculate_metrics(human_moves, robot_moves, results)

        # Move histories and predictions (append per round)
        game_status = {
            'in_game': in_game,
            'end_game': end_game,
            'round_number': round_num,
            'metrics': metrics,
            'human_move_history': session.get('human_moves', []),
            'robot_move_history': session.get('robot_moves', []),
            'random_ai_predictions': session.get('model_predictions_history', {}).get('random', []),
            'frequency_ai_predictions': session.get('model_predictions_history', {}).get('frequency', []),
            'markov_ai_predictions': session.get('model_predictions_history', {}).get('markov', []),
            'lstm_ai_predictions': session.get('model_predictions_history', {}).get('lstm', []),
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