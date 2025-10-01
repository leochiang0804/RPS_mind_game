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
        
    def build_game_context(
        self, 
        session: Dict[str, Any], 
        overrides: Optional[Dict[str, Any]] = None,
        config: Optional[GameContextConfig] = None
    ) -> Dict[str, Any]:
        """
        Build comprehensive game context from session data and optional overrides.
        
        This is the single source of truth for game context construction.
        
        Args:
            session: Flask session dictionary containing game state
            overrides: Optional dict to override specific fields (for requests)
            config: Optional config to control what data is included
            
        Returns:
            Complete game context dictionary ready for AI coach or analytics
            
        Raises:
            ValueError: If required session data is missing
        """
        self.build_count += 1
        
        if config is None:
            config = GameContextConfig()
            
        # Start with base structure
        context = self._build_base_context(session, overrides)
        
        # Add AI behavior data if requested and available
        if config.include_ai_behavior:
            context.update(self._build_ai_behavior_context(session, overrides))
            
        # Add advanced metrics if requested
        if config.include_advanced_metrics:
            context.update(self._build_advanced_context(session, overrides))
            
        # Add temporal data if requested
        if config.include_temporal_data:
            context.update(self._build_temporal_context(session, overrides))
            
        # Validate schema if requested
        if config.validate_schema:
            self._validate_context(context, config)
            
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
            'strategy_preference': session.get('strategy_preference', 'balanced'),
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