"""
Enhanced Coach System with AI/Basic Mode Toggle
Provides seamless switching between basic rule-based coaching and AI-powered coaching
"""

import os
import json
import time
from typing import Dict, List, Any, Optional
from coach_tips import CoachTipsGenerator
from ai_coach_metrics import ai_metrics_aggregator
from change_point_detector import ChangePointDetector


class EnhancedCoachSystem:
    """
    Enhanced coaching system that supports both basic and AI modes
    Provides graceful fallback and seamless mode switching
    """
    
    def __init__(self):
        # Always available basic coach
        self.basic_coach = CoachTipsGenerator()
        
        # Change point detector for pattern analysis
        self.change_point_detector = ChangePointDetector()
        
        # AI coach components (lazy loaded)
        self.ai_coach = None
        self.langchain_coach = None
        
        # Current mode
        self.mode = 'basic'  # Default to basic mode
        
        # Coaching style preference
        self.coaching_style = 'easy'  # 'easy' or 'scientific'
        
        # Performance monitoring
        self.performance_metrics = {
            'ai_success_count': 0,
            'ai_failure_count': 0,
            'fallback_count': 0,
            'average_response_time': 0.0
        }
        
        # Attempt to initialize AI coach
        self._initialize_ai_coach()
    
    def _initialize_ai_coach(self):
        """Initialize AI coach components if available"""
        try:
            # Try to import AI coach components
            from ai_coach_langchain import LangChainAICoach
            self.langchain_coach = LangChainAICoach()
            print("✅ AI Coach successfully initialized")
            # IMPORTANT: Automatically switch to AI mode when AI coach is available
            self.mode = 'ai'
            print("🔄 Enhanced Coach automatically switched to AI mode")
        except ImportError as e:
            print(f"ℹ️ AI Coach not available: {e}")
        except Exception as e:
            print(f"⚠️ AI Coach initialization failed: {e}")
    
    def set_mode(self, mode: str) -> Dict[str, Any]:
        """
        Set coaching mode with validation
        
        Args:
            mode: 'basic' or 'ai'
            
        Returns:
            Status dictionary with success/failure info
        """
        if mode not in ['basic', 'ai']:
            return {
                'success': False,
                'error': f"Invalid mode '{mode}'. Must be 'basic' or 'ai'",
                'current_mode': self.mode
            }
        
        # Check if AI mode is available
        if mode == 'ai' and self.langchain_coach is None:
            return {
                'success': False,
                'error': 'AI coaching not available. Please check AI coach setup.',
                'current_mode': self.mode,
                'fallback': 'Remaining in basic mode'
            }
        
        old_mode = self.mode
        self.mode = mode
        
        return {
            'success': True,
            'old_mode': old_mode,
            'new_mode': self.mode,
            'ai_available': self.langchain_coach is not None
        }
    
    def get_mode(self) -> str:
        """Get current coaching mode"""
        return self.mode
    
    def is_ai_available(self) -> bool:
        """Check if AI coaching is available"""
        return self.langchain_coach is not None
    
    def set_coaching_style(self, style: str) -> Dict[str, Any]:
        """
        Set coaching style preference
        
        Args:
            style: 'easy' for easy-to-understand tips, 'scientific' for detailed analytics
            
        Returns:
            Status dictionary
        """
        if style not in ['easy', 'scientific']:
            return {
                'success': False,
                'error': f'Invalid style: {style}. Must be "easy" or "scientific"',
                'current_style': self.coaching_style
            }
        
        old_style = self.coaching_style
        self.coaching_style = style
        
        # Update AI coach style if available
        if self.langchain_coach is not None:
            self.langchain_coach.set_coaching_style(style)
        
        return {
            'success': True,
            'old_style': old_style,
            'new_style': self.coaching_style,
            'description': {
                'easy': 'Simple, easy-to-understand coaching tips',
                'scientific': 'Detailed analytics with scientific terminology'
            }[style]
        }
    
    def get_coaching_style(self) -> str:
        """Get current coaching style"""
        return self.coaching_style
    
    def get_style_description(self) -> Dict[str, str]:
        """Get description of available coaching styles"""
        return {
            'easy': 'Simple, friendly tips that anyone can understand',
            'scientific': 'Detailed analytics with entropy, Nash equilibrium, and complexity metrics'
        }
    
    def set_llm_type(self, llm_type: str) -> Dict[str, Any]:
        """
        Set the LLM type for AI coaching
        
        Args:
            llm_type: 'mock' for fast MockLLM, 'real' for actual LLM models
            
        Returns:
            Status dictionary
        """
        if llm_type not in ['mock', 'real']:
            return {
                'success': False,
                'error': f'Invalid LLM type: {llm_type}. Must be "mock" or "real"',
                'current_llm_type': self.get_llm_type()
            }
        
        # If AI coach is not available, can't change LLM type
        if self.langchain_coach is None:
            return {
                'success': False,
                'error': 'AI coaching not available. LLM type change requires AI coach.',
                'current_llm_type': 'none'
            }
        
        old_type = self.get_llm_type()
        
        # Update the LangChain coach's LLM type
        result = self.langchain_coach.set_llm_type(llm_type)
        
        return {
            'success': result.get('success', False),
            'old_llm_type': old_type,
            'new_llm_type': llm_type,
            'ai_available': True,
            'details': result
        }
    
    def get_llm_type(self) -> str:
        """Get current LLM type"""
        if self.langchain_coach is None:
            return 'none'
        return self.langchain_coach.get_llm_type()
    
    def generate_coaching_advice(self, game_state: Dict[str, Any], coaching_type: str = 'real_time') -> Dict[str, Any]:
        """
        Generate coaching advice based on current mode
        
        Args:
            game_state: Complete game state dictionary
            coaching_type: 'real_time' or 'comprehensive'
            
        Returns:
            Coaching advice dictionary with tips, insights, etc.
        """
        start_time = time.time()
        
        try:
            if self.mode == 'ai' and self.langchain_coach is not None:
                return self._generate_ai_coaching(game_state, coaching_type, start_time)
            else:
                return self._generate_basic_coaching(game_state, start_time)
                
        except Exception as e:
            print(f"⚠️ Coaching generation failed: {e}")
            # Always fallback to basic coaching
            self.performance_metrics['fallback_count'] += 1
            return self._generate_basic_coaching(game_state, start_time, fallback_reason=str(e))
    
    def _generate_ai_coaching(self, game_state: Dict[str, Any], coaching_type: str, start_time: float) -> Dict[str, Any]:
        """Generate AI-powered coaching advice using appropriate metric category"""
        
        try:
            # Choose appropriate metrics based on coaching type
            if coaching_type == 'real_time':
                # Use fast, lightweight metrics for real-time coaching
                metrics = ai_metrics_aggregator.get_realtime_metrics(game_state)
                print(f"📊 USING REAL-TIME METRICS: {len(metrics) if isinstance(metrics, dict) else 0} categories")
            elif coaching_type == 'post_game' or coaching_type == 'comprehensive':
                # Use comprehensive metrics for detailed analysis
                metrics = ai_metrics_aggregator.get_postgame_metrics(game_state)
                print(f"📊 USING POST-GAME METRICS: {len(metrics) if isinstance(metrics, dict) else 0} categories")
            else:
                # Fallback to comprehensive metrics
                metrics = ai_metrics_aggregator.aggregate_comprehensive_metrics(game_state)
                print(f"📊 USING COMPREHENSIVE METRICS: {len(metrics) if isinstance(metrics, dict) else 0} categories")
            
            # Generate AI coaching advice
            ai_advice = self.langchain_coach.generate_coaching_advice(
                metrics, 
                coaching_type
            )
            
            # Calculate response time
            response_time = time.time() - start_time
            self._update_performance_metrics(True, response_time)
            
            # Enhance with basic coach insights for completeness
            basic_advice = self._generate_basic_coaching_data(game_state)
            
            # Combine AI and basic insights
            combined_advice = {
                'mode': 'ai',
                'coaching_type': coaching_type,
                'metrics_category': 'realtime' if coaching_type == 'real_time' else 'postgame',
                'ai_advice': ai_advice,
                'tips': ai_advice.get('tips', basic_advice.get('tips', [])),
                'experiments': basic_advice.get('experiments', []),  # Use basic experiments
                'insights': {
                    'ai_insights': ai_advice.get('insights', {}),
                    'basic_insights': basic_advice.get('insights', {}),
                    'metrics_summary': metrics.get('meta', {}) if 'meta' in metrics else {'type': coaching_type}
                },
                'educational_content': ai_advice.get('educational_content', {}),
                'behavioral_analysis': ai_advice.get('behavioral_analysis', {}),
                'performance': {
                    'response_time_ms': response_time * 1000,
                    'data_quality': metrics.get('meta', {}).get('data_quality_score', 0.5) if 'meta' in metrics else 0.8
                },
                'round': game_state.get('round', 0),
                'current_strategy': game_state.get('current_strategy', 'unknown'),
                'raw_response': ai_advice.get('raw_response', ''),
                'natural_language_full': ai_advice.get('natural_language_full', ''),
                'llm_type': ai_advice.get('llm_type', self.langchain_coach.get_llm_type() if self.langchain_coach else 'unknown')
            }
            
            return combined_advice
            
        except Exception as e:
            print(f"⚠️ AI coaching failed: {e}")
            self._update_performance_metrics(False, time.time() - start_time)
            # Fallback to basic coaching
            self.performance_metrics['fallback_count'] += 1
            return self._generate_basic_coaching(game_state, start_time, fallback_reason=str(e))
    
    def _generate_basic_coaching(self, game_state: Dict[str, Any], start_time: float, fallback_reason: Optional[str] = None) -> Dict[str, Any]:
        """Generate basic coaching advice using existing system"""
        
        basic_advice = self._generate_basic_coaching_data(game_state)
        response_time = time.time() - start_time
        
        result = {
            'mode': 'basic',
            'coaching_type': 'basic',
            'tips': basic_advice.get('tips', []),
            'experiments': basic_advice.get('experiments', []),
            'insights': basic_advice.get('insights', {}),
            'performance': {
                'response_time_ms': response_time * 1000
            },
            'round': game_state.get('round', 0),
            'current_strategy': game_state.get('current_strategy', 'unknown')
        }
        
        if fallback_reason:
            result['fallback_info'] = {
                'reason': fallback_reason,
                'original_mode': 'ai'
            }
        
        return result
    
    def _generate_basic_coaching_data(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate basic coaching data using existing coach"""
        
        # Map game_state keys to the expected parameters for CoachTipsGenerator
        human_moves = game_state.get('human_moves', [])
        robot_moves = game_state.get('robot_moves', [])
        results = game_state.get('results', [])
        
        # Get change points - use the built-in change points if available
        change_points = game_state.get('change_points', [])
        
        # If no change points provided, try to detect them
        if not change_points and len(human_moves) > 10:
            try:
                # Reset and process moves to detect change points
                temp_detector = ChangePointDetector()
                for move in human_moves:
                    change_point = temp_detector.add_move(move)
                    if change_point:
                        change_points.append(change_point)
            except Exception as e:
                print(f"⚠️ Change point detection failed: {e}")
                change_points = []
        
        return self.basic_coach.generate_tips(
            human_history=human_moves,
            robot_history=robot_moves,
            result_history=results,
            change_points=change_points,
            current_strategy=game_state.get('current_strategy', 'unknown')
        )
    
    def _update_performance_metrics(self, success: bool, response_time: float):
        """Update performance tracking metrics"""
        
        if success:
            self.performance_metrics['ai_success_count'] += 1
        else:
            self.performance_metrics['ai_failure_count'] += 1
        
        # Update average response time
        total_attempts = self.performance_metrics['ai_success_count'] + self.performance_metrics['ai_failure_count']
        current_avg = self.performance_metrics['average_response_time']
        self.performance_metrics['average_response_time'] = (
            (current_avg * (total_attempts - 1) + response_time) / total_attempts
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring"""
        
        total_attempts = self.performance_metrics['ai_success_count'] + self.performance_metrics['ai_failure_count']
        success_rate = 0.0
        
        if total_attempts > 0:
            success_rate = self.performance_metrics['ai_success_count'] / total_attempts
        
        return {
            'current_mode': self.mode,
            'ai_available': self.is_ai_available(),
            'success_rate': success_rate,
            'total_attempts': total_attempts,
            'fallback_count': self.performance_metrics['fallback_count'],
            'average_response_time_ms': self.performance_metrics['average_response_time'] * 1000,
            'performance_summary': self._get_performance_summary(success_rate, total_attempts)
        }
    
    def _get_performance_summary(self, success_rate: float, total_attempts: int) -> str:
        """Get human-readable performance summary"""
        
        if total_attempts == 0:
            return "No AI coaching attempts yet"
        elif success_rate > 0.9:
            return "Excellent AI performance"
        elif success_rate > 0.7:
            return "Good AI performance"
        elif success_rate > 0.5:
            return "Moderate AI performance"
        else:
            return "Poor AI performance - frequent fallbacks"
    
    def generate_comprehensive_analysis(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive post-game analysis
        Only available in AI mode
        """
        
        if self.mode != 'ai' or self.langchain_coach is None:
            return {
                'error': 'Comprehensive analysis only available in AI mode',
                'current_mode': self.mode,
                'ai_available': self.is_ai_available()
            }
        
        try:
            # Get comprehensive metrics
            comprehensive_metrics = ai_metrics_aggregator.aggregate_comprehensive_metrics(game_state)
            
            # Generate comprehensive analysis
            analysis = self.langchain_coach.generate_comprehensive_analysis(comprehensive_metrics)
            
            return {
                'success': True,
                'analysis': analysis,
                'session_summary': {
                    'total_rounds': game_state.get('round', 0),
                    'data_quality': comprehensive_metrics.get('meta', {}).get('data_quality_score', 0.5),
                    'analysis_timestamp': time.time()
                }
            }
            
        except Exception as e:
            return {
                'error': f'Comprehensive analysis failed: {str(e)}',
                'fallback_available': False
            }
    
    def reset_performance_metrics(self):
        """Reset performance metrics (useful for testing)"""
        self.performance_metrics = {
            'ai_success_count': 0,
            'ai_failure_count': 0,
            'fallback_count': 0,
            'average_response_time': 0.0
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        return {
            'current_mode': self.mode,
            'basic_coach_available': True,  # Always available
            'ai_coach_available': self.is_ai_available(),
            'performance_metrics': self.get_performance_metrics(),
            'system_health': {
                'basic_coach_status': 'healthy',
                'ai_coach_status': 'healthy' if self.is_ai_available() else 'unavailable',
                'metrics_aggregator_status': 'healthy'
            }
        }


# Global instance for use across the application
enhanced_coach = EnhancedCoachSystem()

def get_enhanced_coach():
    """Get global enhanced coach instance"""
    return enhanced_coach
