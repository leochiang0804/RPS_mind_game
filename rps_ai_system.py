"""
RPS AI System - 42 Opponents Framework
======================================

Main AI system that integrates:
- Markov predictors (orders 1-3)  
- Human-Like Bias Module (HLBM)
- Parameter-Synthesis Engine (PSE)

Provides complete replacement for LSTM-based system with 42 distinct opponents.

Author: AI Assistant
Created: 2025-10-03
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Sequence
import json
import time
from collections import defaultdict

# Import our modules
from markov_predictor import MarkovPredictor, EnsembleMarkovPredictor
from hlbm import HumanLikeBiasModule, create_hlbm
from parameter_synthesis_engine import (
    ParameterSynthesisEngine, OpponentParameters, create_parameter_synthesis_engine
)
from move_mapping import (
    MOVES, MOVE_TO_NUMBER, NUMBER_TO_MOVE,
    normalize_move, get_counter_move, number_to_move
)


class RPSAISystem:
    """
    Complete RPS AI system with 42 opponents framework.
    
    Combines Markov prediction, HLBM psychological modeling,
    and PSE parameter synthesis for diverse gameplay.
    """
    
    def __init__(self):
        """Initialize the AI system."""
        
        # Core components
        self.pse = create_parameter_synthesis_engine()
        self.current_opponent = None
        self.markov_predictor = None
        self.hlbm = None
        
        # Game state
        self.move_history = []
        self.outcome_history = []  # From human perspective
        self.prediction_history = []
        self.confidence_history = []
        
        # Performance tracking
        self.prediction_count = 0
        self.correct_predictions = 0
        self.session_start_time = None
        
        # Metadata tracking
        self.metadata_log = []
        
    def set_opponent(self, difficulty: str, strategy: str, personality: str) -> bool:
        """
        Set the current opponent and configure AI components.
        
        Args:
            difficulty: 'rookie', 'challenger', or 'master'
            strategy: 'to_win' or 'not_to_lose'
            personality: 'neutral', 'aggressive', 'defensive', etc.
            
        Returns:
            True if opponent was set successfully
        """
        # Get opponent parameters
        opponent = self.pse.get_opponent(difficulty, strategy, personality)
        if not opponent:
            print(f"Warning: No opponent found for {difficulty}/{strategy}/{personality}")
            return False
            
        self.current_opponent = opponent
        
        # Initialize Markov predictor
        if opponent.markov_order == 1:
            self.markov_predictor = MarkovPredictor(
                order=1, 
                smoothing_factor=opponent.smoothing_factor
            )
        else:
            # Use ensemble for higher orders
            orders = list(opponent.ensemble_weights.keys())
            self.markov_predictor = EnsembleMarkovPredictor(
                orders=orders,
                smoothing_factor=opponent.smoothing_factor,
                ensemble_weights=opponent.ensemble_weights
            )
            
        # Initialize HLBM
        self.hlbm = HumanLikeBiasModule(
            lambda_influence=opponent.lambda_influence,
            weights=opponent.bias_weights.copy(),
            bias_params=opponent.bias_params.copy()
        )
        
        # Reset game state
        self.reset_game_state()
        
        print(f"Opponent set: {opponent.opponent_id} - {opponent.description}")
        return True
        
    def reset_game_state(self) -> None:
        """Reset game state for new game."""
        self.move_history.clear()
        self.outcome_history.clear()
        self.prediction_history.clear()
        self.confidence_history.clear()
        self.prediction_count = 0
        self.correct_predictions = 0
        self.session_start_time = time.time()
        self.metadata_log.clear()
        
        # Reset predictor state
        if self.markov_predictor:
            self.markov_predictor.reset()
            
    def predict_next_move(self, 
                         human_moves: Optional[Sequence[Union[str, int]]] = None,
                         outcomes: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, str, Dict]:
        """
        Predict human's next move and return AI's counter-move.
        
        Args:
            human_moves: Optional move history (uses internal if None)
            outcomes: Optional outcome history (uses internal if None)
            
        Returns:
            Tuple of (probabilities, ai_move, metadata)
        """
        if not self.current_opponent:
            raise ValueError("No opponent set. Call set_opponent() first.")
            
        # Use provided or internal history
        if human_moves is not None:
            moves_to_use = list(human_moves)
        else:
            moves_to_use = self.move_history.copy()
            
        if outcomes is not None:
            outcomes_to_use = list(outcomes)
        else:
            outcomes_to_use = self.outcome_history.copy()
            
        # Get base prediction from Markov predictor
        if len(moves_to_use) == 0:
            # No history - uniform distribution
            p_base = np.array([1/3, 1/3, 1/3])
            markov_metadata = {'method': 'uniform', 'reason': 'no_history'}
        elif self.markov_predictor is None:
            # Predictor not initialized
            p_base = np.array([1/3, 1/3, 1/3])
            markov_metadata = {'method': 'uniform', 'reason': 'no_predictor'}
        else:
            p_base, markov_metadata = self.markov_predictor.predict(moves_to_use)
            
            # Apply epsilon noise to p_base for exploration at prediction level
            if self.current_opponent and self.current_opponent.epsilon > 0:
                epsilon = self.current_opponent.epsilon
                uniform_noise = np.array([1/3, 1/3, 1/3])
                # Mix base prediction with uniform distribution based on epsilon
                p_base = (1 - epsilon) * p_base + epsilon * uniform_noise
                markov_metadata['epsilon_applied'] = epsilon
                markov_metadata['method'] += '_with_epsilon_noise'
            
        # Apply HLBM biases
        if len(moves_to_use) >= 2 and self.hlbm is not None:  # Need some history for biases
            p_adjusted, hlbm_metadata = self.hlbm.apply_biases(
                p_base, moves_to_use, outcomes_to_use
            )
        else:
            p_adjusted = p_base
            hlbm_metadata = {'method': 'none', 'reason': 'insufficient_history'}
            
        # Apply strategy layer
        ai_move, strategy_metadata = self._apply_strategy_layer(p_adjusted)
        
        # Calculate confidence
        confidence = self._calculate_confidence(p_adjusted, markov_metadata)
        
        # Store prediction
        self.prediction_history.append({
            'probabilities': p_adjusted.tolist(),
            'ai_move': ai_move,
            'confidence': confidence,
            'timestamp': time.time()
        })
        self.confidence_history.append(confidence)
        
        # Comprehensive metadata
        metadata = {
            'opponent_id': self.current_opponent.opponent_id,
            'prediction_number': len(self.prediction_history),
            'markov_prediction': {
                'probabilities': p_base.tolist(),
                'metadata': markov_metadata
            },
            'hlbm_adjustment': {
                'probabilities': p_adjusted.tolist(),
                'metadata': hlbm_metadata
            },
            'strategy_application': strategy_metadata,
            'final_move': ai_move,
            'confidence': confidence,
            'move_history_length': len(moves_to_use),
            'current_accuracy': self.get_accuracy(),
            'timestamp': time.time()
        }
        
        self.metadata_log.append(metadata)
        
        return p_adjusted, ai_move, metadata
        
    def update_with_human_move(self, human_move: Union[str, int], ai_move: str) -> Dict:
        """
        Update the system with the human's actual move.
        
        Args:
            human_move: Human's actual move
            ai_move: AI's move that was played
            
        Returns:
            Evaluation metadata
        """
        if not self.current_opponent:
            raise ValueError("No opponent set.")
            
        # Normalize moves
        if isinstance(human_move, int):
            human_move = number_to_move(human_move)
        human_move = normalize_move(human_move)
        ai_move = normalize_move(ai_move)
        
        # Add to history
        self.move_history.append(human_move)
        
        # Determine outcome from human perspective
        outcome = self._determine_outcome(human_move, ai_move)
        self.outcome_history.append(outcome)
        
        # Update Markov predictor
        if self.markov_predictor is not None:
            self.markov_predictor.update(human_move)
        
        # Evaluate prediction accuracy
        if len(self.prediction_history) > 0:
            last_prediction = self.prediction_history[-1]
            predicted_probs = np.array(last_prediction['probabilities'])
            
            # Check if AI's prediction was correct
            predicted_move_idx = np.argmax(predicted_probs)
            predicted_move = MOVES[predicted_move_idx]
            actual_move_idx = MOVE_TO_NUMBER[human_move]
            
            self.prediction_count += 1
            if predicted_move == human_move:
                self.correct_predictions += 1
                
            # Evaluate prediction quality
            prediction_quality = float(predicted_probs[actual_move_idx])
            
        else:
            prediction_quality = 0.0
            
        # Update metadata
        update_metadata = {
            'human_move': human_move,
            'ai_move': ai_move,
            'outcome': outcome,
            'prediction_quality': prediction_quality,
            'current_accuracy': self.get_accuracy(),
            'game_round': len(self.move_history),
            'timestamp': time.time()
        }
        
        return update_metadata
        
    def _get_gamma_weighted_prediction(self, probabilities: np.ndarray, gamma: float) -> str:
        """Apply gamma weighting to predictions for exploitation vs exploration."""
        if gamma >= 1.0:
            # Pure exploitation - always take highest probability
            return MOVES[np.argmax(probabilities)]
        elif gamma <= 0.0:
            # Pure exploration - uniform random
            return np.random.choice(MOVES)
        else:
            # Weighted by gamma - higher gamma = more exploitation
            weighted_probs = np.power(probabilities, gamma)
            weighted_probs /= np.sum(weighted_probs)  # Normalize
            
            # Sample from weighted distribution
            move_idx = np.random.choice(3, p=weighted_probs)
            return MOVES[move_idx]

    def _apply_strategy_layer(self, probabilities: np.ndarray) -> Tuple[str, Dict]:
        """Apply strategy layer to convert probabilities to move."""
        
        if not self.current_opponent:
            return MOVES[np.argmax(probabilities)], {'method': 'default'}
            
        alpha = self.current_opponent.alpha
        epsilon = self.current_opponent.epsilon
        gamma = self.current_opponent.gamma
        
        # Strategy logic with alpha blending
        # Alpha determines how much to blend To-Win vs Not-to-Lose behaviors
        # High alpha (~0.85) = more To-Win behavior
        # Low alpha (~0.15) = more Not-to-Lose behavior
        
        use_to_win_strategy = np.random.random() < alpha
        
        if use_to_win_strategy:
            # To-win strategy: predict human move, play counter
            
            # Use gamma to determine exploitation vs exploration of predictions
            if np.random.random() > epsilon:
                # Exploit with gamma weighting
                predicted_human_move = self._get_gamma_weighted_prediction(probabilities, gamma)
                ai_move = get_counter_move(predicted_human_move)
                method = f'to_win_gamma_counter (α={alpha:.2f}, γ={gamma:.2f})'
            else:
                # Exploration: weighted random based on probabilities
                # Convert human prediction to AI counter probabilities
                ai_probs = np.zeros(3)
                for i, human_move in enumerate(MOVES):
                    counter_move = get_counter_move(human_move)
                    counter_idx = MOVE_TO_NUMBER[counter_move]
                    ai_probs[counter_idx] += probabilities[i]
                    
                ai_probs = ai_probs / np.sum(ai_probs)
                ai_move_idx = np.random.choice(3, p=ai_probs)
                ai_move = MOVES[ai_move_idx]
                method = f'to_win_exploration (α={alpha:.2f})'
                
        else:  # not_to_lose
            # Not-to-lose strategy: avoid moves that lose to predicted human move
            predicted_human_move = self._get_gamma_weighted_prediction(probabilities, gamma)
            
            # Find moves that don't lose to predicted move
            safe_moves = []
            for move in MOVES:
                if get_counter_move(move) != predicted_human_move:
                    safe_moves.append(move)
                    
            if safe_moves and np.random.random() > epsilon:
                ai_move = np.random.choice(safe_moves)
                method = f'not_to_lose_safe (α={alpha:.2f}, γ={gamma:.2f})'
            else:
                # Fallback to counter-move
                ai_move = get_counter_move(predicted_human_move)
                method = f'not_to_lose_fallback (α={alpha:.2f})'
                
        metadata = {
            'strategy': self.current_opponent.strategy.value,
            'alpha': alpha,
            'epsilon': epsilon,
            'gamma': gamma,
            'method': method,
            'predicted_human_move': MOVES[np.argmax(probabilities)]
        }
        
        return ai_move, metadata
        
    def _determine_outcome(self, human_move: str, ai_move: str) -> str:
        """Determine game outcome from human perspective."""
        if human_move == ai_move:
            return 'tie'
        elif get_counter_move(ai_move) == human_move:
            return 'win'  # Human wins
        else:
            return 'loss'  # Human loses
            
    def _calculate_confidence(self, probabilities: np.ndarray, markov_metadata: Dict) -> float:
        """Calculate prediction confidence."""
        
        # Base confidence from prediction certainty (how skewed the distribution is)
        max_prob = np.max(probabilities)
        
        # Method 1: Use maximum probability directly (more intuitive)
        # Transform max_prob from [0.33, 1.0] to [0.0, 1.0] confidence scale
        prob_confidence = (max_prob - 1/3) / (1 - 1/3)  # Normalize from uniform baseline
        prob_confidence = max(0.0, prob_confidence)  # Clamp to [0, 1]
        
        # Method 2: Use entropy-based measure as backup
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        max_entropy = np.log(3)  # Maximum entropy for 3 outcomes
        entropy_confidence = 1.0 - (entropy / max_entropy)
        
        # Take the maximum of both methods
        certainty = max(prob_confidence, entropy_confidence)
        
        # Adjust based on Markov model confidence
        markov_confidence = markov_metadata.get('confidence', 0.33)
        
        # Combine confidences with more weight on statistical certainty
        base_confidence = 0.8 * certainty + 0.2 * markov_confidence
        
        # Adjust based on game history length (less penalizing for short games)
        history_factor = min(1.0, len(self.move_history) / 5)  # Reach full confidence faster
        confidence = base_confidence * (0.7 + 0.3 * history_factor)  # Less harsh penalty
        
        # Add small boost for higher-order predictions when available
        if 'method' in markov_metadata and 'markov_order' in markov_metadata.get('method', ''):
            confidence += 0.05  # Small boost for Markov predictions
        
        return float(np.clip(confidence, 0.15, 0.95))  # Higher minimum confidence
        
    def get_accuracy(self) -> float:
        """Get current prediction accuracy."""
        if self.prediction_count == 0:
            return 0.0
        return self.correct_predictions / self.prediction_count
        
    def get_opponent_info(self) -> Dict[str, Any]:
        """Get current opponent information."""
        if not self.current_opponent:
            return {}
            
        return {
            'opponent_id': self.current_opponent.opponent_id,
            'difficulty': self.current_opponent.difficulty.value,
            'strategy': self.current_opponent.strategy.value,
            'personality': self.current_opponent.personality.value,
            'description': self.current_opponent.description,
            'expected_win_rate': self.current_opponent.expected_win_rate,
            'alpha': self.current_opponent.alpha,
            'lambda_influence': self.current_opponent.lambda_influence,
            'markov_order': self.current_opponent.markov_order
        }
        
    def get_game_state(self) -> Dict[str, Any]:
        """Get current game state."""
        return {
            'move_history': self.move_history.copy(),
            'outcome_history': self.outcome_history.copy(),
            'prediction_count': self.prediction_count,
            'correct_predictions': self.correct_predictions,
            'accuracy': self.get_accuracy(),
            'confidence_history': self.confidence_history.copy(),
            'game_length': len(self.move_history),
            'session_duration': time.time() - self.session_start_time if self.session_start_time else 0
        }
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        
        if not self.move_history:
            return {'status': 'no_data'}
            
        # Recent performance (last 10 moves)
        recent_outcomes = self.outcome_history[-10:]
        recent_wins = recent_outcomes.count('win')
        recent_ties = recent_outcomes.count('tie')
        recent_losses = recent_outcomes.count('loss')
        
        # Confidence statistics
        if self.confidence_history:
            avg_confidence = np.mean(self.confidence_history)
            confidence_trend = 'stable'
            if len(self.confidence_history) >= 5:
                recent_conf = np.mean(self.confidence_history[-5:])
                early_conf = np.mean(self.confidence_history[:5])
                if recent_conf > early_conf + 0.1:
                    confidence_trend = 'increasing'
                elif recent_conf < early_conf - 0.1:
                    confidence_trend = 'decreasing'
        else:
            avg_confidence = 0.0
            confidence_trend = 'unknown'
            
        return {
            'total_moves': len(self.move_history),
            'accuracy': self.get_accuracy(),
            'recent_performance': {
                'wins': recent_wins,
                'ties': recent_ties,
                'losses': recent_losses,
                'win_rate': recent_wins / len(recent_outcomes) if recent_outcomes else 0
            },
            'confidence_stats': {
                'average': avg_confidence,
                'current': self.confidence_history[-1] if self.confidence_history else 0,
                'trend': confidence_trend
            },
            'opponent_info': self.get_opponent_info()
        }
        
    def save_session(self, filepath: str) -> None:
        """Save current session data."""
        session_data = {
            'meta': {
                'version': '1.0',
                'save_time': time.time(),
                'session_duration': time.time() - self.session_start_time if self.session_start_time else 0
            },
            'opponent': self.current_opponent.to_dict() if self.current_opponent else None,
            'game_state': self.get_game_state(),
            'performance': self.get_performance_summary(),
            'metadata_log': self.metadata_log
        }
        
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)


# Singleton instance for easy access
_ai_system = None

def get_ai_system() -> RPSAISystem:
    """Get the global AI system instance."""
    global _ai_system
    if _ai_system is None:
        _ai_system = RPSAISystem()
    return _ai_system


def initialize_ai_system(difficulty: str, strategy: str, personality: str) -> bool:
    """Initialize AI system with specified opponent."""
    ai_system = get_ai_system()
    return ai_system.set_opponent(difficulty, strategy, personality)


# Test function
def test_ai_system():
    """Test the complete AI system."""
    print("Testing RPS AI System...")
    
    # Initialize system
    ai_system = RPSAISystem()
    
    # Set opponent
    success = ai_system.set_opponent('challenger', 'to_win', 'aggressive')
    print(f"Opponent set: {success}")
    
    if success:
        # Get opponent info
        info = ai_system.get_opponent_info()
        print(f"Opponent: {info['opponent_id']} - {info['description']}")
        
        # Simulate a few moves
        human_moves = ['rock', 'paper', 'rock', 'scissors', 'paper']
        ai_moves = []
        
        for i, human_move in enumerate(human_moves):
            print(f"\n--- Round {i+1} ---")
            
            # AI predicts and moves
            probs, ai_move, metadata = ai_system.predict_next_move()
            ai_moves.append(ai_move)
            print(f"AI predicts human probabilities: {probs}")
            print(f"AI plays: {ai_move}")
            print(f"Confidence: {metadata['confidence']:.3f}")
            
            # Update with actual human move
            update_meta = ai_system.update_with_human_move(human_move, ai_move)
            print(f"Human played: {human_move}")
            print(f"Outcome: {update_meta['outcome']} (human perspective)")
            print(f"Prediction quality: {update_meta['prediction_quality']:.3f}")
            
        # Final summary
        summary = ai_system.get_performance_summary()
        print(f"\n--- Final Summary ---")
        print(f"Total moves: {summary['total_moves']}")
        print(f"AI accuracy: {summary['accuracy']:.3f}")
        print(f"Average confidence: {summary['confidence_stats']['average']:.3f}")
        print(f"Recent human win rate: {summary['recent_performance']['win_rate']:.3f}")


if __name__ == "__main__":
    test_ai_system()