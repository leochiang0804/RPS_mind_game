"""
Markov Predictor System for RPS AI
==================================

Implements Markov chain predictors of orders 1-3 with proper smoothing and count tracking.
This replaces the LSTM-based prediction system with a systematic approach.

Features:
- Multi-order Markov chains (1st, 2nd, 3rd order)
- Laplace smoothing for unseen sequences
- Count tracking for statistical confidence
- Move mapping utilities for consistent encoding

Author: AI Assistant
Created: 2025-10-03
"""

import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Union
import json
from move_mapping import (
    MOVES, MOVE_TO_NUMBER, NUMBER_TO_MOVE, 
    normalize_move, move_to_number, number_to_move
)


class MarkovPredictor:
    """
    Multi-order Markov chain predictor for Rock-Paper-Scissors.
    
    Supports orders 1-3 with Laplace smoothing and count tracking.
    """
    
    def __init__(self, order: int = 2, smoothing_factor: float = 1.0):
        """
        Initialize Markov predictor.
        
        Args:
            order: Markov order (1, 2, or 3)
            smoothing_factor: Laplace smoothing parameter (default 1.0)
        """
        if order not in [1, 2, 3]:
            raise ValueError(f"Order must be 1, 2, or 3. Got {order}")
            
        self.order = order
        self.smoothing_factor = smoothing_factor
        
        # Transition counts: state -> {next_move: count}
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        
        # Total observations for each state
        self.state_counts = defaultdict(int)
        
        # Move history for building states
        self.move_history = []
        
        # Statistics
        self.total_predictions = 0
        self.correct_predictions = 0
        
    def update(self, move: Union[str, int]) -> None:
        """
        Update the Markov model with a new move.
        
        Args:
            move: New move ('rock', 'paper', 'scissors' or 0, 1, 2)
        """
        # Normalize move to string
        if isinstance(move, int):
            move = number_to_move(move)
        move = normalize_move(move)
        
        if move not in MOVES:
            raise ValueError(f"Invalid move: {move}")
            
        # Add to history
        self.move_history.append(move)
        
        # Update transition counts if we have enough history
        if len(self.move_history) > self.order:
            # Get the state (previous 'order' moves)
            state = tuple(self.move_history[-(self.order+1):-1])
            next_move = self.move_history[-1]
            
            # Update counts
            self.transition_counts[state][next_move] += 1
            self.state_counts[state] += 1
    
    def predict(self, move_history: Optional[List[Union[str, int]]] = None) -> Tuple[np.ndarray, Dict]:
        """
        Predict next move probabilities.
        
        Args:
            move_history: Optional move history to use instead of internal history
            
        Returns:
            Tuple of (probabilities, metadata)
            - probabilities: Array of [P(rock), P(paper), P(scissors)]
            - metadata: Dict with prediction details
        """
        # Use provided history or internal history
        if move_history is not None:
            # Normalize history
            history = []
            for move in move_history:
                if isinstance(move, int):
                    move = number_to_move(move)
                history.append(normalize_move(move))
        else:
            history = self.move_history.copy()
            
        # Check if we have enough history
        if len(history) < self.order:
            # Not enough history - return uniform distribution
            probs = np.array([1/3, 1/3, 1/3])
            metadata = {
                'method': 'uniform',
                'reason': f'insufficient_history',
                'history_length': len(history),
                'required_length': self.order,
                'state': None,
                'state_count': 0,
                'confidence': 0.33
            }
            return probs, metadata
            
        # Get current state (last 'order' moves)
        state = tuple(history[-self.order:])
        
        # Get transition counts for this state
        next_move_counts = self.transition_counts[state]
        state_total = self.state_counts[state]
        
        # Calculate probabilities with Laplace smoothing
        probs = np.zeros(3)
        for i, move in enumerate(MOVES):
            count = next_move_counts[move]
            # Laplace smoothing: (count + α) / (total + α * vocab_size)
            probs[i] = (count + self.smoothing_factor) / (state_total + self.smoothing_factor * 3)
            
        # Calculate confidence based on state frequency
        confidence = min(0.95, state_total / (state_total + 10))  # Sigmoid-like confidence
        
        # Metadata
        metadata = {
            'method': f'markov_order_{self.order}',
            'state': state,
            'state_count': state_total,
            'transition_counts': dict(next_move_counts),
            'confidence': confidence,
            'smoothing_used': state_total < 3
        }
        
        return probs, metadata
        
    def evaluate_prediction(self, predicted_probs: np.ndarray, actual_move: Union[str, int]) -> float:
        """
        Evaluate a prediction against the actual move.
        
        Args:
            predicted_probs: Predicted probabilities [P(rock), P(paper), P(scissors)]
            actual_move: Actual move that occurred
            
        Returns:
            Prediction quality score (higher is better)
        """
        # Normalize actual move
        if isinstance(actual_move, int):
            actual_move = number_to_move(actual_move)
        actual_move = normalize_move(actual_move)
        
        if actual_move not in MOVES:
            return 0.0
            
        # Get probability assigned to actual move
        actual_idx = MOVE_TO_NUMBER[actual_move]
        assigned_prob = float(predicted_probs[actual_idx])
        
        # Update statistics
        self.total_predictions += 1
        predicted_move = MOVES[np.argmax(predicted_probs)]
        if predicted_move == actual_move:
            self.correct_predictions += 1
            
        return assigned_prob
        
    def get_accuracy(self) -> float:
        """Get current prediction accuracy."""
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions
        
    def reset(self) -> None:
        """Reset the predictor to initial state."""
        self.transition_counts.clear()
        self.state_counts.clear()
        self.move_history.clear()
        self.total_predictions = 0
        self.correct_predictions = 0
        
    def get_state_summary(self) -> Dict:
        """Get summary of current model state."""
        total_states = len(self.transition_counts)
        total_transitions = sum(self.state_counts.values())
        
        # Most common states
        top_states = sorted(self.state_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'order': self.order,
            'total_states_seen': total_states,
            'total_transitions': total_transitions,
            'history_length': len(self.move_history),
            'accuracy': self.get_accuracy(),
            'top_states': [(list(state), count) for state, count in top_states],
            'smoothing_factor': self.smoothing_factor
        }
        
    def save_model(self, filepath: str) -> None:
        """Save model state to file."""
        state = {
            'order': self.order,
            'smoothing_factor': self.smoothing_factor,
            'transition_counts': {
                str(state): dict(counts) 
                for state, counts in self.transition_counts.items()
            },
            'state_counts': dict(self.state_counts),
            'move_history': self.move_history,
            'total_predictions': self.total_predictions,
            'correct_predictions': self.correct_predictions
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
            
    def load_model(self, filepath: str) -> None:
        """Load model state from file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
            
        self.order = state['order']
        self.smoothing_factor = state['smoothing_factor']
        
        # Reconstruct transition counts
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        for state_str, counts in state['transition_counts'].items():
            state_tuple = eval(state_str)  # Safe since we control the format
            for move, count in counts.items():
                self.transition_counts[state_tuple][move] = count
                
        self.state_counts = defaultdict(int, state['state_counts'])
        self.move_history = state['move_history']
        self.total_predictions = state['total_predictions']
        self.correct_predictions = state['correct_predictions']


class EnsembleMarkovPredictor:
    """
    Ensemble of multiple Markov predictors with different orders.
    
    Combines predictions from 1st, 2nd, and 3rd order Markov chains
    with adaptive weighting based on confidence.
    """
    
    def __init__(self, orders: List[int] = [1, 2, 3], smoothing_factor: float = 1.0, ensemble_weights: Optional[Dict[int, float]] = None):
        """
        Initialize ensemble predictor.
        
        Args:
            orders: List of Markov orders to include
            smoothing_factor: Laplace smoothing parameter
            ensemble_weights: Fixed weights for each order {order: weight}. If None, uses adaptive weighting.
        """
        self.predictors = {}
        for order in orders:
            self.predictors[order] = MarkovPredictor(order, smoothing_factor)
            
        self.orders = orders
        self.ensemble_weights = ensemble_weights
        
    def update(self, move: Union[str, int]) -> None:
        """Update all predictors with new move."""
        for predictor in self.predictors.values():
            predictor.update(move)
            
    def predict(self, move_history: Optional[List[Union[str, int]]] = None) -> Tuple[np.ndarray, Dict]:
        """
        Predict using ensemble of predictors.
        
        Args:
            move_history: Optional move history to use
            
        Returns:
            Tuple of (probabilities, metadata)
        """
        # Get predictions from all predictors
        predictions = {}
        confidences = {}
        
        for order, predictor in self.predictors.items():
            probs, metadata = predictor.predict(move_history)
            predictions[order] = probs
            confidences[order] = metadata['confidence']
            
        # Adaptive weighting based on confidence OR fixed ensemble weights
        if self.ensemble_weights is not None:
            # Use fixed ensemble weights from PSE configuration
            weights = {}
            for order in self.orders:
                weights[order] = self.ensemble_weights.get(order, 0.0)
            # Normalize weights to sum to 1
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {order: w/total_weight for order, w in weights.items()}
            else:
                # Fallback to equal weights
                weights = {order: 1/len(self.orders) for order in self.orders}
        else:
            # Original adaptive weighting based on confidence
            total_confidence = sum(confidences.values())
            if total_confidence == 0:
                # Equal weights if no confidence
                weights = {order: 1/len(self.orders) for order in self.orders}
            else:
                weights = {order: conf/total_confidence for order, conf in confidences.items()}
            
        # Combine predictions
        ensemble_probs = np.zeros(3)
        for order, probs in predictions.items():
            ensemble_probs += weights[order] * probs
            
        # Metadata
        metadata = {
            'method': 'ensemble_markov',
            'individual_predictions': {order: probs.tolist() for order, probs in predictions.items()},
            'confidences': confidences,
            'weights': weights,
            'ensemble_confidence': max(confidences.values()) if confidences else 0.33
        }
        
        return ensemble_probs, metadata
        
    def evaluate_prediction(self, predicted_probs: np.ndarray, actual_move: Union[str, int]) -> float:
        """Evaluate prediction for all constituent predictors."""
        scores = []
        for predictor in self.predictors.values():
            score = predictor.evaluate_prediction(predicted_probs, actual_move)
            scores.append(score)
        return float(np.mean(scores))
        
    def get_accuracy(self) -> Dict[int, float]:
        """Get accuracy for each predictor."""
        return {order: predictor.get_accuracy() for order, predictor in self.predictors.items()}
        
    def reset(self) -> None:
        """Reset all predictors."""
        for predictor in self.predictors.values():
            predictor.reset()
            
    def get_state_summary(self) -> Dict:
        """Get summary of ensemble state."""
        return {
            'ensemble_orders': self.orders,
            'individual_summaries': {order: predictor.get_state_summary() 
                                   for order, predictor in self.predictors.items()}
        }


def create_markov_predictor(order: int = 2, smoothing_factor: float = 1.0) -> MarkovPredictor:
    """
    Factory function to create a Markov predictor.
    
    Args:
        order: Markov order (1, 2, or 3)
        smoothing_factor: Laplace smoothing parameter
        
    Returns:
        Configured MarkovPredictor instance
    """
    return MarkovPredictor(order, smoothing_factor)


def create_ensemble_predictor(orders: List[int] = [1, 2, 3], smoothing_factor: float = 1.0, ensemble_weights: Optional[Dict[int, float]] = None) -> EnsembleMarkovPredictor:
    """
    Factory function to create an ensemble Markov predictor.
    
    Args:
        orders: List of Markov orders to include
        smoothing_factor: Laplace smoothing parameter
        ensemble_weights: Fixed weights for each order
        
    Returns:
        Configured EnsembleMarkovPredictor instance
    """
    return EnsembleMarkovPredictor(orders, smoothing_factor, ensemble_weights)


# Test function
def test_markov_predictor():
    """Test the Markov predictor with sample data."""
    print("Testing Markov Predictor...")
    
    # Create predictor
    predictor = MarkovPredictor(order=2)
    
    # Sample sequence: rock -> paper -> scissors -> rock -> paper
    sequence = ['rock', 'paper', 'scissors', 'rock', 'paper']
    
    for i, move in enumerate(sequence):
        if i >= 2:  # Can make predictions after 2 moves for order-2
            probs, metadata = predictor.predict()
            print(f"After moves {sequence[:i]}, predicted: {probs}")
            print(f"Metadata: {metadata}")
            
        # Update with actual move
        predictor.update(move)
        
    # Final state
    print(f"\nFinal state: {predictor.get_state_summary()}")
    
    # Test ensemble
    print("\nTesting Ensemble Predictor...")
    ensemble = EnsembleMarkovPredictor()
    
    for move in sequence:
        ensemble.update(move)
        
    probs, metadata = ensemble.predict()
    print(f"Ensemble prediction: {probs}")
    print(f"Ensemble metadata: {metadata}")


if __name__ == "__main__":
    test_markov_predictor()