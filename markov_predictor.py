"""
Markov Predictor System for RPS AI
==================================

Implements Markov chain predictors of orders 1-3 with proper smoothing and count tracking.
This replaces the LSTM-based prediction system with a syst        # NEW: Check for frequency bias first (most reliable pattern)
        freq_strength = self._detect_frequency_bias(history)
        if freq_strength > config['frequency_bias']:
            return self._counter_frequency_bias(history, config['exploitation_strength'])
        
        # Check for alternating patterns (length 2)
        pattern_strength = self._detect_alternating_pattern(history)
        if pattern_strength > config['alternating']:
            return self._counter_alternating_pattern(history, config['exploitation_strength'])
        
        # Check for single move repetition
        single_strength = self._detect_single_move_pattern(history)
        if single_strength > config['single_move']:
            return self._counter_single_move_pattern(history, config['exploitation_strength'])
        
        # Check for cycle patterns (length 3+)
        cycle_strength = self._detect_cycle_pattern(history)
        if cycle_strength > config['cycle']:
            return self._counter_cycle_pattern(history, config['exploitation_strength'])eatures:
- Multi-order Markov chains (1st, 2nd, 3rd order)
- Laplace smoothing for unseen sequences
- Count tracking for statistical confidence
- Move mapping utilities for consistent encoding

Author: AI Assistant
Created: 2025-10-03
"""

import numpy as np
from collections import defaultdict, Counter, deque
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
    
    def __init__(self, order: int = 2, smoothing_factor: float = 1.0, pattern_memory_limit: Optional[int] = None, pattern_detection_speed: float = 1.0):
        """
        Initialize Markov predictor.
        
        Args:
            order: Markov order (1, 2, or 3)
            smoothing_factor: Laplace smoothing parameter (default 1.0)
            pattern_memory_limit: Maximum moves to remember (15/25/50 for rookie/challenger/master)
            pattern_detection_speed: Speed of pattern detection (0.5=slow, 1.0=normal, 2.0=fast)
        """
        if order < 1:
            raise ValueError(f"Order must be >= 1. Got {order}")
        if order > 20:  # Reasonable upper limit to prevent memory issues
            raise ValueError(f"Order must be <= 20. Got {order}")
            
        self.order = order
        self.smoothing_factor = smoothing_factor
        self.pattern_memory_limit = pattern_memory_limit
        self.pattern_detection_speed = pattern_detection_speed
        
        # Transition counts: state -> {next_move: count}
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        
        # Total observations for each state
        self.state_counts = defaultdict(int)
        
        # Move history for building states - with memory limit based on difficulty
        if pattern_memory_limit:
            self.move_history = deque(maxlen=pattern_memory_limit)
        else:
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
            # Get the state (previous 'order' moves) - handle both deque and list
            if isinstance(self.move_history, deque):
                history_list = list(self.move_history)
                state = tuple(history_list[-(self.order+1):-1])
                next_move = history_list[-1]
            else:
                state = tuple(self.move_history[-(self.order+1):-1])
                next_move = self.move_history[-1]
            
            # Update counts with pattern detection speed multiplier
            # Higher speed = faster learning, lower speed = slower learning
            count_increment = max(1, int(self.pattern_detection_speed))
            self.transition_counts[state][next_move] += count_increment
            self.state_counts[state] += count_increment
    
    def predict(self, move_history: Optional[List[Union[str, int]]] = None, difficulty_level: str = 'challenger') -> Tuple[np.ndarray, Dict]:
        """
        Predict next move probabilities with enhanced pattern detection.
        
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
            # Convert internal history to list for consistent handling
            if isinstance(self.move_history, deque):
                history = list(self.move_history)
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
        
        # Enhanced pattern detection: Check for repetitive patterns with difficulty scaling
        pattern_prediction = self._detect_and_counter_patterns(history, 'challenger')  # Default to challenger
        if pattern_prediction is not None:
            return pattern_prediction
            
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
    
    def _detect_and_counter_patterns(self, history: List[str], difficulty_level: str = 'medium') -> Optional[Tuple[np.ndarray, Dict]]:
        """
        Enhanced pattern detection and counter-exploitation with difficulty-scaled thresholds.
        
        Args:
            history: List of normalized move strings
            difficulty_level: 'rookie', 'challenger', or 'master' for scaled detection
            
        Returns:
            Tuple of (probabilities, metadata) if pattern detected, None otherwise
        """
        if len(history) < 4:
            return None
        
        # REBALANCED difficulty-scaled detection thresholds
        thresholds = {
            'rookie': {
                'alternating': 0.65,     # Moderate threshold = moderate detection
                'single_move': 0.70,     # Moderate threshold = moderate detection  
                'cycle': 0.65,           # Moderate threshold = moderate detection
                'frequency_bias': 0.55,  # Moderate threshold = moderate detection
                'min_history': 7,        # Moderate history needed
                'exploitation_strength': 0.65  # Moderate exploitation
            },
            'challenger': {
                'alternating': 0.45,     # Good detection
                'single_move': 0.55,     # Good detection
                'cycle': 0.45,           # Good detection
                'frequency_bias': 0.40,  # Good detection
                'min_history': 5,        # Moderate history needed
                'exploitation_strength': 0.80  # Strong exploitation
            },
            'master': {
                'alternating': 0.35,     # Aggressive detection
                'single_move': 0.45,     # Aggressive detection
                'cycle': 0.35,           # Aggressive detection  
                'frequency_bias': 0.30,  # Aggressive detection
                'min_history': 4,        # Quick detection
                'exploitation_strength': 0.95  # Very strong exploitation
            }
        }
        
        # Get thresholds for this difficulty
        config = thresholds.get(difficulty_level, thresholds['challenger'])
        
        # Check minimum history requirement
        if len(history) < config['min_history']:
            return None
        
        # NEW: Check for frequency bias first (most reliable pattern)
        freq_strength = self._detect_frequency_bias(history)
        if freq_strength > config['frequency_bias']:
            return self._counter_frequency_bias(history)
        
        # Check for alternating patterns (length 2)
        pattern_strength = self._detect_alternating_pattern(history)
        if pattern_strength > config['alternating']:
            return self._counter_alternating_pattern(history)
        
        # Check for repeating single move
        single_move_strength = self._detect_single_move_pattern(history)
        if single_move_strength > config['single_move']:
            return self._counter_single_move_pattern(history)
        
        # Check for cycle patterns (length 3+)
        cycle_strength = self._detect_cycle_pattern(history)
        if cycle_strength > config['cycle']:
            return self._counter_cycle_pattern(history)
        
        return None
    
    def _detect_frequency_bias(self, history: List[str], window_size: int = 12) -> float:
        """
        Detect frequency bias patterns more sensitively.
        
        Args:
            history: List of move strings
            window_size: Size of window to analyze (default 12)
            
        Returns:
            Strength of frequency bias (0-1)
        """
        if len(history) < 6:
            return 0.0
        
        # Use shorter window for more responsive detection
        window = min(window_size, len(history))
        recent = history[-window:]
        
        from collections import Counter
        move_counts = Counter(recent)
        total = len(recent)
        
        # Calculate the strongest bias
        max_frequency = max(move_counts.values()) / total
        
        # Expected frequency is 1/3, so bias strength is deviation from uniform
        expected_freq = 1/3
        bias_strength = max(0, max_frequency - expected_freq) / (1 - expected_freq)
        
        return bias_strength
    
    def _counter_frequency_bias(self, history: List[str], exploitation_strength: float = 0.8) -> Tuple[np.ndarray, Dict]:
        """Counter detected frequency bias pattern with difficulty-scaled exploitation."""
        window_size = min(12, len(history))
        recent = history[-window_size:]
        
        from collections import Counter
        move_counts = Counter(recent)
        most_common_move, count = move_counts.most_common(1)[0]
        frequency = count / len(recent)
        
        # Counter the most frequent move with difficulty-scaled aggression
        counter_move = self._get_counter_move(most_common_move)
        
        # Apply exploitation strength (rookie=0.6, challenger=0.8, master=1.0)
        base_counter_prob = 0.33 + (0.6 * exploitation_strength)  # Range: 0.69-0.93
        remaining_prob = 1.0 - base_counter_prob
        other_prob = remaining_prob / 2
        
        probs = np.array([other_prob, other_prob, other_prob])
        counter_idx = MOVES.index(counter_move)
        probs[counter_idx] = base_counter_prob
        
        probs = probs / np.sum(probs)
        
        metadata = {
            'method': 'pattern_counter_frequency_bias',
            'pattern_detected': 'frequency_bias',
            'biased_move': most_common_move,
            'predicted_human_move': most_common_move,
            'frequency': frequency,
            'counter_move': counter_move,
            'exploitation_strength': exploitation_strength,
            'confidence': min(0.95, frequency * exploitation_strength),
            'window_size': window_size
        }
        
        return probs, metadata

    def _detect_alternating_pattern(self, history: List[str]) -> float:
        """Detect alternating patterns like rock-paper-rock-paper."""
        if len(history) < 4:
            return 0.0
        
        # Check last 8 moves for alternating pattern
        recent = history[-8:] if len(history) >= 8 else history
        
        if len(recent) < 4:
            return 0.0
        
        # Check if moves alternate between two values
        alternations = 0
        total_checks = len(recent) - 2
        
        for i in range(len(recent) - 2):
            if recent[i] == recent[i + 2]:  # Same move 2 positions apart
                alternations += 1
        
        return alternations / total_checks if total_checks > 0 else 0.0
    
    def _counter_alternating_pattern(self, history: List[str], exploitation_strength: float = 0.8) -> Tuple[np.ndarray, Dict]:
        """Counter detected alternating pattern with difficulty-scaled exploitation."""
        # Predict what the next move will be based on alternating pattern
        if len(history) >= 4:
            # For true alternating pattern like A-B-A-B, look at the cycle
            last_move = history[-1]
            second_last = history[-2] 
            
            # If it's truly alternating between two moves
            if last_move != second_last:
                # The next move should be the same as second_last (continuing the alternation)
                predicted_move = second_last
            else:
                # Fallback: assume next is different from last
                # Find the other move in the alternating pattern
                recent_moves = history[-6:] if len(history) >= 6 else history
                unique_moves = list(set(recent_moves))
                if len(unique_moves) >= 2:
                    # Find the move that's not the last move
                    predicted_move = next((m for m in unique_moves if m != last_move), last_move)
                else:
                    predicted_move = last_move
        else:
            # Not enough history, just predict different from last
            predicted_move = history[-1] if history else 'rock'
        
        # Counter the predicted move with difficulty-scaled strength
        counter_move = self._get_counter_move(predicted_move)
        
        # Apply exploitation strength (rookie=0.6, challenger=0.8, master=1.0)
        base_counter_prob = 0.33 + (0.5 * exploitation_strength)  # Range: 0.63-0.83
        remaining_prob = 1.0 - base_counter_prob
        other_prob = remaining_prob / 2
        
        probs = np.array([other_prob, other_prob, other_prob])
        counter_idx = MOVES.index(counter_move)
        probs[counter_idx] = base_counter_prob
        
        # Normalize
        probs = probs / np.sum(probs)
        
        metadata = {
            'method': 'pattern_counter_alternating',
            'pattern_detected': 'alternating',
            'predicted_human_move': predicted_move,
            'counter_move': counter_move,
            'exploitation_strength': exploitation_strength,
            'confidence': 0.80 * exploitation_strength  # Scale confidence with strength
        }
        
        return probs, metadata
    
    def _detect_single_move_pattern(self, history: List[str]) -> float:
        """Detect single move repetition patterns."""
        if len(history) < 3:
            return 0.0
        
        # Check last 6 moves for single move dominance
        recent = history[-6:] if len(history) >= 6 else history
        
        from collections import Counter
        move_counts = Counter(recent)
        most_common_count = move_counts.most_common(1)[0][1]
        
        return most_common_count / len(recent)
    
    def _counter_single_move_pattern(self, history: List[str], exploitation_strength: float = 0.8) -> Tuple[np.ndarray, Dict]:
        """Counter detected single move pattern with difficulty-scaled exploitation."""
        from collections import Counter
        recent = history[-6:] if len(history) >= 6 else history
        most_common_move = Counter(recent).most_common(1)[0][0]
        
        counter_move = self._get_counter_move(most_common_move)
        
        # Apply exploitation strength (rookie=0.6, challenger=0.8, master=1.0)
        base_counter_prob = 0.33 + (0.6 * exploitation_strength)  # Range: 0.69-0.93
        remaining_prob = 1.0 - base_counter_prob
        other_prob = remaining_prob / 2
        
        probs = np.array([other_prob, other_prob, other_prob])
        counter_idx = MOVES.index(counter_move)
        probs[counter_idx] = base_counter_prob
        
        probs = probs / np.sum(probs)
        
        metadata = {
            'method': 'pattern_counter_single',
            'pattern_detected': 'single_move_repetition',
            'predicted_human_move': most_common_move,
            'counter_move': counter_move,
            'exploitation_strength': exploitation_strength,
            'confidence': 0.85 * exploitation_strength
        }
        
        return probs, metadata
    
    def _detect_cycle_pattern(self, history: List[str]) -> float:
        """
        Enhanced cycle detection for patterns like rock-paper-scissors-rock-paper-scissors
        and other repeating sequences.
        """
        if len(history) < 6:
            return 0.0
        
        max_strength = 0.0
        
        # Check for cycles of different lengths (2-8)
        for cycle_length in range(2, min(9, len(history) // 2 + 1)):
            strength = self._detect_cycle_of_length(history, cycle_length)
            max_strength = max(max_strength, strength)
        
        return max_strength
    
    def _detect_cycle_of_length(self, history: List[str], cycle_length: int) -> float:
        """Detect cycle of specific length."""
        if len(history) < cycle_length * 2:
            return 0.0
        
        # Look at recent history for cycles
        max_check_length = min(cycle_length * 4, len(history))
        recent = history[-max_check_length:]
        
        if len(recent) < cycle_length * 2:
            return 0.0
        
        # Extract the potential cycle pattern
        pattern = recent[:cycle_length]
        
        # Count how many times this pattern repeats
        matches = 0
        total_checks = 0
        
        for i in range(cycle_length, len(recent), cycle_length):
            if i + cycle_length <= len(recent):
                segment = recent[i:i + cycle_length]
                if segment == pattern:
                    matches += 1
                total_checks += 1
        
        if total_checks == 0:
            return 0.0
            
        return matches / total_checks
    
    def _counter_cycle_pattern(self, history: List[str], exploitation_strength: float = 0.8) -> Tuple[np.ndarray, Dict]:
        """Enhanced counter for detected cycle patterns with difficulty-scaled exploitation."""
        
        best_cycle = None
        best_strength = 0.0
        best_length = 0
        
        # Find the strongest cycle
        for cycle_length in range(2, min(9, len(history) // 2 + 1)):
            strength = self._detect_cycle_of_length(history, cycle_length)
            if strength > best_strength:
                best_strength = strength
                best_length = cycle_length
                # Extract the cycle pattern
                max_check_length = min(cycle_length * 4, len(history))
                recent = history[-max_check_length:]
                best_cycle = recent[:cycle_length]
        
        if best_cycle is None:
            # Fallback to simple RPS cycle
            rps_cycle = ['rock', 'paper', 'scissors']
            cycle_position = len(history) % 3
            predicted_move = rps_cycle[cycle_position]
        else:
            # Predict next move in detected cycle
            position_in_cycle = len(history) % best_length
            predicted_move = best_cycle[position_in_cycle]
        
        counter_move = self._get_counter_move(predicted_move)
        
        # Apply exploitation strength (rookie=0.6, challenger=0.8, master=1.0)
        base_counter_prob = 0.33 + (0.45 * exploitation_strength)  # Range: 0.60-0.78
        remaining_prob = 1.0 - base_counter_prob
        other_prob = remaining_prob / 2
        
        probs = np.array([other_prob, other_prob, other_prob])
        counter_idx = MOVES.index(counter_move)
        probs[counter_idx] = base_counter_prob
        
        probs = probs / np.sum(probs)
        
        metadata = {
            'method': 'pattern_counter_cycle_enhanced',
            'pattern_detected': 'cycle_pattern',
            'cycle_pattern': best_cycle if best_cycle else ['rock', 'paper', 'scissors'],
            'cycle_length': best_length if best_cycle else 3,
            'cycle_strength': best_strength,
            'predicted_human_move': predicted_move,
            'counter_move': counter_move,
            'exploitation_strength': exploitation_strength,
            'confidence': min(0.9, (best_strength + 0.2) * exploitation_strength)
        }
        
        return probs, metadata
    
    def _get_counter_move(self, move: str) -> str:
        """Get the move that beats the given move."""
        counters = {'rock': 'paper', 'paper': 'scissors', 'scissors': 'rock'}
        return counters.get(move, 'rock')
        
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
    
    def __init__(self, orders: List[int] = [1, 2, 3], smoothing_factor: float = 1.0, ensemble_weights: Optional[Dict[int, float]] = None, pattern_memory_limit: Optional[int] = None, pattern_detection_speed: float = 1.0):
        """
        Initialize ensemble predictor.
        
        Args:
            orders: List of Markov orders to include
            smoothing_factor: Laplace smoothing parameter
            ensemble_weights: Fixed weights for each order {order: weight}. If None, uses adaptive weighting.
            pattern_memory_limit: Maximum moves to remember (15/25/50 for rookie/challenger/master)
            pattern_detection_speed: Speed of pattern detection (0.5=slow, 1.0=normal, 2.0=fast)
        """
        self.predictors = {}
        for order in orders:
            self.predictors[order] = MarkovPredictor(order, smoothing_factor, pattern_memory_limit, pattern_detection_speed)
            
        self.orders = orders
        self.ensemble_weights = ensemble_weights
        self.pattern_memory_limit = pattern_memory_limit
        self.pattern_detection_speed = pattern_detection_speed
        
    def update(self, move: Union[str, int]) -> None:
        """Update all predictors with new move."""
        for predictor in self.predictors.values():
            predictor.update(move)
            
    def predict(self, move_history: Optional[List[Union[str, int]]] = None, difficulty_level: str = 'challenger') -> Tuple[np.ndarray, Dict]:
        """
        Predict using ensemble of predictors with pattern detection.
        
        Args:
            move_history: Optional move history to use
            difficulty_level: 'rookie', 'challenger', or 'master' for pattern detection scaling
            
        Returns:
            Tuple of (probabilities, metadata)
        """
        # Normalize move history like individual predictors do
        if move_history is not None:
            history = []
            for move in move_history:
                if isinstance(move, int):
                    move = number_to_move(move)
                history.append(normalize_move(move))
        else:
            # Use history from first predictor (they should all be the same)
            history = list(self.predictors[self.orders[0]].move_history)
        
        # PRIORITY 1: Check for patterns first (like individual predictors) with difficulty scaling
        if len(history) >= 4:
            # Use the pattern detection from the first predictor with difficulty scaling
            base_predictor = self.predictors[self.orders[0]]
            pattern_result = base_predictor._detect_and_counter_patterns(history, difficulty_level)
            if pattern_result is not None:
                # Pattern detected! Use it with high priority
                pattern_probs, pattern_metadata = pattern_result
                pattern_metadata['method'] += '_ensemble'  # Mark as ensemble pattern detection
                
                # Apply adaptive epsilon reduction for patterns (like in RPS AI system)
                pattern_metadata['epsilon_reduced'] = True
                
                return pattern_probs, pattern_metadata
        
        # PRIORITY 2: No pattern detected, use normal ensemble logic
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