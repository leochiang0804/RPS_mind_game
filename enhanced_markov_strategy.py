"""
Enhanced Markov Strategy with Advanced Pattern Recognition
Improved Markov chain implementation with multi-order chains and pattern detection
"""

import random
from collections import defaultdict, deque, Counter
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from move_mapping import normalize_move, get_counter_move, MOVES

class EnhancedMarkovStrategy:
    """
    Enhanced Markov strategy with multiple order chains and pattern recognition
    """
    
    def __init__(self, max_order=5, confidence_threshold=0.6):
        self.max_order = max_order
        self.confidence_threshold = confidence_threshold
        
        # Multiple order Markov chains
        self.chains = {}  # order -> chain dictionary
        for order in range(1, max_order + 1):
            self.chains[order] = defaultdict(lambda: defaultdict(int))
        
        # Pattern recognition
        self.pattern_detector = PatternDetector()
        
        # Frequency analysis
        self.move_frequencies = defaultdict(int)
        self.transition_frequencies = defaultdict(lambda: defaultdict(int))
        
        # Adaptive parameters
        self.learning_rate = 0.1
        self.decay_factor = 0.95
        
        # Performance tracking
        self.prediction_history = []
        self.accuracy_history = []
        self._last_confidence = 0.33
        
    def train(self, history: List[str]):
        """Enhanced training with multiple strategies"""
        if len(history) < 2:
            return
        
        # Normalize history
        normalized_history = [normalize_move(move) for move in history]
        
        # Update frequency counts
        for move in normalized_history:
            self.move_frequencies[move] += 1
        
        # Update transition frequencies
        for i in range(len(normalized_history) - 1):
            current_move = normalized_history[i]
            next_move = normalized_history[i + 1]
            self.transition_frequencies[current_move][next_move] += 1
        
        # Train multiple order Markov chains
        for order in range(1, min(self.max_order + 1, len(normalized_history))):
            for i in range(len(normalized_history) - order):
                context = tuple(normalized_history[i:i + order])
                next_move = normalized_history[i + order]
                self.chains[order][context][next_move] += 1
        
        # Update pattern detector
        self.pattern_detector.update(normalized_history)
    
    def predict(self, history: List[str]) -> Tuple[str, float]:
        """Enhanced prediction combining multiple strategies"""
        if len(history) == 0:
            return random.choice(MOVES), 0.33
        
        normalized_history = [normalize_move(move) for move in history]
        
        # Get predictions from different strategies
        predictions = []
        
        # 1. Multi-order Markov prediction
        markov_pred, markov_conf = self._predict_markov(normalized_history)
        predictions.append(('markov', markov_pred, markov_conf))
        
        # 2. Pattern-based prediction
        pattern_pred, pattern_conf = self._predict_pattern(normalized_history)
        predictions.append(('pattern', pattern_pred, pattern_conf))
        
        # 3. Frequency-based prediction with trends
        freq_pred, freq_conf = self._predict_frequency(normalized_history)
        predictions.append(('frequency', freq_pred, freq_conf))
        
        # 4. Adaptive ensemble prediction
        final_pred, final_conf = self._ensemble_predict(predictions, normalized_history)
        
        # Store confidence for later retrieval
        self._last_confidence = final_conf
        
        # Convert to robot move (counter the predicted human move)
        robot_move = get_counter_move(final_pred)
        
        return robot_move, final_conf
    
    def _predict_markov(self, history: List[str]) -> Tuple[str, float]:
        """Multi-order Markov chain prediction"""
        best_prediction = random.choice(MOVES)
        best_confidence = 0.33
        
        # Try different orders from highest to lowest
        for order in range(min(self.max_order, len(history)), 0, -1):
            if len(history) >= order:
                context = tuple(history[-order:])
                
                if context in self.chains[order]:
                    chain = self.chains[order][context]
                    total_count = sum(chain.values())
                    
                    if total_count > 0:
                        # Find most likely next move
                        most_likely_move = max(chain.keys(), key=lambda k: chain[k])
                        confidence = chain[most_likely_move] / total_count
                        
                        # Use this prediction if confidence is high enough
                        if confidence > best_confidence:
                            best_prediction = most_likely_move
                            best_confidence = confidence
                            break  # Use highest order with good confidence
        
        return best_prediction, best_confidence
    
    def _predict_pattern(self, history: List[str]) -> Tuple[str, float]:
        """Pattern-based prediction"""
        pattern_pred = self.pattern_detector.predict_next(history)
        confidence = self.pattern_detector.get_confidence()
        return pattern_pred, confidence
    
    def _predict_frequency(self, history: List[str]) -> Tuple[str, float]:
        """Enhanced frequency prediction with recent bias"""
        if not history:
            return random.choice(MOVES), 0.33
        
        # Weight recent moves more heavily
        recent_window = min(20, len(history))
        recent_history = history[-recent_window:]
        
        # Calculate weighted frequencies
        weights = np.exp(np.linspace(-1, 0, len(recent_history)))  # Exponential decay
        weighted_counts = defaultdict(float)
        
        for i, move in enumerate(recent_history):
            weighted_counts[move] += weights[i]
        
        if not weighted_counts:
            return random.choice(MOVES), 0.33
        
        # Predict most frequent move
        most_frequent = max(weighted_counts.keys(), key=lambda k: weighted_counts[k])
        total_weight = sum(weighted_counts.values())
        confidence = weighted_counts[most_frequent] / total_weight
        
        return most_frequent, confidence
    
    def _ensemble_predict(self, predictions: List[Tuple[str, str, float]], 
                         history: List[str]) -> Tuple[str, float]:
        """Ensemble prediction combining all strategies"""
        if not predictions:
            return random.choice(MOVES), 0.33
        
        # Adaptive weights based on recent performance
        strategy_weights = {
            'markov': 0.5,
            'pattern': 0.3,
            'frequency': 0.2
        }
        
        # Adjust weights based on recent accuracy
        if len(self.accuracy_history) > 10:
            recent_accuracy = np.mean(self.accuracy_history[-10:])
            if recent_accuracy < 0.4:  # Poor performance, be more exploratory
                strategy_weights['pattern'] += 0.2
                strategy_weights['markov'] -= 0.1
                strategy_weights['frequency'] -= 0.1
        
        # Weight predictions by confidence and strategy weight
        move_scores = defaultdict(float)
        total_weight = 0
        
        for strategy, move, confidence in predictions:
            weight = strategy_weights.get(strategy, 0.1) * confidence
            move_scores[move] += weight
            total_weight += weight
        
        if total_weight == 0:
            return random.choice(MOVES), 0.33
        
        # Select best move
        best_move = max(move_scores.keys(), key=lambda k: move_scores[k])
        final_confidence = move_scores[best_move] / total_weight
        
        return best_move, final_confidence
    
    def get_confidence(self) -> float:
        """Get current prediction confidence"""
        if not hasattr(self, '_last_confidence'):
            return 0.33
        return self._last_confidence
    
    def update_performance(self, predicted_move: str, actual_move: str):
        """Update performance tracking"""
        correct = predicted_move == actual_move
        self.accuracy_history.append(1.0 if correct else 0.0)
        
        # Keep only recent history
        if len(self.accuracy_history) > 50:
            self.accuracy_history = self.accuracy_history[-50:]


class PatternDetector:
    """Advanced pattern detection for Rock Paper Scissors"""
    
    def __init__(self):
        self.patterns = {}  # pattern -> next_move frequency
        self.cycle_detector = CycleDetector()
        self.sequence_patterns = defaultdict(lambda: defaultdict(int))
        
        # Common RPS patterns
        self.known_patterns = {
            ('rock', 'rock'): 'paper',  # Counter double rock
            ('paper', 'paper'): 'scissors',  # Counter double paper
            ('scissors', 'scissors'): 'rock',  # Counter double scissors
            ('rock', 'paper', 'scissors'): 'rock',  # RPS cycle
            ('scissors', 'paper', 'rock'): 'scissors',  # Reverse cycle
        }
    
    def update(self, history: List[str]):
        """Update pattern knowledge"""
        if len(history) < 3:
            return
        
        # Update cycle detector
        self.cycle_detector.update(history)
        
        # Extract patterns of different lengths
        for pattern_length in range(2, min(6, len(history))):
            for i in range(len(history) - pattern_length):
                pattern = tuple(history[i:i + pattern_length])
                if i + pattern_length < len(history):
                    next_move = history[i + pattern_length]
                    self.sequence_patterns[pattern][next_move] += 1
    
    def predict_next(self, history: List[str]) -> str:
        """Predict next move based on patterns"""
        if len(history) < 2:
            return random.choice(MOVES)
        
        # Check for cycles first
        cycle_pred = self.cycle_detector.predict_next(history)
        if cycle_pred:
            return cycle_pred
        
        # Check known patterns
        for pattern_length in range(min(5, len(history)), 1, -1):
            if len(history) >= pattern_length:
                recent_pattern = tuple(history[-pattern_length:])
                
                # Check known patterns
                if recent_pattern in self.known_patterns:
                    return self.known_patterns[recent_pattern]
                
                # Check learned patterns
                if recent_pattern in self.sequence_patterns:
                    pattern_counts = self.sequence_patterns[recent_pattern]
                    if pattern_counts:
                        return max(pattern_counts.keys(), key=lambda k: pattern_counts[k])
        
        return random.choice(MOVES)
    
    def get_confidence(self) -> float:
        """Get confidence in pattern prediction"""
        # This could be enhanced with more sophisticated confidence calculation
        return 0.6  # Moderate confidence for pattern-based predictions


class CycleDetector:
    """Detect and predict cyclical patterns"""
    
    def __init__(self):
        self.cycles = []
        self.recent_moves = deque(maxlen=20)
    
    def update(self, history: List[str]):
        """Update cycle detection"""
        self.recent_moves.extend(history)
    
    def predict_next(self, history: List[str]) -> Optional[str]:
        """Predict next move if a cycle is detected"""
        if len(history) < 6:
            return None
        
        recent = history[-6:]
        
        # Check for simple cycles
        if len(recent) >= 6:
            # Check for 3-move cycle (RPS or SRP)
            if recent[-6:-3] == recent[-3:]:
                cycle = recent[-3:]
                next_in_cycle = cycle[0]  # Next move in the cycle
                return next_in_cycle
            
            # Check for 2-move cycle
            if recent[-4:-2] == recent[-2:]:
                cycle = recent[-2:]
                next_in_cycle = cycle[0]
                return next_in_cycle
        
        return None


# Update the original MarkovStrategy to use enhanced version
class MarkovStrategy:
    """Enhanced Markov Strategy wrapper for backward compatibility"""
    
    def __init__(self):
        self.enhanced_markov = EnhancedMarkovStrategy()
    
    def train(self, history: List[str]):
        """Train the enhanced Markov model"""
        self.enhanced_markov.train(history)
    
    def predict(self, history: List[str]) -> Tuple[str, float]:
        """Predict using enhanced Markov model"""
        return self.enhanced_markov.predict(history)
    
    def get_confidence(self) -> float:
        """Get prediction confidence"""
        return self.enhanced_markov.get_confidence()


if __name__ == "__main__":
    # Test enhanced Markov strategy
    strategy = EnhancedMarkovStrategy()
    test_history = ['rock', 'paper', 'scissors', 'rock', 'paper', 'scissors']
    strategy.train(test_history)
    
    prediction, confidence = strategy.predict(test_history)
    print(f"Enhanced Markov prediction: {prediction} (confidence: {confidence:.3f})")