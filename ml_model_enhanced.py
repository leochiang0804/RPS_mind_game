# Enhanced Markov Chain ML model with recency weighting and validation
import random
from collections import defaultdict, deque
import json

class EnhancedMLModel:
    def __init__(self, order=2, recency_weight=0.8, max_history=500):
        """
        Enhanced ML model with configurable parameters
        
        Args:
            order: How many previous moves to consider (1-3)
            recency_weight: Weight decay for older moves (0.5-0.95)
            max_history: Maximum history to keep in memory
        """
        self.order = min(max(order, 1), 3)  # Clamp between 1-3
        self.recency_weight = max(0.5, min(recency_weight, 0.95))  # Clamp between 0.5-0.95
        self.max_history = max_history
        
        # Transition tables for different orders
        self.transitions = defaultdict(lambda: defaultdict(float))
        self.history = deque(maxlen=max_history)
        
        # Statistics for validation
        self.stats = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'confidence_scores': [],
            'training_rounds': 0
        }

    def _get_sequence_key(self, moves, order):
        """Convert move sequence to string key"""
        if len(moves) < order:
            return None
        return '|'.join(moves[-order:])

    def train(self, history):
        """Train model on historical data with recency weighting"""
        if not history:
            return
            
        self.history.clear()
        self.history.extend(history[-self.max_history:])
        
        # Reset transitions
        self.transitions.clear()
        
        # Build transitions for all orders up to self.order
        for order in range(1, self.order + 1):
            if len(history) <= order:
                continue
                
            for i in range(order, len(history)):
                # Get sequence of previous moves
                sequence = history[i-order:i]
                next_move = history[i]
                
                seq_key = self._get_sequence_key(sequence, order)
                if seq_key:
                    # Apply recency weighting (more recent = higher weight)
                    weight = self.recency_weight ** (len(history) - i - 1)
                    key = f"order_{order}:{seq_key}"
                    self.transitions[key][next_move] += weight
        
        self.stats['training_rounds'] += 1

    def predict(self, history):
        """Predict next move with confidence score"""
        counter = {'paper': 'scissor', 'scissor': 'stone', 'stone': 'paper'}
        
        if not history:
            return random.choice(['paper', 'scissor', 'stone']), 0.33
        
        # Try different orders, starting from highest
        best_prediction = None
        best_confidence = 0
        
        for order in range(self.order, 0, -1):
            if len(history) < order:
                continue
                
            sequence = history[-order:]
            seq_key = self._get_sequence_key(sequence, order)
            
            if seq_key:
                key = f"order_{order}:{seq_key}"
                if key in self.transitions:
                    next_probs = self.transitions[key]
                    
                    if next_probs:
                        # Calculate probabilities
                        total = sum(next_probs.values())
                        if total > 0:
                            probs = {move: count/total for move, count in next_probs.items()}
                            predicted_move = max(probs.keys(), key=lambda k: probs[k])
                            confidence = probs[predicted_move]
                            
                            if confidence > best_confidence:
                                best_prediction = predicted_move
                                best_confidence = confidence
                                break
        
        # Fallback to frequency analysis if no pattern found
        if best_prediction is None:
            if len(self.history) > 0:
                freq = defaultdict(float)
                # Weight recent moves more heavily
                for i, move in enumerate(self.history):
                    weight = self.recency_weight ** (len(self.history) - i - 1)
                    freq[move] += weight
                
                if freq:
                    total = sum(freq.values())
                    best_prediction = max(freq.keys(), key=lambda k: freq[k])
                    best_confidence = freq[best_prediction] / total
                else:
                    best_prediction = random.choice(['paper', 'scissor', 'stone'])
                    best_confidence = 0.33
            else:
                best_prediction = random.choice(['paper', 'scissor', 'stone'])
                best_confidence = 0.33
        
        # Return counter move
        robot_move = counter[best_prediction]
        return robot_move, best_confidence

    def update_stats(self, predicted_human_move, actual_human_move):
        """Update prediction statistics for validation"""
        self.stats['total_predictions'] += 1
        if predicted_human_move == actual_human_move:
            self.stats['correct_predictions'] += 1

    def get_accuracy(self):
        """Get current prediction accuracy"""
        if self.stats['total_predictions'] == 0:
            return 0.0
        return self.stats['correct_predictions'] / self.stats['total_predictions']

    def get_stats(self):
        """Get detailed statistics"""
        accuracy = self.get_accuracy()
        return {
            'accuracy': accuracy,
            'total_predictions': self.stats['total_predictions'],
            'correct_predictions': self.stats['correct_predictions'],
            'training_rounds': self.stats['training_rounds'],
            'model_order': self.order,
            'recency_weight': self.recency_weight,
            'history_size': len(self.history)
        }

    def export_config(self):
        """Export model configuration for reproducibility"""
        return {
            'order': self.order,
            'recency_weight': self.recency_weight,
            'max_history': self.max_history,
            'stats': self.get_stats()
        }

# Legacy wrapper for backward compatibility
class MLModel:
    def __init__(self):
        self.enhanced_model = EnhancedMLModel(order=1, recency_weight=0.9)
        self.last_move = None

    def train(self, history):
        self.enhanced_model.train(history)
        self.last_move = history[-1] if history else None

    def predict(self, history):
        robot_move, confidence = self.enhanced_model.predict(history)
        return robot_move