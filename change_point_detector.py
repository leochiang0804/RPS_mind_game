"""
Change-point detection for Rock-Paper-Scissors strategy analysis
Detects when human player changes their playing strategy
"""

from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional
import math

class ChangePointDetector:
    def __init__(self, window_size=10, min_segment_length=5, chi2_threshold=6.0):
        """
        Initialize change-point detector
        
        Args:
            window_size: Size of sliding window for feature calculation
            min_segment_length: Minimum length of a strategy segment
            chi2_threshold: Chi-squared threshold for detecting significant changes
        """
        self.window_size = window_size
        self.min_segment_length = min_segment_length
        self.chi2_threshold = chi2_threshold
        
        # History storage
        self.move_history = deque(maxlen=1000)
        self.feature_history = []
        self.change_points = []
        
        # Feature tracking
        self.last_features = None
        
    def _calculate_features(self, moves: List[str]) -> Dict[str, float]:
        """Calculate strategy features for a sequence of moves"""
        if len(moves) < 2:
            return {
                'repeat_prob': 0.0,
                'cycle_score': 0.0,
                'switch_prob': 0.0,
                'entropy': 0.0,
                'bias_paper': 0.0,
                'bias_scissor': 0.0,
                'bias_stone': 0.0
            }
        
        # Basic probabilities
        move_counts = {'paper': 0, 'scissor': 0, 'stone': 0}
        for move in moves:
            move_counts[move] += 1
        
        total = len(moves)
        probs = {move: count/total for move, count in move_counts.items()}
        
        # Repeat probability (tendency to repeat last move)
        repeats = sum(1 for i in range(1, len(moves)) if moves[i] == moves[i-1])
        repeat_prob = repeats / (len(moves) - 1) if len(moves) > 1 else 0
        
        # Switch probability (tendency to never repeat)
        switch_prob = 1.0 - repeat_prob
        
        # Cycle detection (Rock->Paper->Scissors pattern)
        cycle_forward = 0
        cycle_backward = 0
        cycle_map_forward = {'stone': 'paper', 'paper': 'scissor', 'scissor': 'stone'}
        cycle_map_backward = {'paper': 'stone', 'scissor': 'paper', 'stone': 'scissor'}
        
        for i in range(1, len(moves)):
            if cycle_map_forward.get(moves[i-1]) == moves[i]:
                cycle_forward += 1
            if cycle_map_backward.get(moves[i-1]) == moves[i]:
                cycle_backward += 1
        
        max_cycles = len(moves) - 1 if len(moves) > 1 else 1
        cycle_score = max(cycle_forward, cycle_backward) / max_cycles
        
        # Entropy (randomness measure)
        entropy = 0
        for prob in probs.values():
            if prob > 0:
                entropy -= prob * math.log2(prob)
        
        return {
            'repeat_prob': repeat_prob,
            'cycle_score': cycle_score,
            'switch_prob': switch_prob,
            'entropy': entropy,
            'bias_paper': probs['paper'],
            'bias_scissor': probs['scissor'],
            'bias_stone': probs['stone']
        }
    
    def _chi_squared_test(self, features1: Dict[str, float], features2: Dict[str, float]) -> float:
        """Calculate chi-squared statistic between two feature sets"""
        chi2 = 0
        feature_keys = ['repeat_prob', 'cycle_score', 'switch_prob', 'bias_paper', 'bias_scissor', 'bias_stone']
        
        for key in feature_keys:
            expected = features1[key] + 1e-6  # Add small epsilon to avoid division by zero
            observed = features2[key]
            
            # Use a more sensitive test for meaningful differences
            diff = abs(observed - expected)
            if diff > 0.1:  # Only count significant differences
                chi2 += (diff ** 2) / (expected + 0.1)  # Normalized difference
        
        return chi2 * 10  # Scale up to make it more sensitive
    
    def add_move(self, move: str) -> Optional[Dict]:
        """
        Add a new move and check for change points
        
        Returns:
            Dict with change point info if detected, None otherwise
        """
        self.move_history.append(move)
        
        # Need enough history to detect changes
        if len(self.move_history) < self.window_size * 2:
            return None
        
        # Calculate features for current window
        recent_moves = list(self.move_history)[-self.window_size:]
        current_features = self._calculate_features(recent_moves)
        self.feature_history.append(current_features)
        
        # Compare with previous window if we have enough data
        if len(self.feature_history) >= 2:
            prev_features = self.feature_history[-2]
            chi2_stat = self._chi_squared_test(prev_features, current_features)
            
            # Detect significant change
            if chi2_stat > self.chi2_threshold:
                change_point = {
                    'round': len(self.move_history),
                    'chi2_statistic': chi2_stat,
                    'confidence': min(chi2_stat / self.chi2_threshold, 2.0),  # Cap at 2.0
                    'old_features': prev_features,
                    'new_features': current_features,
                    'description': self._describe_change(prev_features, current_features)
                }
                
                self.change_points.append(change_point)
                return change_point
        
        return None
    
    def _describe_change(self, old_features: Dict[str, float], new_features: Dict[str, float]) -> str:
        """Generate human-readable description of strategy change"""
        descriptions = []
        
        # Check for major feature changes
        repeat_change = new_features['repeat_prob'] - old_features['repeat_prob']
        cycle_change = new_features['cycle_score'] - old_features['cycle_score']
        entropy_change = new_features['entropy'] - old_features['entropy']
        
        if repeat_change > 0.3:
            descriptions.append("started repeating moves more")
        elif repeat_change < -0.3:
            descriptions.append("stopped repeating moves")
        
        if cycle_change > 0.3:
            descriptions.append("began cycling through moves")
        elif cycle_change < -0.3:
            descriptions.append("stopped cycling pattern")
        
        if entropy_change > 0.5:
            descriptions.append("became more random")
        elif entropy_change < -0.5:
            descriptions.append("became more predictable")
        
        # Check for bias changes
        old_bias = max(old_features['bias_paper'], old_features['bias_scissor'], old_features['bias_stone'])
        new_bias = max(new_features['bias_paper'], new_features['bias_scissor'], new_features['bias_stone'])
        
        if new_bias > 0.6 and old_bias < 0.5:
            # Find which move is now favored
            if new_features['bias_paper'] > 0.6:
                descriptions.append("started favoring paper")
            elif new_features['bias_scissor'] > 0.6:
                descriptions.append("started favoring scissors")
            elif new_features['bias_stone'] > 0.6:
                descriptions.append("started favoring stone")
        
        if not descriptions:
            descriptions.append("changed strategy pattern")
        
        return "Player " + " and ".join(descriptions)
    
    def get_recent_features(self, window_size: Optional[int] = None) -> Dict[str, float]:
        """Get features for recent moves"""
        if not self.move_history:
            return {}
        
        window = window_size or self.window_size
        recent_moves = list(self.move_history)[-window:]
        return self._calculate_features(recent_moves)
    
    def get_all_change_points(self) -> List[Dict]:
        """Get all detected change points"""
        return self.change_points.copy()
    
    def get_current_strategy_label(self) -> str:
        """Get a label for the current strategy based on recent features"""
        if len(self.move_history) < 5:
            return "warming up"
        
        features = self.get_recent_features()
        
        # Classify current strategy
        if features['repeat_prob'] > 0.7:
            return "repeater"
        elif features['cycle_score'] > 0.6:
            return "cycler"
        elif features['switch_prob'] > 0.8:
            return "anti-repeater"
        elif features['entropy'] > 1.4:
            return "randomizer"
        elif max(features['bias_paper'], features['bias_scissor'], features['bias_stone']) > 0.6:
            # Find favored move
            if features['bias_paper'] > 0.6:
                return "paper-biased"
            elif features['bias_scissor'] > 0.6:
                return "scissor-biased"
            else:
                return "stone-biased"
        else:
            return "balanced"
    
    def reset(self):
        """Reset detector state"""
        self.move_history.clear()
        self.feature_history.clear()
        self.change_points.clear()
        self.last_features = None
    
    def export_analysis(self) -> Dict:
        """Export complete analysis for debugging/visualization"""
        return {
            'total_moves': len(self.move_history),
            'move_history': list(self.move_history),
            'change_points': self.change_points,
            'current_features': self.get_recent_features() if self.move_history else {},
            'current_strategy': self.get_current_strategy_label(),
            'parameters': {
                'window_size': self.window_size,
                'min_segment_length': self.min_segment_length,
                'chi2_threshold': self.chi2_threshold
            }
        }