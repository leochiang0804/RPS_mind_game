"""
Advanced Robot Strategies: To Win vs Not to Lose
Implements different strategic approaches based on human move prediction probabilities
"""

import random
from typing import List, Dict, Tuple
from collections import Counter

class OptimizedStrategy:
    """Base class for optimized strategies"""
    
    def __init__(self):
        self.name = "Base Optimized Strategy"
        self.prediction_history = []
        self.confidence_threshold = 0.4  # Minimum confidence to act on prediction
    
    def get_move_probabilities(self, history: List[str]) -> Dict[str, float]:
        """Calculate probabilities for each move based on history"""
        if not history:
            return {'paper': 0.33, 'stone': 0.33, 'scissor': 0.33}
        
        # Count recent moves (last 10 for better adaptation)
        recent_history = history[-10:] if len(history) > 10 else history
        move_counts = Counter(recent_history)
        total_moves = len(recent_history)
        
        probabilities = {
            'paper': move_counts.get('paper', 0) / total_moves,
            'stone': move_counts.get('stone', 0) / total_moves,
            'scissor': move_counts.get('scissor', 0) / total_moves
        }
        
        return probabilities
    
    def get_win_probabilities(self, human_probs: Dict[str, float]) -> Dict[str, float]:
        """Calculate win probabilities for each robot move"""
        # Robot move -> probability of winning
        win_probs = {
            'paper': human_probs['stone'],      # Paper beats Stone
            'stone': human_probs['scissor'],    # Stone beats Scissor  
            'scissor': human_probs['paper']     # Scissor beats Paper
        }
        return win_probs
    
    def get_not_lose_probabilities(self, human_probs: Dict[str, float]) -> Dict[str, float]:
        """Calculate not-lose probabilities for each robot move (win + tie)"""
        # Robot move -> probability of not losing (winning + tying)
        not_lose_probs = {
            'paper': human_probs['stone'] + human_probs['paper'],      # Win vs Stone + Tie vs Paper
            'stone': human_probs['scissor'] + human_probs['stone'],    # Win vs Scissor + Tie vs Stone
            'scissor': human_probs['paper'] + human_probs['scissor']   # Win vs Paper + Tie vs Scissor
        }
        return not_lose_probs

class ToWinStrategy(OptimizedStrategy):
    """Strategy focused on maximizing wins"""
    
    def __init__(self):
        super().__init__()
        self.name = "To Win Strategy"
        self.aggressive_factor = 1.2  # Boost for high-confidence predictions
    
    def predict(self, history: List[str]) -> str:
        """Predict robot move to maximize winning probability"""
        if len(history) < 3:
            return random.choice(['paper', 'stone', 'scissor'])
        
        # Get human move probabilities
        human_probs = self.get_move_probabilities(history)
        
        # Calculate win probabilities for each robot move
        win_probs = self.get_win_probabilities(human_probs)
        
        # Find the move with highest win probability
        best_move = max(win_probs.keys(), key=lambda move: win_probs[move])
        best_prob = win_probs[best_move]
        
        # Apply aggressive factor if confidence is high
        if best_prob > self.confidence_threshold:
            confidence = best_prob * self.aggressive_factor
        else:
            # If no clear advantage, add some randomness to avoid predictability
            if random.random() < 0.3:
                best_move = random.choice(list(win_probs.keys()))
            confidence = best_prob
        
        # Store prediction for analysis
        self.prediction_history.append({
            'strategy': 'to_win',
            'human_probs': human_probs,
            'win_probs': win_probs,
            'chosen_move': best_move,
            'confidence': min(confidence, 1.0)
        })
        
        return best_move
    
    def get_confidence(self) -> float:
        """Get confidence of last prediction"""
        if not self.prediction_history:
            return 0.33
        return self.prediction_history[-1]['confidence']
    
    def get_stats(self) -> Dict:
        """Get strategy statistics"""
        if not self.prediction_history:
            return {'predictions': 0, 'avg_confidence': 0.33}
        
        avg_confidence = sum(p['confidence'] for p in self.prediction_history) / len(self.prediction_history)
        return {
            'predictions': len(self.prediction_history),
            'avg_confidence': avg_confidence,
            'strategy_type': 'aggressive_winning'
        }

class NotToLoseStrategy(OptimizedStrategy):
    """Strategy focused on minimizing losses (maximize win + tie probability)"""
    
    def __init__(self):
        super().__init__()
        self.name = "Not to Lose Strategy"
        self.defensive_factor = 0.8  # More conservative approach
        self.tie_value = 0.5  # Value assigned to ties (between 0 and 1)
    
    def predict(self, history: List[str]) -> str:
        """Predict robot move to maximize not-losing probability"""
        if len(history) < 3:
            return random.choice(['paper', 'stone', 'scissor'])
        
        # Get human move probabilities
        human_probs = self.get_move_probabilities(history)
        
        # Calculate not-lose probabilities (win + tie) for each robot move
        not_lose_probs = self.get_not_lose_probabilities(human_probs)
        
        # Find the move with highest not-lose probability
        best_move = max(not_lose_probs.keys(), key=lambda move: not_lose_probs[move])
        best_prob = not_lose_probs[best_move]
        
        # Apply defensive factor for more conservative play
        confidence = best_prob * self.defensive_factor
        
        # If probabilities are very close, prefer the move that maximizes pure wins
        win_probs = self.get_win_probabilities(human_probs)
        prob_diff = max(not_lose_probs.values()) - min(not_lose_probs.values())
        
        if prob_diff < 0.1:  # If not-lose probabilities are very close
            # Fall back to win-maximizing strategy
            best_move = max(win_probs.keys(), key=lambda move: win_probs[move])
            confidence = win_probs[best_move] * 0.9  # Slightly less confident
        
        # Store prediction for analysis
        self.prediction_history.append({
            'strategy': 'not_to_lose',
            'human_probs': human_probs,
            'not_lose_probs': not_lose_probs,
            'win_probs': win_probs,
            'chosen_move': best_move,
            'confidence': min(confidence, 1.0),
            'prob_diff': prob_diff
        })
        
        return best_move
    
    def get_confidence(self) -> float:
        """Get confidence of last prediction"""
        if not self.prediction_history:
            return 0.33
        return self.prediction_history[-1]['confidence']
    
    def get_stats(self) -> Dict:
        """Get strategy statistics"""
        if not self.prediction_history:
            return {'predictions': 0, 'avg_confidence': 0.33}
        
        avg_confidence = sum(p['confidence'] for p in self.prediction_history) / len(self.prediction_history)
        avg_prob_diff = sum(p['prob_diff'] for p in self.prediction_history) / len(self.prediction_history)
        
        return {
            'predictions': len(self.prediction_history),
            'avg_confidence': avg_confidence,
            'avg_prob_diff': avg_prob_diff,
            'strategy_type': 'defensive_not_losing'
        }

# Example usage and testing
if __name__ == "__main__":
    # Test the strategies with example scenarios
    
    print("🎯 Testing Optimized Robot Strategies\n")
    
    # Scenario 1: Your first example
    print("Scenario 1: Human tendencies - Scissor(34%), Stone(33%), Paper(33%)")
    history1 = ['scissor'] * 34 + ['stone'] * 33 + ['paper'] * 33
    random.shuffle(history1)
    
    to_win = ToWinStrategy()
    not_lose = NotToLoseStrategy()
    
    win_move = to_win.predict(history1)
    lose_move = not_lose.predict(history1)
    
    print(f"To Win Strategy chooses: {win_move} (confidence: {to_win.get_confidence():.2f})")
    print(f"Not to Lose Strategy chooses: {lose_move} (confidence: {not_lose.get_confidence():.2f})")
    print()
    
    # Scenario 2: Your second example  
    print("Scenario 2: Human tendencies - Scissor(50%), Stone(26%), Paper(24%)")
    history2 = ['scissor'] * 50 + ['stone'] * 26 + ['paper'] * 24
    random.shuffle(history2)
    
    to_win2 = ToWinStrategy()
    not_lose2 = NotToLoseStrategy()
    
    win_move2 = to_win2.predict(history2)
    lose_move2 = not_lose2.predict(history2)
    
    print(f"To Win Strategy chooses: {win_move2} (confidence: {to_win2.get_confidence():.2f})")
    print(f"Not to Lose Strategy chooses: {lose_move2} (confidence: {not_lose2.get_confidence():.2f})")
    print()
    
    # Show detailed analysis
    print("📊 Detailed Analysis:")
    if to_win.prediction_history:
        last_pred = to_win.prediction_history[-1]
        print("To Win - Human probabilities:", last_pred['human_probs'])
        print("To Win - Win probabilities:", last_pred['win_probs'])
    
    if not_lose2.prediction_history:
        last_pred = not_lose2.prediction_history[-1]
        print("Not to Lose - Not-lose probabilities:", last_pred['not_lose_probs'])
        print("Not to Lose - Probability difference:", last_pred['prob_diff'])