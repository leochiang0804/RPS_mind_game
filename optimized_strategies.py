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
            return {'paper': 0.33, 'rock': 0.33, 'scissors': 0.33}
        
        # Count recent moves (last 10 for better adaptation)
        recent_history = history[-10:] if len(history) > 10 else history
        move_counts = Counter(recent_history)
        total_moves = len(recent_history)
        
        probabilities = {
            'paper': move_counts.get('paper', 0) / total_moves,
            'rock': move_counts.get('rock', 0) / total_moves,
            'scissors': move_counts.get('scissors', 0) / total_moves
        }
        
        return probabilities
    
    def get_win_probabilities(self, human_probs: Dict[str, float]) -> Dict[str, float]:
        """Calculate win probabilities for each robot move"""
        # Robot move -> probability of winning
        win_probs = {
            'paper': human_probs['rock'],       # Paper beats Rock
            'rock': human_probs['scissors'],    # Rock beats Scissors  
            'scissors': human_probs['paper']    # Scissors beats Paper
        }
        return win_probs
    
    def get_not_lose_probabilities(self, human_probs: Dict[str, float]) -> Dict[str, float]:
        """Calculate not-lose probabilities for each robot move (win + tie)"""
        # Robot move -> probability of not losing (winning + tying)
        not_lose_probs = {
            'paper': human_probs['rock'] + human_probs['paper'],       # Win vs Rock + Tie vs Paper
            'rock': human_probs['scissors'] + human_probs['rock'],     # Win vs Scissors + Tie vs Rock
            'scissors': human_probs['paper'] + human_probs['scissors'] # Win vs Paper + Tie vs Scissors
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
            return random.choice(['paper', 'rock', 'scissors'])
        
        # Get human move probabilities
        human_probs = self.get_move_probabilities(history)
        
        # Calculate win probabilities for each robot move
        win_probs = self.get_win_probabilities(human_probs)
        
        # Find the move with highest win probability
        best_move = max(win_probs.keys(), key=lambda move: win_probs[move])
        highest_prob = max(win_probs.values())
        
        # Calculate confidence score as absolute(2*highest_prob-1)
        confidence = abs(2 * highest_prob - 1)
        
        # Apply aggressive factor if confidence is high
        if highest_prob > self.confidence_threshold:
            # Already calculated confidence above - no need to modify
            pass
        else:
            # If no clear advantage, add some randomness to avoid predictability
            if random.random() < 0.3:
                best_move = random.choice(list(win_probs.keys()))
        
        # Store prediction for analysis
        self.prediction_history.append({
            'strategy': 'to_win',
            'human_probs': human_probs,
            'win_probs': win_probs,
            'chosen_move': best_move,
            'confidence': min(confidence, 1.0),
            'highest_prob': highest_prob
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
            return random.choice(['paper', 'rock', 'scissors'])
        
        # Get human move probabilities
        human_probs = self.get_move_probabilities(history)
        
        # For "not to lose" strategy, we need to calculate the probability of not losing
        # For each robot move, calculate the probability of not losing (win + tie)
        # Robot loses when: Robot=Paper & Human=Scissors, Robot=Rock & Human=Paper, Robot=Scissors & Human=Rock
        
        lose_probs = {
            'paper': human_probs['scissors'],   # Robot paper loses to human scissors
            'rock': human_probs['paper'],       # Robot rock loses to human paper  
            'scissors': human_probs['rock']     # Robot scissors loses to human rock
        }
        
        # Not-to-lose probability = 1 - lose_probability
        not_lose_probs = {move: 1 - lose_prob for move, lose_prob in lose_probs.items()}
        
        # Find the move with highest not-lose probability (lowest losing probability)
        best_move = max(not_lose_probs.keys(), key=lambda move: not_lose_probs[move])
        
        # Calculate confidence score as absolute(2*(sum of highest two probs)-1)
        # Get the two highest human move probabilities
        sorted_probs = sorted(human_probs.values(), reverse=True)
        highest_two_sum = sorted_probs[0] + sorted_probs[1]
        confidence = abs(2 * highest_two_sum - 1)
        
        # Store prediction for analysis
        self.prediction_history.append({
            'strategy': 'not_to_lose',
            'human_probs': human_probs,
            'lose_probs': lose_probs,
            'not_lose_probs': not_lose_probs,
            'chosen_move': best_move,
            'confidence': min(confidence, 1.0),
            'highest_two_sum': highest_two_sum
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
    
    print("🎯 Testing Updated Optimized Robot Strategies\n")
    
    # Test scenario from user requirements: R:15%, P:65%, S:20%
    print("Scenario from requirements: Human tendencies - Rock:15%, Paper:65%, Scissors:20%")
    # Create history with these proportions
    history_test = ['paper'] * 65 + ['scissors'] * 20 + ['rock'] * 15  # Using internal names
    random.shuffle(history_test)
    
    to_win = ToWinStrategy()
    not_lose = NotToLoseStrategy()
    
    win_move = to_win.predict(history_test)
    lose_move = not_lose.predict(history_test)
    
    print(f"To Win Strategy chooses: {win_move} (confidence: {to_win.get_confidence():.3f})")
    print(f"Not to Lose Strategy chooses: {lose_move} (confidence: {not_lose.get_confidence():.3f})")
    
    # Verify the "not to lose" logic
    if not_lose.prediction_history:
        last_pred = not_lose.prediction_history[-1]
        print("\nDetailed analysis for 'Not to Lose' strategy:")
        print("Human probabilities:", {k: f"{v:.2f}" for k, v in last_pred['human_probs'].items()})
        print("Lose probabilities:", {k: f"{v:.2f}" for k, v in last_pred['lose_probs'].items()})
        print("Not-lose probabilities:", {k: f"{v:.2f}" for k, v in last_pred['not_lose_probs'].items()})
        print(f"Sum of highest two human probs: {last_pred['highest_two_sum']:.2f}")
        print(f"Confidence calculation: abs(2 * {last_pred['highest_two_sum']:.2f} - 1) = {last_pred['confidence']:.3f}")
        
        # Verify the robot should choose scissors according to requirements
        expected_not_lose = {
            'paper': 0.15 + 0.65,  # 80% - beats rock + ties with paper
            'rock': 0.20 + 0.15,   # 35% - beats scissors + ties with rock  
            'scissors': 0.65 + 0.20  # 85% - beats paper + ties with scissors
        }
        print(f"Expected not-lose rates: {expected_not_lose}")
        print(f"Robot should choose scissors (85% not-lose rate): {'✓' if lose_move == 'scissors' else '✗'}")
    
    print()
    
    # Original scenario 1: Your first example
    print("Scenario 1: Human tendencies - Scissors(34%), Rock(33%), Paper(33%)")
    history1 = ['scissors'] * 34 + ['rock'] * 33 + ['paper'] * 33
    random.shuffle(history1)
    
    to_win1 = ToWinStrategy()
    not_lose1 = NotToLoseStrategy()
    
    win_move1 = to_win1.predict(history1)
    lose_move1 = not_lose1.predict(history1)
    
    print(f"To Win Strategy chooses: {win_move1} (confidence: {to_win1.get_confidence():.3f})")
    print(f"Not to Lose Strategy chooses: {lose_move1} (confidence: {not_lose1.get_confidence():.3f})")
    print()
    
    # Scenario 2: Your second example  
    print("Scenario 2: Human tendencies - Scissors(50%), Rock(26%), Paper(24%)")
    history2 = ['scissors'] * 50 + ['rock'] * 26 + ['paper'] * 24
    random.shuffle(history2)
    
    to_win2 = ToWinStrategy()
    not_lose2 = NotToLoseStrategy()
    
    win_move2 = to_win2.predict(history2)
    lose_move2 = not_lose2.predict(history2)
    
    print(f"To Win Strategy chooses: {win_move2} (confidence: {to_win2.get_confidence():.3f})")
    print(f"Not to Lose Strategy chooses: {lose_move2} (confidence: {not_lose2.get_confidence():.3f})")
    print()
    
    # Show detailed analysis for To Win strategy
    print("📊 Detailed Analysis for To Win Strategy:")
    if to_win2.prediction_history:
        last_pred = to_win2.prediction_history[-1]
        print("Human probabilities:", {k: f"{v:.2f}" for k, v in last_pred['human_probs'].items()})
        print("Win probabilities:", {k: f"{v:.2f}" for k, v in last_pred['win_probs'].items()})
        print(f"Highest win prob: {last_pred['highest_prob']:.2f}")
        print(f"Confidence calculation: abs(2 * {last_pred['highest_prob']:.2f} - 1) = {last_pred['confidence']:.3f}")