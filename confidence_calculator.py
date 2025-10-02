"""
Confidence Calculator for Rock-Paper-Scissors Models
Calculates normalized margin confidence based on probability distributions and strategy types.
"""

from typing import Dict, Tuple

def calculate_margin_confidence(probabilities: Dict[str, float], strategy_type: str = "to_win") -> float:
    """
    Calculate normalized margin confidence based on probability distribution and strategy.
    
    Args:
        probabilities: Dict with 'paper', 'stone', 'scissor' probabilities (should sum to 1.0)
        strategy_type: Either "to_win" or "not_to_lose"
    
    Returns:
        Confidence score as float between 0.0 and 1.0
    
    Example:
        For probabilities {'paper': 0.6, 'stone': 0.25, 'scissor': 0.15}:
        - "to_win" strategy: Robot chooses scissor (60% win), confidence = |0.6 - 0.4| = 0.2
        - "not_to_lose" strategy: Robot chooses paper (15% lose), confidence = |0.15 - 0.85| = 0.7
    """
    
    # Normalize probabilities to ensure they sum to 1.0
    total = sum(probabilities.values())
    if total == 0:
        return 0.0
    
    normalized_probs = {move: prob/total for move, prob in probabilities.items()}
    
    # Ensure we have all three moves
    paper_prob = normalized_probs.get('paper', 0.0)
    stone_prob = normalized_probs.get('stone', 0.0) 
    scissor_prob = normalized_probs.get('scissor', 0.0)
    
    if strategy_type == "to_win":
        # Calculate win probabilities for each robot move
        robot_win_probs = {
            'scissor': paper_prob,    # Scissor beats Paper
            'stone': scissor_prob,    # Stone beats Scissor  
            'paper': stone_prob       # Paper beats Stone
        }
        
        # Find best move (highest win probability)
        best_win_prob = max(robot_win_probs.values())
        
        # Confidence = normalized margin = |win_prob - (1 - win_prob)|
        confidence = abs(best_win_prob - (1.0 - best_win_prob))
        
    elif strategy_type == "not_to_lose":
        # Calculate lose probabilities for each robot move
        robot_lose_probs = {
            'scissor': stone_prob,    # Scissor loses to Stone
            'stone': paper_prob,      # Stone loses to Paper
            'paper': scissor_prob     # Paper loses to Scissor
        }
        
        # Find best move (lowest lose probability)
        best_lose_prob = min(robot_lose_probs.values())
        
        # Confidence = normalized margin = |lose_prob - (1 - lose_prob)|
        confidence = abs(best_lose_prob - (1.0 - best_lose_prob))
        
    else:
        raise ValueError(f"Unknown strategy_type: {strategy_type}. Must be 'to_win' or 'not_to_lose'")
    
    return min(confidence, 1.0)  # Cap at 1.0


def get_best_robot_move(probabilities: Dict[str, float], strategy_type: str = "to_win") -> Tuple[str, float]:
    """
    Get the best robot move and its confidence based on human move probabilities.
    
    Args:
        probabilities: Dict with 'paper', 'stone', 'scissor' probabilities
        strategy_type: Either "to_win" or "not_to_lose"
    
    Returns:
        Tuple of (best_robot_move, confidence_score)
    """
    
    # Normalize probabilities
    total = sum(probabilities.values())
    if total == 0:
        return 'paper', 0.0
    
    normalized_probs = {move: prob/total for move, prob in probabilities.items()}
    
    paper_prob = normalized_probs.get('paper', 0.0)
    stone_prob = normalized_probs.get('stone', 0.0)
    scissor_prob = normalized_probs.get('scissor', 0.0)
    
    if strategy_type == "to_win":
        # Calculate win probabilities for each robot move
        robot_moves = {
            'scissor': paper_prob,    # Scissor beats Paper
            'stone': scissor_prob,    # Stone beats Scissor
            'paper': stone_prob       # Paper beats Stone
        }
        best_move = max(robot_moves.keys(), key=lambda move: robot_moves[move])
        
    elif strategy_type == "not_to_lose":
        # Calculate lose probabilities for each robot move
        robot_lose_probs = {
            'scissor': stone_prob,    # Scissor loses to Stone
            'stone': paper_prob,      # Stone loses to Paper
            'paper': scissor_prob     # Paper loses to Scissor
        }
        best_move = min(robot_lose_probs.keys(), key=lambda move: robot_lose_probs[move])
        
    else:
        raise ValueError(f"Unknown strategy_type: {strategy_type}")
    
    confidence = calculate_margin_confidence(probabilities, strategy_type)
    return best_move, confidence


# Test function
if __name__ == "__main__":
    print("🧮 Testing Confidence Calculator\n")
    
    # Test example from user's description
    test_probs = {'paper': 0.6, 'stone': 0.25, 'scissor': 0.15}
    
    print("Test probabilities: Paper=60%, Stone=25%, Scissor=15%")
    print()
    
    # Test "to_win" strategy
    move_win, conf_win = get_best_robot_move(test_probs, "to_win")
    print(f"To Win Strategy:")
    print(f"  Best move: {move_win}")
    print(f"  Confidence: {conf_win:.1%}")
    print(f"  Logic: Robot chooses {move_win} to beat paper (60% win prob)")
    print(f"  Margin: |0.6 - 0.4| = {abs(0.6 - 0.4):.1f} = {abs(0.6 - 0.4):.1%}")
    print()
    
    # Test "not_to_lose" strategy  
    move_lose, conf_lose = get_best_robot_move(test_probs, "not_to_lose")
    print(f"Not To Lose Strategy:")
    print(f"  Best move: {move_lose}")
    print(f"  Confidence: {conf_lose:.1%}")
    print(f"  Logic: Robot chooses {move_lose} (only 15% chance to lose)")
    print(f"  Margin: |0.15 - 0.85| = {abs(0.15 - 0.85):.1f} = {abs(0.15 - 0.85):.1%}")
    print()
    
    # Test edge case: uniform distribution
    uniform_probs = {'paper': 0.33, 'stone': 0.33, 'scissor': 0.34}
    move_uniform, conf_uniform = get_best_robot_move(uniform_probs, "to_win")
    print(f"Uniform Distribution Test:")
    print(f"  Best move: {move_uniform}")
    print(f"  Confidence: {conf_uniform:.1%}")
    print(f"  Logic: Nearly uniform, so low confidence expected")