"""
Human-Like Bias Module (HLBM) for RPS AI
========================================

Implements psychological biases to transform raw statistical predictions
into human-like behavior patterns.

Features 5 bias types:
- RA: Repetition Aversion
- WSLS: Win-Stay / Lose-Shift
- ALT: Alternation bias
- CYC: Cycle bias
- META: Meta-expectation (double-guessing on long streaks)

Based on hlbm_spec.md specification.

Author: AI Assistant
Created: 2025-10-03
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Sequence
from collections import Counter
from move_mapping import (
    MOVES, MOVE_TO_NUMBER, NUMBER_TO_MOVE,
    normalize_move, get_counter_move, number_to_move
)


class HumanLikeBiasModule:
    """
    Human-Like Bias Module that adds psychological realism to statistical predictions.
    
    Transforms p_base (Markov predictor output) into p_adj (psychologically adjusted probabilities).
    """
    
    def __init__(self, 
                 lambda_influence: float = 0.25,
                 weights: Optional[Dict[str, float]] = None,
                 bias_params: Optional[Dict[str, Dict[str, float]]] = None):
        """
        Initialize HLBM.
        
        Args:
            lambda_influence: HLBM influence factor λ ∈ [0.10, 0.45]
            weights: Bias weights {RA, WSLS, ALT, CYC, META} that sum to 1
            bias_params: Parameters for each bias type
        """
        if not (0.10 <= lambda_influence <= 0.45):
            raise ValueError(f"lambda_influence must be in [0.10, 0.45], got {lambda_influence}")
            
        self.lambda_influence = lambda_influence
        
        # Default weights (Challenger difficulty)
        if weights is None:
            weights = {
                'RA': 0.28,
                'WSLS': 0.30,
                'ALT': 0.18,
                'CYC': 0.16,
                'META': 0.08
            }
            
        # Validate weights
        if not np.isclose(sum(weights.values()), 1.0, atol=1e-6):
            raise ValueError(f"Weights must sum to 1.0, got {sum(weights.values())}")
            
        self.weights = weights
        
        # Default bias parameters (Challenger difficulty)
        if bias_params is None:
            bias_params = {
                'RA': {'rho': 0.06},
                'WSLS': {'delta_WS': 0.04, 'delta_LS': 0.05, 'delta_T': 0.02},
                'ALT': {'delta_ALT': 0.04},
                'CYC': {'delta_CYC': 0.04},
                'META': {'delta_META': 0.015}
            }
            
        self.bias_params = bias_params
        
        # Minimum probability to avoid zero probabilities
        self.epsilon_min = 0.02
        
    def apply_biases(self, 
                    p_base: np.ndarray,
                    move_history: Sequence[Union[str, int]],
                    outcome_history: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, Dict]:
        """
        Apply all biases to base probabilities.
        
        Args:
            p_base: Base probabilities [P(rock), P(paper), P(scissors)]
            move_history: Recent move history (human moves)
            outcome_history: Recent outcomes ['win', 'loss', 'tie'] from human perspective
            
        Returns:
            Tuple of (adjusted_probabilities, metadata)
        """
        if len(p_base) != 3:
            raise ValueError(f"p_base must have length 3, got {len(p_base)}")
            
        if not np.isclose(np.sum(p_base), 1.0, atol=1e-6):
            raise ValueError(f"p_base must sum to 1.0, got {np.sum(p_base)}")
            
        # Normalize move history
        normalized_history = []
        for move in move_history:
            if isinstance(move, int):
                move = number_to_move(move)
            normalized_history.append(normalize_move(move))
            
        # Calculate individual bias contributions
        bias_contributions = {}
        total_bias = np.zeros(3)
        
        # RA: Repetition Aversion
        b_ra, ra_meta = self._calculate_ra_bias(normalized_history)
        bias_contributions['RA'] = (b_ra, ra_meta)
        total_bias += self.weights['RA'] * b_ra
        
        # WSLS: Win-Stay / Lose-Shift
        b_wsls, wsls_meta = self._calculate_wsls_bias(normalized_history, outcome_history)
        bias_contributions['WSLS'] = (b_wsls, wsls_meta)
        total_bias += self.weights['WSLS'] * b_wsls
        
        # ALT: Alternation bias
        b_alt, alt_meta = self._calculate_alt_bias(normalized_history)
        bias_contributions['ALT'] = (b_alt, alt_meta)
        total_bias += self.weights['ALT'] * b_alt
        
        # CYC: Cycle bias
        b_cyc, cyc_meta = self._calculate_cyc_bias(normalized_history)
        bias_contributions['CYC'] = (b_cyc, cyc_meta)
        total_bias += self.weights['CYC'] * b_cyc
        
        # META: Meta-expectation
        b_meta, meta_meta = self._calculate_meta_bias(normalized_history)
        bias_contributions['META'] = (b_meta, meta_meta)
        total_bias += self.weights['META'] * b_meta
        
        # Apply lambda scaling and combine with base probabilities
        adjustment = self.lambda_influence * total_bias
        p_adjusted = p_base + adjustment
        
        # Clamp to valid range and normalize
        p_adjusted = np.clip(p_adjusted, self.epsilon_min, 1.0 - self.epsilon_min)
        p_adjusted = p_adjusted / np.sum(p_adjusted)
        
        # Metadata
        metadata = {
            'method': 'HLBM',
            'lambda_influence': self.lambda_influence,
            'weights': self.weights,
            'bias_contributions': {name: contrib[0].tolist() for name, contrib in bias_contributions.items()},
            'bias_metadata': {name: contrib[1] for name, contrib in bias_contributions.items()},
            'total_bias': total_bias.tolist(),
            'adjustment': adjustment.tolist(),
            'p_base': p_base.tolist(),
            'p_adjusted': p_adjusted.tolist()
        }
        
        return p_adjusted, metadata
        
    def _calculate_ra_bias(self, move_history: List[str]) -> Tuple[np.ndarray, Dict]:
        """Calculate Repetition Aversion bias."""
        bias = np.zeros(3)
        metadata = {'type': 'RA', 'active': False}
        
        if len(move_history) < 2:
            return bias, metadata
            
        # Find current streak length
        last_move = move_history[-1]
        streak_length = 1
        
        for i in range(len(move_history) - 2, -1, -1):
            if move_history[i] == last_move:
                streak_length += 1
            else:
                break
                
        # Apply RA based on streak length
        rho = self.bias_params['RA']['rho']
        last_idx = MOVE_TO_NUMBER[last_move]
        
        if streak_length == 1:
            # No effect
            pass
        elif streak_length == 2:
            # Subtract rho/2 from last move, add rho/4 to others
            bias[last_idx] = -rho / 2
            for i in range(3):
                if i != last_idx:
                    bias[i] = rho / 4
        else:  # streak_length >= 3
            # Subtract rho from last move, add rho/2 to others
            bias[last_idx] = -rho
            for i in range(3):
                if i != last_idx:
                    bias[i] = rho / 2
                    
        metadata.update({
            'active': streak_length >= 2,
            'streak_length': streak_length,
            'last_move': last_move,
            'rho': rho
        })
        
        return bias, metadata
        
    def _calculate_wsls_bias(self, move_history: List[str], outcome_history: Optional[Sequence[str]]) -> Tuple[np.ndarray, Dict]:
        """Calculate Win-Stay / Lose-Shift bias."""
        bias = np.zeros(3)
        metadata = {'type': 'WSLS', 'active': False}
        
        if not outcome_history or len(outcome_history) == 0 or len(move_history) == 0:
            return bias, metadata
            
        last_outcome = outcome_history[-1].lower()
        last_move = move_history[-1]
        last_idx = MOVE_TO_NUMBER[last_move]
        
        params = self.bias_params['WSLS']
        delta_applied = 0
        
        if last_outcome == 'win':
            # Win-Stay: add to last move, subtract from others
            delta = params['delta_WS']
            delta_applied = delta
            bias[last_idx] = delta
            for i in range(3):
                if i != last_idx:
                    bias[i] = -delta / 2
                    
        elif last_outcome == 'loss':
            # Lose-Shift: subtract from last move, add to others
            delta = params['delta_LS']
            delta_applied = delta
            bias[last_idx] = -delta
            for i in range(3):
                if i != last_idx:
                    bias[i] = delta / 2
                    
        elif last_outcome == 'tie':
            # After tie: add to counter of last move, subtract from others
            delta = params['delta_T']
            delta_applied = delta
            counter_move = get_counter_move(last_move)
            counter_idx = MOVE_TO_NUMBER[counter_move]
            bias[counter_idx] = delta
            for i in range(3):
                if i != counter_idx:
                    bias[i] = -delta / 2
                    
        metadata.update({
            'active': True,
            'last_outcome': last_outcome,
            'last_move': last_move,
            'delta_applied': delta_applied
        })
        
        return bias, metadata
        
    def _calculate_alt_bias(self, move_history: List[str]) -> Tuple[np.ndarray, Dict]:
        """Calculate Alternation bias."""
        bias = np.zeros(3)
        metadata = {'type': 'ALT', 'active': False, 'alternation_score': 0.0}
        
        if len(move_history) < 4:
            return bias, metadata
            
        # Calculate alternation score
        alt_score = self._compute_alternation_score(move_history)
        
        if alt_score > 0.1:  # Only apply if there's significant alternation
            # Predict alternation-consistent move
            alt_move = self._predict_alternation_move(move_history)
            if alt_move:
                delta = self.bias_params['ALT']['delta_ALT'] * alt_score
                alt_idx = MOVE_TO_NUMBER[alt_move]
                
                bias[alt_idx] = delta
                for i in range(3):
                    if i != alt_idx:
                        bias[i] = -delta / 2
                        
                metadata.update({
                    'active': True,
                    'alternation_score': alt_score,
                    'predicted_move': alt_move
                })
                
        return bias, metadata
        
    def _calculate_cyc_bias(self, move_history: List[str]) -> Tuple[np.ndarray, Dict]:
        """Calculate Cycle bias."""
        bias = np.zeros(3)
        metadata = {'type': 'CYC', 'active': False, 'cycle_score': 0.0}
        
        if len(move_history) < 3:
            return bias, metadata
            
        # Calculate cycle score
        cyc_score = self._compute_cycle_score(move_history)
        
        if cyc_score > 0.1:  # Only apply if there's significant cycling
            # Predict cycle-consistent move
            cyc_move = self._predict_cycle_move(move_history)
            if cyc_move:
                delta = self.bias_params['CYC']['delta_CYC'] * cyc_score
                cyc_idx = MOVE_TO_NUMBER[cyc_move]
                
                bias[cyc_idx] = delta
                for i in range(3):
                    if i != cyc_idx:
                        bias[i] = -delta / 2
                        
                metadata.update({
                    'active': True,
                    'cycle_score': cyc_score,
                    'predicted_move': cyc_move
                })
                
        return bias, metadata
        
    def _calculate_meta_bias(self, move_history: List[str]) -> Tuple[np.ndarray, Dict]:
        """Calculate Meta-expectation bias."""
        bias = np.zeros(3)
        metadata = {'type': 'META', 'active': False}
        
        if len(move_history) < 4:
            return bias, metadata
            
        # Find current streak length
        last_move = move_history[-1]
        streak_length = 1
        
        for i in range(len(move_history) - 2, -1, -1):
            if move_history[i] == last_move:
                streak_length += 1
            else:
                break
                
        # Apply META bias for very long streaks (L >= 4)
        if streak_length >= 4:
            delta = self.bias_params['META']['delta_META']
            last_idx = MOVE_TO_NUMBER[last_move]
            
            # Add to last move (double-guessing: maybe they'll stay)
            bias[last_idx] = delta
            for i in range(3):
                if i != last_idx:
                    bias[i] = -delta / 2
                    
            metadata.update({
                'active': True,
                'streak_length': streak_length,
                'last_move': last_move
            })
            
        return bias, metadata
        
    def _compute_alternation_score(self, move_history: List[str]) -> float:
        """Compute alternation consistency score."""
        if len(move_history) < 4:
            return 0.0
            
        # Look at last 6 moves for alternation pattern
        recent = move_history[-6:]
        
        # Count alternating pairs
        alternations = 0
        total_pairs = 0
        
        for i in range(len(recent) - 2):
            if recent[i] == recent[i + 2]:  # Pattern: A-B-A
                alternations += 1
            total_pairs += 1
            
        return alternations / total_pairs if total_pairs > 0 else 0.0
        
    def _predict_alternation_move(self, move_history: List[str]) -> Optional[str]:
        """Predict next move based on alternation pattern."""
        if len(move_history) < 2:
            return None
            
        # Simple alternation: return move from 2 steps back
        return move_history[-2]
        
    def _compute_cycle_score(self, move_history: List[str]) -> float:
        """Compute cycle consistency score."""
        if len(move_history) < 6:
            return 0.0
            
        # Look for R->P->S->R pattern
        cycle_pattern = ['rock', 'paper', 'scissors']
        recent = move_history[-6:]
        
        cycle_matches = 0
        total_triplets = 0
        
        for i in range(len(recent) - 2):
            triplet = recent[i:i+3]
            total_triplets += 1
            
            # Check if this triplet matches cycle pattern (starting at any position)
            for start_pos in range(3):
                expected = [cycle_pattern[(start_pos + j) % 3] for j in range(3)]
                if triplet == expected:
                    cycle_matches += 1
                    break
                    
        return cycle_matches / total_triplets if total_triplets > 0 else 0.0
        
    def _predict_cycle_move(self, move_history: List[str]) -> Optional[str]:
        """Predict next move based on cycle pattern."""
        if len(move_history) < 1:
            return None
            
        last_move = move_history[-1]
        
        # R->P->S->R cycle
        cycle_map = {
            'rock': 'paper',
            'paper': 'scissors',
            'scissors': 'rock'
        }
        
        return cycle_map.get(last_move)
        
    def adjust_for_difficulty(self, difficulty: str) -> None:
        """Adjust bias parameters based on difficulty level."""
        if difficulty.lower() == 'rookie':
            # Higher bias magnitudes
            self.bias_params['RA']['rho'] = 0.08
            self.bias_params['WSLS']['delta_LS'] = 0.06
            self.bias_params['ALT']['delta_ALT'] = 0.05
            self.weights = {'RA': 0.32, 'WSLS': 0.28, 'ALT': 0.18, 'CYC': 0.12, 'META': 0.10}
            
        elif difficulty.lower() == 'challenger':
            # Medium bias magnitudes (default)
            self.bias_params['RA']['rho'] = 0.06
            self.bias_params['WSLS']['delta_LS'] = 0.05
            self.bias_params['ALT']['delta_ALT'] = 0.04
            self.weights = {'RA': 0.28, 'WSLS': 0.30, 'ALT': 0.18, 'CYC': 0.16, 'META': 0.08}
            
        elif difficulty.lower() == 'master':
            # Lower bias magnitudes
            self.bias_params['RA']['rho'] = 0.04
            self.bias_params['WSLS']['delta_LS'] = 0.04
            self.bias_params['ALT']['delta_ALT'] = 0.03
            self.weights = {'RA': 0.24, 'WSLS': 0.30, 'ALT': 0.16, 'CYC': 0.24, 'META': 0.06}
            
    def adjust_for_personality(self, personality: str) -> None:
        """Adjust lambda influence based on personality."""
        personality = personality.lower()
        
        if personality in ['aggressive', 'confident']:
            # Less psychological, more statistical
            self.lambda_influence *= 0.8
        elif personality in ['defensive', 'cautious', 'chameleon']:
            # More psychological influence
            self.lambda_influence *= 1.2
        elif personality in ['unpredictable', 'wild']:
            # High psychological influence
            self.lambda_influence *= 1.3
            
        # Clamp to valid range
        self.lambda_influence = np.clip(self.lambda_influence, 0.10, 0.45)


def create_hlbm(difficulty: str = 'challenger', 
               personality: str = 'neutral',
               lambda_influence: Optional[float] = None) -> HumanLikeBiasModule:
    """
    Factory function to create HLBM with appropriate settings.
    
    Args:
        difficulty: Difficulty level ('rookie', 'challenger', 'master')
        personality: Personality type
        lambda_influence: Override lambda value
        
    Returns:
        Configured HumanLikeBiasModule instance
    """
    # Default lambda values by difficulty
    if lambda_influence is None:
        lambda_map = {
            'rookie': 0.35,
            'challenger': 0.25,
            'master': 0.15
        }
        lambda_influence = lambda_map.get(difficulty.lower(), 0.25)
        
    hlbm = HumanLikeBiasModule(lambda_influence=lambda_influence)
    hlbm.adjust_for_difficulty(difficulty)
    hlbm.adjust_for_personality(personality)
    
    return hlbm


# Test function
def test_hlbm():
    """Test the HLBM with sample data."""
    print("Testing Human-Like Bias Module...")
    
    # Create HLBM
    hlbm = create_hlbm(difficulty='challenger', personality='neutral')
    
    # Sample data
    p_base = np.array([0.60, 0.25, 0.15])
    move_history = ['rock', 'rock', 'rock', 'paper']  # R streak then P
    outcome_history = ['loss', 'loss', 'loss']  # Human lost last 3
    
    print(f"Base probabilities: {p_base}")
    print(f"Move history: {move_history}")
    print(f"Outcome history: {outcome_history}")
    
    # Apply biases
    p_adjusted, metadata = hlbm.apply_biases(p_base, move_history, outcome_history)
    
    print(f"Adjusted probabilities: {p_adjusted}")
    print(f"Total adjustment: {metadata['adjustment']}")
    
    # Show individual bias contributions
    print("\nIndividual bias contributions:")
    for bias_name, contribution in metadata['bias_contributions'].items():
        print(f"  {bias_name}: {contribution}")
        
    # Test different difficulties
    print("\nTesting different difficulties:")
    for diff in ['rookie', 'challenger', 'master']:
        hlbm_diff = create_hlbm(difficulty=diff)
        p_adj, _ = hlbm_diff.apply_biases(p_base, move_history, outcome_history)
        print(f"  {diff}: {p_adj}")


if __name__ == "__main__":
    test_hlbm()