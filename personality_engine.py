"""
Phase 3.4: Advanced AI Personality Modes
Enhanced personality system with distinct behavioral patterns, visual themes, and adaptive traits
"""

import random
import statistics
from collections import Counter, defaultdict
from enum import Enum
from typing import Dict, List, Tuple, Optional


class PersonalityTrait(Enum):
    """Personality traits that affect AI behavior"""
    AGGRESSION = "aggression"
    DEFENSIVENESS = "defensiveness"
    ADAPTABILITY = "adaptability"
    PREDICTABILITY = "predictability"
    RISK_TOLERANCE = "risk_tolerance"
    MEMORY_SPAN = "memory_span"
    CONFIDENCE_SENSITIVITY = "confidence_sensitivity"


class PersonalityProfile:
    """Defines a personality with specific trait values and behaviors"""
    
    def __init__(self, name: str, description: str, traits: Dict[PersonalityTrait, float], 
                 color_theme: Dict[str, str], behavior_modifiers: Dict[str, float]):
        self.name = name
        self.description = description
        self.traits = traits  # Values from 0.0 to 1.0
        self.color_theme = color_theme
        self.behavior_modifiers = behavior_modifiers
        self.adaptive_state = {}  # State tracking for adaptive behaviors
        
    def get_trait(self, trait: PersonalityTrait) -> float:
        """Get a trait value (0.0 to 1.0)"""
        return self.traits.get(trait, 0.5)


class AdvancedPersonalityEngine:
    """Advanced AI personality engine with sophisticated behavioral patterns"""
    
    def __init__(self):
        self.personalities = self._initialize_personalities()
        self.current_personality = None
        self.game_state = {
            'recent_performance': [],
            'human_patterns': defaultdict(int),
            'confidence_history': [],
            'streak_data': {'wins': 0, 'losses': 0, 'ties': 0},
            'adaptation_counter': 0
        }
    
    def _initialize_personalities(self) -> Dict[str, PersonalityProfile]:
        """Initialize all personality profiles"""
        
        personalities = {
            'berserker': PersonalityProfile(
                name="The Berserker",
                description="Extremely aggressive, high-risk high-reward playstyle. Targets weaknesses ruthlessly.",
                traits={
                    PersonalityTrait.AGGRESSION: 0.95,
                    PersonalityTrait.DEFENSIVENESS: 0.1,
                    PersonalityTrait.ADAPTABILITY: 0.3,
                    PersonalityTrait.PREDICTABILITY: 0.2,
                    PersonalityTrait.RISK_TOLERANCE: 0.9,
                    PersonalityTrait.MEMORY_SPAN: 0.4,
                    PersonalityTrait.CONFIDENCE_SENSITIVITY: 0.8
                },
                color_theme={
                    'primary': '#FF0000',
                    'secondary': '#8B0000',
                    'accent': '#FF4500',
                    'background': '#FFE4E4'
                },
                behavior_modifiers={
                    'aggression_multiplier': 1.8,
                    'pattern_exploit_chance': 0.8,
                    'risk_taking': 0.9,
                    'counter_attack_prob': 0.7
                }
            ),
            
            'guardian': PersonalityProfile(
                name="The Guardian",
                description="Highly defensive, prioritizes not losing over winning. Patient and calculating.",
                traits={
                    PersonalityTrait.AGGRESSION: 0.2,
                    PersonalityTrait.DEFENSIVENESS: 0.9,
                    PersonalityTrait.ADAPTABILITY: 0.6,
                    PersonalityTrait.PREDICTABILITY: 0.7,
                    PersonalityTrait.RISK_TOLERANCE: 0.1,
                    PersonalityTrait.MEMORY_SPAN: 0.8,
                    PersonalityTrait.CONFIDENCE_SENSITIVITY: 0.3
                },
                color_theme={
                    'primary': '#0066CC',
                    'secondary': '#003366',
                    'accent': '#4A90E2',
                    'background': '#E8F4FF'
                },
                behavior_modifiers={
                    'defensive_bonus': 1.5,
                    'tie_preference': 0.8,
                    'loss_avoidance': 0.9,
                    'patience_factor': 1.3
                }
            ),
            
            'chameleon': PersonalityProfile(
                name="The Chameleon",
                description="Highly adaptive, changes strategy based on opponent and performance. Unpredictable.",
                traits={
                    PersonalityTrait.AGGRESSION: 0.5,
                    PersonalityTrait.DEFENSIVENESS: 0.5,
                    PersonalityTrait.ADAPTABILITY: 0.95,
                    PersonalityTrait.PREDICTABILITY: 0.1,
                    PersonalityTrait.RISK_TOLERANCE: 0.6,
                    PersonalityTrait.MEMORY_SPAN: 0.9,
                    PersonalityTrait.CONFIDENCE_SENSITIVITY: 0.7
                },
                color_theme={
                    'primary': '#9932CC',
                    'secondary': '#4B0082',
                    'accent': '#BA55D3',
                    'background': '#F5E6FF'
                },
                behavior_modifiers={
                    'adaptation_rate': 0.9,
                    'strategy_switch_threshold': 0.3,
                    'mimicry_chance': 0.4,
                    'innovation_rate': 0.6
                }
            ),
            
            'professor': PersonalityProfile(
                name="The Professor",
                description="Analytical and methodical. Uses complex patterns and psychological insights.",
                traits={
                    PersonalityTrait.AGGRESSION: 0.4,
                    PersonalityTrait.DEFENSIVENESS: 0.6,
                    PersonalityTrait.ADAPTABILITY: 0.8,
                    PersonalityTrait.PREDICTABILITY: 0.8,
                    PersonalityTrait.RISK_TOLERANCE: 0.3,
                    PersonalityTrait.MEMORY_SPAN: 0.95,
                    PersonalityTrait.CONFIDENCE_SENSITIVITY: 0.9
                },
                color_theme={
                    'primary': '#006400',
                    'secondary': '#004225',
                    'accent': '#228B22',
                    'background': '#E8F5E8'
                },
                behavior_modifiers={
                    'analysis_depth': 1.5,
                    'pattern_recognition': 0.9,
                    'prediction_accuracy': 1.2,
                    'psychological_insight': 0.8
                }
            ),
            
            'wildcard': PersonalityProfile(
                name="The Wildcard",
                description="Completely unpredictable and chaotic. Thrives on confusion and misdirection.",
                traits={
                    PersonalityTrait.AGGRESSION: 0.6,
                    PersonalityTrait.DEFENSIVENESS: 0.3,
                    PersonalityTrait.ADAPTABILITY: 0.7,
                    PersonalityTrait.PREDICTABILITY: 0.05,
                    PersonalityTrait.RISK_TOLERANCE: 0.95,
                    PersonalityTrait.MEMORY_SPAN: 0.2,
                    PersonalityTrait.CONFIDENCE_SENSITIVITY: 0.1
                },
                color_theme={
                    'primary': '#FF6347',
                    'secondary': '#CD5C5C',
                    'accent': '#FFD700',
                    'background': '#FFF8DC'
                },
                behavior_modifiers={
                    'chaos_factor': 0.8,
                    'randomness': 0.7,
                    'misdirection': 0.6,
                    'surprise_moves': 0.9
                }
            ),
            
            'mirror': PersonalityProfile(
                name="The Mirror",
                description="Reflects and learns from opponent's style. Becomes more similar over time.",
                traits={
                    PersonalityTrait.AGGRESSION: 0.5,
                    PersonalityTrait.DEFENSIVENESS: 0.5,
                    PersonalityTrait.ADAPTABILITY: 0.8,
                    PersonalityTrait.PREDICTABILITY: 0.6,
                    PersonalityTrait.RISK_TOLERANCE: 0.5,
                    PersonalityTrait.MEMORY_SPAN: 0.85,
                    PersonalityTrait.CONFIDENCE_SENSITIVITY: 0.6
                },
                color_theme={
                    'primary': '#708090',
                    'secondary': '#2F4F4F',
                    'accent': '#87CEEB',
                    'background': '#F0F8FF'
                },
                behavior_modifiers={
                    'mimicry_strength': 0.8,
                    'learning_rate': 0.9,
                    'reflection_accuracy': 0.7,
                    'style_adoption': 0.6
                }
            )
        }
        
        return personalities
    
    def set_personality(self, personality_name: str):
        """Set the current personality"""
        if personality_name in self.personalities:
            self.current_personality = self.personalities[personality_name]
            self.game_state = {
                'recent_performance': [],
                'human_patterns': defaultdict(int),
                'confidence_history': [],
                'streak_data': {'wins': 0, 'losses': 0, 'ties': 0},
                'adaptation_counter': 0
            }
        else:
            raise ValueError(f"Unknown personality: {personality_name}")
    
    def update_game_state(self, human_move: str, robot_move: str, result: str, confidence: float):
        """Update the personality's internal game state"""
        if not self.current_personality:
            return
        
        # Update performance tracking
        self.game_state['recent_performance'].append(result)
        if len(self.game_state['recent_performance']) > 10:
            self.game_state['recent_performance'].pop(0)
        
        # Update human pattern tracking
        self.game_state['human_patterns'][human_move] += 1
        
        # Update confidence history
        self.game_state['confidence_history'].append(confidence)
        if len(self.game_state['confidence_history']) > 15:
            self.game_state['confidence_history'].pop(0)
        
        # Update streak data
        if result == 'robot':
            self.game_state['streak_data']['wins'] += 1
            self.game_state['streak_data']['losses'] = 0
            self.game_state['streak_data']['ties'] = 0
        elif result == 'human':
            self.game_state['streak_data']['losses'] += 1
            self.game_state['streak_data']['wins'] = 0
            self.game_state['streak_data']['ties'] = 0
        else:  # tie
            self.game_state['streak_data']['ties'] += 1
            self.game_state['streak_data']['wins'] = 0
            self.game_state['streak_data']['losses'] = 0
        
        self.game_state['adaptation_counter'] += 1
    
    def apply_personality_to_move(self, base_move: str, confidence: float, 
                                human_history: List[str], game_history: List[Tuple[str, str]]) -> Tuple[str, float]:
        """Apply personality modifications to the base AI move and return both move and modified confidence"""
        if not self.current_personality:
            return base_move, confidence
        
        personality = self.current_personality
        moves = ['paper', 'rock', 'scissors']
        
        # First modify confidence based on personality traits
        modified_confidence = self.modify_confidence_by_personality(confidence)
        
        # Then apply move modifications
        modified_move = base_move
        
        # Berserker personality
        if personality.name == "The Berserker":
            modified_move = self._apply_berserker_behavior(base_move, modified_confidence, human_history, game_history)
        
        # Guardian personality
        elif personality.name == "The Guardian":
            modified_move = self._apply_guardian_behavior(base_move, modified_confidence, human_history, game_history)
        
        # Chameleon personality
        elif personality.name == "The Chameleon":
            modified_move = self._apply_chameleon_behavior(base_move, modified_confidence, human_history, game_history)
        
        # Professor personality
        elif personality.name == "The Professor":
            modified_move = self._apply_professor_behavior(base_move, modified_confidence, human_history, game_history)
        
        # Wildcard personality
        elif personality.name == "The Wildcard":
            modified_move = self._apply_wildcard_behavior(base_move, modified_confidence, human_history, game_history)
        
        # Mirror personality
        elif personality.name == "The Mirror":
            modified_move = self._apply_mirror_behavior(base_move, modified_confidence, human_history, game_history)
        
        return modified_move, modified_confidence
    
    def modify_confidence_by_personality(self, base_confidence: float) -> float:
        """Modify confidence score based on personality traits"""
        if not self.current_personality:
            return base_confidence
        
        # Get trait values
        confidence_sensitivity = self.current_personality.get_trait(PersonalityTrait.CONFIDENCE_SENSITIVITY)
        risk_tolerance = self.current_personality.get_trait(PersonalityTrait.RISK_TOLERANCE)
        aggression = self.current_personality.get_trait(PersonalityTrait.AGGRESSION)
        defensiveness = self.current_personality.get_trait(PersonalityTrait.DEFENSIVENESS)
        
        modified_confidence = base_confidence
        
        # Apply confidence sensitivity: high sensitivity amplifies confidence differences
        if confidence_sensitivity > 0.5:
            # High sensitivity: amplify confidence (make confident more confident, uncertain more uncertain)
            if base_confidence > 0.5:
                modified_confidence = base_confidence + (confidence_sensitivity - 0.5) * (1.0 - base_confidence)
            else:
                modified_confidence = base_confidence - (confidence_sensitivity - 0.5) * base_confidence
        elif confidence_sensitivity < 0.5:
            # Low sensitivity: moderate confidence (bring towards 0.5)
            sensitivity_factor = 0.5 - confidence_sensitivity
            modified_confidence = base_confidence + sensitivity_factor * (0.5 - base_confidence)
        
        # Apply risk tolerance: high risk tolerance increases confidence in risky situations
        risk_factor = (risk_tolerance - 0.5) * 0.3  # -0.15 to +0.15 adjustment
        modified_confidence += risk_factor
        
        # Apply aggression: aggressive personalities are more confident
        aggression_factor = (aggression - 0.5) * 0.2  # -0.1 to +0.1 adjustment
        modified_confidence += aggression_factor
        
        # Apply defensiveness: defensive personalities are less confident
        defensiveness_factor = (defensiveness - 0.5) * -0.15  # -0.075 to +0.075 adjustment
        modified_confidence += defensiveness_factor
        
        # Ensure confidence stays in valid range [0.0, 1.0]
        modified_confidence = max(0.0, min(1.0, modified_confidence))
        
        return modified_confidence
    
    def _apply_berserker_behavior(self, base_move: str, confidence: float, 
                                human_history: List[str], game_history: List[Tuple[str, str]]) -> str:
        """Apply Berserker personality - extremely aggressive targeting"""
        if len(human_history) < 3 or not self.current_personality:
            return base_move
        
        # Get trait values
        aggression = self.current_personality.get_trait(PersonalityTrait.AGGRESSION)
        memory_span = self.current_personality.get_trait(PersonalityTrait.MEMORY_SPAN)
        risk_tolerance = self.current_personality.get_trait(PersonalityTrait.RISK_TOLERANCE)
        
        # Memory span determines how many recent moves to analyze (2-10 moves)
        memory_window = max(2, int(memory_span * 8) + 2)
        
        # Find most common human move and aggressively counter it
        most_common = Counter(human_history[-memory_window:]).most_common(1)[0][0]
        counter_moves = {'paper': 'scissors', 'rock': 'paper', 'scissors': 'rock'}
        
        # Aggression determines counter attack probability (minimum 40%, up to 95%)
        counter_probability = 0.4 + (aggression * 0.55)
        if random.random() < counter_probability:
            return counter_moves[most_common]
        
        # Risk tolerance affects winning streak behavior
        streak_threshold = max(1, int(3 - risk_tolerance * 2))  # 1-3 wins needed
        if self.game_state['streak_data']['wins'] >= streak_threshold:
            return counter_moves[most_common]
        
        return base_move
    
    def _apply_guardian_behavior(self, base_move: str, confidence: float, 
                               human_history: List[str], game_history: List[Tuple[str, str]]) -> str:
        """Apply Guardian personality - defensive and tie-seeking"""
        if len(human_history) < 2 or not self.current_personality:
            return base_move
        
        # Get trait values
        defensiveness = self.current_personality.get_trait(PersonalityTrait.DEFENSIVENESS)
        memory_span = self.current_personality.get_trait(PersonalityTrait.MEMORY_SPAN)
        risk_tolerance = self.current_personality.get_trait(PersonalityTrait.RISK_TOLERANCE)
        confidence_sensitivity = self.current_personality.get_trait(PersonalityTrait.CONFIDENCE_SENSITIVITY)
        
        # Memory span determines analysis window (2-8 moves)
        memory_window = max(2, int(memory_span * 6) + 2)
        
        # Prefer moves that tie with common human moves
        most_common = Counter(human_history[-memory_window:]).most_common(1)[0][0]
        
        # Defensiveness affects loss streak threshold (1-4 losses needed to trigger tie mode)
        loss_threshold = max(1, int(4 - defensiveness * 3))
        if self.game_state['streak_data']['losses'] >= loss_threshold:
            return most_common  # Tie move
        
        # Confidence sensitivity affects safe play threshold
        safe_confidence_threshold = 0.2 + (confidence_sensitivity * 0.4)  # 0.2 to 0.6
        if confidence < safe_confidence_threshold:
            return most_common
        
        # Risk tolerance affects choice between base move and safe move
        safety_probability = 0.3 + (defensiveness * 0.4) - (risk_tolerance * 0.2)  # 0.1 to 0.7
        if random.random() < safety_probability:
            return most_common
        
        return base_move
    
    def _apply_chameleon_behavior(self, base_move: str, confidence: float, 
                                human_history: List[str], game_history: List[Tuple[str, str]]) -> str:
        """Apply Chameleon personality - highly adaptive"""
        if not self.current_personality:
            return base_move
            
        # Get trait values
        adaptability = self.current_personality.get_trait(PersonalityTrait.ADAPTABILITY)
        memory_span = self.current_personality.get_trait(PersonalityTrait.MEMORY_SPAN)
        predictability = self.current_personality.get_trait(PersonalityTrait.PREDICTABILITY)
        
        # Adaptability affects the adaptation threshold (0.1 to 0.5)
        adaptation_threshold = 0.5 - (adaptability * 0.4)
        
        # Memory span determines analysis window (3-8 moves)
        analysis_window = max(3, int(memory_span * 5) + 3)
        
        # Check recent performance to decide adaptation
        if len(self.game_state['recent_performance']) >= analysis_window:
            recent_results = self.game_state['recent_performance'][-analysis_window:]
            recent_wins = recent_results.count('robot')
            win_rate = recent_wins / len(recent_results)
            
            # Poor performance - adapt aggressively
            if win_rate < adaptation_threshold:
                # Switch to counter-strategy
                if len(human_history) >= 3:
                    memory_window = max(3, int(memory_span * 5) + 2)
                    most_common = Counter(human_history[-memory_window:]).most_common(1)[0][0]
                    counter_moves = {'paper': 'scissors', 'rock': 'paper', 'scissors': 'rock'}
                    return counter_moves[most_common]
            
            # Good performance - maintain with slight variations
            elif win_rate > (0.6 + adaptability * 0.2):  # 0.6 to 0.8 threshold
                # Add unpredictability based on predictability trait
                unpredictability_chance = 0.5 - (predictability * 0.4)  # 0.5 to 0.1
                if random.random() < unpredictability_chance:
                    moves = ['paper', 'rock', 'scissors']
                    return random.choice(moves)
        
        return base_move
    
    def _apply_professor_behavior(self, base_move: str, confidence: float, 
                                human_history: List[str], game_history: List[Tuple[str, str]]) -> str:
        """Apply Professor personality - analytical and pattern-based"""
        if len(human_history) < 5 or not self.current_personality:
            return base_move
        
        # Get trait values
        memory_span = self.current_personality.get_trait(PersonalityTrait.MEMORY_SPAN)
        confidence_sensitivity = self.current_personality.get_trait(PersonalityTrait.CONFIDENCE_SENSITIVITY)
        predictability = self.current_personality.get_trait(PersonalityTrait.PREDICTABILITY)
        
        # Memory span determines analysis depth (5-15 moves)
        analysis_depth = max(5, int(memory_span * 10) + 5)
        recent_moves = human_history[-analysis_depth:]
        
        # Look for sequences (bigrams) with high analytical depth
        if len(recent_moves) >= 4:
            last_two = tuple(recent_moves[-2:])
            pattern_predictions = defaultdict(int)
            
            # Count what follows this pattern
            for i in range(len(recent_moves) - 2):
                if tuple(recent_moves[i:i+2]) == last_two and i+2 < len(recent_moves):
                    pattern_predictions[recent_moves[i+2]] += 1
            
            if pattern_predictions:
                predicted_move = max(pattern_predictions.keys(), key=lambda x: pattern_predictions[x])
                counter_moves = {'paper': 'scissors', 'rock': 'paper', 'scissors': 'rock'}
                
                # Confidence sensitivity affects pattern confidence threshold
                pattern_confidence_threshold = 0.4 + (confidence_sensitivity * 0.4)  # 0.4 to 0.8
                if confidence > pattern_confidence_threshold:
                    return counter_moves[predicted_move]
        
        # Fallback to frequency analysis with memory span consideration
        memory_window = max(5, int(memory_span * 8) + 2)
        most_common = Counter(recent_moves[-memory_window:]).most_common(1)[0][0]
        counter_moves = {'paper': 'scissors', 'rock': 'paper', 'scissors': 'rock'}
        
        # Predictability affects how often we use the analytical approach
        analytical_probability = 0.6 + (predictability * 0.3)  # 0.6 to 0.9
        if random.random() < analytical_probability:
            return counter_moves[most_common]
        
        return base_move
    
    def _apply_wildcard_behavior(self, base_move: str, confidence: float, 
                               human_history: List[str], game_history: List[Tuple[str, str]]) -> str:
        """Apply Wildcard personality - chaotic and unpredictable"""
        if not self.current_personality:
            return base_move
            
        # Get trait values
        predictability = self.current_personality.get_trait(PersonalityTrait.PREDICTABILITY)
        risk_tolerance = self.current_personality.get_trait(PersonalityTrait.RISK_TOLERANCE)
        memory_span = self.current_personality.get_trait(PersonalityTrait.MEMORY_SPAN)
        
        # Chaos factor based on predictability (inverted) and risk tolerance
        chaos_factor = (1.0 - predictability) * 0.4 + (risk_tolerance * 0.4)  # 0.0 to 0.8
        
        # High chance of completely random move
        if random.random() < chaos_factor:
            moves = ['paper', 'rock', 'scissors']
            return random.choice(moves)
        
        # Sometimes do the opposite of what makes sense (anti-logic)
        anti_logic_chance = (1.0 - predictability) * 0.5  # 0.0 to 0.5
        if len(human_history) >= 3 and random.random() < anti_logic_chance:
            memory_window = max(2, int(memory_span * 5) + 2)
            most_common = Counter(human_history[-memory_window:]).most_common(1)[0][0]
            # Instead of countering, choose what loses to most common
            lose_to_common = {'paper': 'rock', 'rock': 'scissors', 'scissors': 'paper'}
            return lose_to_common[most_common]
        
        return base_move
    
    def _apply_mirror_behavior(self, base_move: str, confidence: float, 
                             human_history: List[str], game_history: List[Tuple[str, str]]) -> str:
        """Apply Mirror personality - learns and mimics human style"""
        if len(human_history) < 3 or not self.current_personality:
            return base_move
        
        # Get trait values
        adaptability = self.current_personality.get_trait(PersonalityTrait.ADAPTABILITY)
        memory_span = self.current_personality.get_trait(PersonalityTrait.MEMORY_SPAN)
        predictability = self.current_personality.get_trait(PersonalityTrait.PREDICTABILITY)
        
        # Memory span affects how much history to consider
        memory_window = max(3, int(memory_span * 10) + 3)
        relevant_history = human_history[-memory_window:]
        
        # Calculate human's preferred moves
        move_probs = Counter(relevant_history)
        total_moves = len(relevant_history)
        
        # Adaptability affects mimicry strength
        mimicry_chance = 0.3 + (adaptability * 0.4)  # 0.3 to 0.7
        if random.random() < mimicry_chance:
            weighted_moves = []
            for move, count in move_probs.items():
                weighted_moves.extend([move] * count)
            
            if weighted_moves:
                return random.choice(weighted_moves)
        
        # Predictability affects copy behavior
        copy_chance = 0.1 + (predictability * 0.3)  # 0.1 to 0.4
        if random.random() < copy_chance:
            return relevant_history[-1]
        
        return base_move
    
    def get_personality_info(self, personality_name: str) -> Dict:
        """Get detailed information about a personality"""
        if personality_name not in self.personalities:
            return {}
        
        personality = self.personalities[personality_name]
        return {
            'name': personality.name,
            'description': personality.description,
            'traits': {trait.value: value for trait, value in personality.traits.items()},
            'color_theme': personality.color_theme,
            'behavior_modifiers': personality.behavior_modifiers
        }
    
    def get_all_personalities(self) -> List[str]:
        """Get list of all available personality names"""
        return list(self.personalities.keys())
    
    def get_personality_stats(self) -> Dict:
        """Get current personality performance stats"""
        if not self.current_personality:
            return {}
        
        recent_perf = self.game_state['recent_performance']
        total_games = len(recent_perf)
        
        if total_games == 0:
            return {
                'personality': self.current_personality.name,
                'games_played': 0,
                'win_rate': 0,
                'current_streak': self.game_state['streak_data'],
                'adaptation_level': 0
            }
        
        wins = recent_perf.count('robot')
        win_rate = wins / total_games * 100
        
        # Calculate adaptation level based on personality traits and performance
        adaptability = self.current_personality.get_trait(PersonalityTrait.ADAPTABILITY)
        adaptation_level = min(100, adaptability * 100 + self.game_state['adaptation_counter'] * 2)
        
        return {
            'personality': self.current_personality.name,
            'games_played': total_games,
            'win_rate': win_rate,
            'current_streak': self.game_state['streak_data'],
            'adaptation_level': adaptation_level,
            'avg_confidence': statistics.mean(self.game_state['confidence_history']) if self.game_state['confidence_history'] else 0
        }


# Global personality engine instance
personality_engine = AdvancedPersonalityEngine()


def get_personality_engine():
    """Get the global personality engine instance"""
    return personality_engine