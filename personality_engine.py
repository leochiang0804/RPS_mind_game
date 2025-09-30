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
                                human_history: List[str], game_history: List[Tuple[str, str]]) -> str:
        """Apply personality modifications to the base AI move"""
        if not self.current_personality:
            return base_move
        
        personality = self.current_personality
        moves = ['paper', 'stone', 'scissor']
        
        # Berserker personality
        if personality.name == "The Berserker":
            return self._apply_berserker_behavior(base_move, confidence, human_history, game_history)
        
        # Guardian personality
        elif personality.name == "The Guardian":
            return self._apply_guardian_behavior(base_move, confidence, human_history, game_history)
        
        # Chameleon personality
        elif personality.name == "The Chameleon":
            return self._apply_chameleon_behavior(base_move, confidence, human_history, game_history)
        
        # Professor personality
        elif personality.name == "The Professor":
            return self._apply_professor_behavior(base_move, confidence, human_history, game_history)
        
        # Wildcard personality
        elif personality.name == "The Wildcard":
            return self._apply_wildcard_behavior(base_move, confidence, human_history, game_history)
        
        # Mirror personality
        elif personality.name == "The Mirror":
            return self._apply_mirror_behavior(base_move, confidence, human_history, game_history)
        
        return base_move
    
    def _apply_berserker_behavior(self, base_move: str, confidence: float, 
                                human_history: List[str], game_history: List[Tuple[str, str]]) -> str:
        """Apply Berserker personality - extremely aggressive targeting"""
        if len(human_history) < 3:
            return base_move
        
        # Find most common human move and aggressively counter it
        most_common = Counter(human_history[-8:]).most_common(1)[0][0]
        counter_moves = {'paper': 'scissor', 'stone': 'paper', 'scissor': 'stone'}
        
        # High aggression - 80% chance to counter most common move
        if random.random() < 0.8:
            return counter_moves[most_common]
        
        # Check for winning streaks - become even more aggressive
        if self.game_state['streak_data']['wins'] >= 2:
            return counter_moves[most_common]
        
        return base_move
    
    def _apply_guardian_behavior(self, base_move: str, confidence: float, 
                               human_history: List[str], game_history: List[Tuple[str, str]]) -> str:
        """Apply Guardian personality - defensive and tie-seeking"""
        if len(human_history) < 2:
            return base_move
        
        # Prefer moves that tie with common human moves
        most_common = Counter(human_history[-6:]).most_common(1)[0][0]
        
        # If losing streak, prioritize ties
        if self.game_state['streak_data']['losses'] >= 2:
            return most_common  # Tie move
        
        # If low confidence, play it safe
        if confidence < 0.4:
            return most_common
        
        # Otherwise, slightly defensive modification of base move
        safe_moves = ['paper', 'stone', 'scissor']
        return random.choice([base_move, most_common])
    
    def _apply_chameleon_behavior(self, base_move: str, confidence: float, 
                                human_history: List[str], game_history: List[Tuple[str, str]]) -> str:
        """Apply Chameleon personality - highly adaptive"""
        adaptation_threshold = 0.3
        
        # Check recent performance to decide adaptation
        if len(self.game_state['recent_performance']) >= 5:
            recent_wins = self.game_state['recent_performance'][-5:].count('robot')
            win_rate = recent_wins / 5
            
            # Poor performance - adapt aggressively
            if win_rate < adaptation_threshold:
                # Switch to counter-strategy
                if len(human_history) >= 3:
                    most_common = Counter(human_history[-5:]).most_common(1)[0][0]
                    counter_moves = {'paper': 'scissor', 'stone': 'paper', 'scissor': 'stone'}
                    return counter_moves[most_common]
            
            # Good performance - maintain with slight variations
            elif win_rate > 0.6:
                # Add some unpredictability
                if random.random() < 0.3:
                    moves = ['paper', 'stone', 'scissor']
                    return random.choice(moves)
        
        return base_move
    
    def _apply_professor_behavior(self, base_move: str, confidence: float, 
                                human_history: List[str], game_history: List[Tuple[str, str]]) -> str:
        """Apply Professor personality - analytical and pattern-based"""
        if len(human_history) < 5:
            return base_move
        
        # Analyze patterns with high memory span
        recent_moves = human_history[-10:]
        
        # Look for sequences (bigrams)
        if len(recent_moves) >= 4:
            last_two = tuple(recent_moves[-2:])
            pattern_predictions = defaultdict(int)
            
            # Count what follows this pattern
            for i in range(len(recent_moves) - 2):
                if tuple(recent_moves[i:i+2]) == last_two and i+2 < len(recent_moves):
                    pattern_predictions[recent_moves[i+2]] += 1
            
            if pattern_predictions:
                predicted_move = max(pattern_predictions.keys(), key=lambda x: pattern_predictions[x])
                counter_moves = {'paper': 'scissor', 'stone': 'paper', 'scissor': 'stone'}
                
                # High confidence in pattern analysis
                if confidence > 0.6:
                    return counter_moves[predicted_move]
        
        # Fallback to frequency analysis
        most_common = Counter(recent_moves).most_common(1)[0][0]
        counter_moves = {'paper': 'scissor', 'stone': 'paper', 'scissor': 'stone'}
        return counter_moves[most_common]
    
    def _apply_wildcard_behavior(self, base_move: str, confidence: float, 
                               human_history: List[str], game_history: List[Tuple[str, str]]) -> str:
        """Apply Wildcard personality - chaotic and unpredictable"""
        chaos_factor = 0.7
        
        # High chance of completely random move
        if random.random() < chaos_factor:
            moves = ['paper', 'stone', 'scissor']
            return random.choice(moves)
        
        # Sometimes do the opposite of what makes sense
        if len(human_history) >= 3 and random.random() < 0.4:
            most_common = Counter(human_history[-5:]).most_common(1)[0][0]
            # Instead of countering, choose what loses to most common
            lose_to_common = {'paper': 'stone', 'stone': 'scissor', 'scissor': 'paper'}
            return lose_to_common[most_common]
        
        return base_move
    
    def _apply_mirror_behavior(self, base_move: str, confidence: float, 
                             human_history: List[str], game_history: List[Tuple[str, str]]) -> str:
        """Apply Mirror personality - learns and mimics human style"""
        if len(human_history) < 3:
            return base_move
        
        # Calculate human's preferred moves
        move_probs = Counter(human_history)
        total_moves = len(human_history)
        
        # Mirror human's frequency distribution
        if random.random() < 0.6:  # 60% chance to mirror
            weighted_moves = []
            for move, count in move_probs.items():
                weighted_moves.extend([move] * count)
            
            if weighted_moves:
                return random.choice(weighted_moves)
        
        # Sometimes copy last move
        if random.random() < 0.3:
            return human_history[-1]
        
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