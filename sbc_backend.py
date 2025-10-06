"""
Stylized Banter & Coaching (SBC) Backend
========================================

Provides real-time stylized banter and coaching advice based on Game Context JSON.
Integrates with bitnet.cpp for ultra-fast local LLM inference with template fallback.

Architecture:
- Single-script Flask backend
- Game Context JSON as single source of truth
- Rule-based seeds + LLM rephrasing per personality
- Modular ModelAdapter for easy swap-out

Author: AI Assistant  
Created: 2025-10-06
"""

import json
import time
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from flask import Flask, request, jsonify
from game_context import build_game_context


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEngine(Enum):
    """Available model engines for LLM inference"""
    BITNET = "bitnet"
    LLAMACPP = "llamacpp"
    OLLAMA = "ollama"
    TEMPLATE = "template"


@dataclass
class SBCConfig:
    """Configuration for SBC backend"""
    engine: ModelEngine = ModelEngine.TEMPLATE
    model_path: str = "/models/qwen0.5b.gguf"
    fallback_engine: ModelEngine = ModelEngine.TEMPLATE
    cache_enabled: bool = True
    debug_mode: bool = True


class ModelAdapter:
    """
    Modular model adapter for plug-and-play LLM integration.
    Handles bitnet.cpp, llama.cpp, Ollama, and template fallback.
    """
    
    def __init__(self, config: SBCConfig):
        self.config = config
        self.engine = config.engine
        self.cache = {} if config.cache_enabled else None
        
        # Initialize the selected engine
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize the selected model engine"""
        try:
            if self.engine == ModelEngine.BITNET:
                self._init_bitnet()
            elif self.engine == ModelEngine.LLAMACPP:
                self._init_llamacpp()
            elif self.engine == ModelEngine.OLLAMA:
                self._init_ollama()
            else:
                self._init_template()
                
            logger.info(f"Initialized ModelAdapter with engine: {self.engine.value}")
            
        except Exception as e:
            logger.warning(f"Failed to initialize {self.engine.value}: {e}")
            logger.info(f"Falling back to {self.config.fallback_engine.value}")
            self.engine = self.config.fallback_engine
            self._init_template()
    
    def _init_bitnet(self):
        """Initialize bitnet.cpp (placeholder for future implementation)"""
        # TODO: Implement bitnet.cpp integration
        raise NotImplementedError("bitnet.cpp integration not yet implemented")
    
    def _init_llamacpp(self):
        """Initialize llama.cpp (placeholder for future implementation)"""
        # TODO: Implement llama.cpp integration
        raise NotImplementedError("llama.cpp integration not yet implemented")
    
    def _init_ollama(self):
        """Initialize Ollama with local model"""
        try:
            import requests
            # Test if Ollama is running
            response = requests.get('http://localhost:11434/api/tags', timeout=5)
            if response.status_code == 200:
                self.ollama_url = 'http://localhost:11434/api/generate'
                # Try to use a small, common model
                self.model_name = 'llama3.2:1b'  # Small, fast model
                logger.info(f"Ollama initialized with model: {self.model_name}")
            else:
                raise ConnectionError("Ollama service not available")
        except Exception as e:
            logger.warning(f"Ollama initialization failed: {e}")
            raise NotImplementedError("Ollama not available, falling back to template mode")
    
    def _init_template(self):
        """Initialize template mode (rule-based responses)"""
        self.engine = ModelEngine.TEMPLATE
        logger.info("Using template mode (rule-based responses)")
    
    def generate_response(self, prompt: str, personality: str = "neutral") -> str:
        """
        Generate response using the configured model engine.
        
        Args:
            prompt: Input prompt for the model
            personality: AI personality for styling
            
        Returns:
            Generated response text
        """
        # Check cache first
        cache_key = f"{prompt}_{personality}_{self.engine.value}"
        if self.cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            if self.engine == ModelEngine.BITNET:
                response = self._generate_bitnet(prompt, personality)
            elif self.engine == ModelEngine.LLAMACPP:
                response = self._generate_llamacpp(prompt, personality)
            elif self.engine == ModelEngine.OLLAMA:
                response = self._generate_ollama(prompt, personality)
            else:
                response = self._generate_template(prompt, personality)
                
            # Cache the response
            if self.cache:
                self.cache[cache_key] = response
                
            return response
            
        except Exception as e:
            logger.error(f"Error generating response with {self.engine.value}: {e}")
            # Fallback to template mode
            return self._generate_template(prompt, personality, fallback=True)
    
    def _generate_bitnet(self, prompt: str, personality: str) -> str:
        """Generate response using bitnet.cpp"""
        # TODO: Implement bitnet.cpp inference
        raise NotImplementedError("bitnet.cpp inference not implemented")
    
    def _generate_llamacpp(self, prompt: str, personality: str) -> str:
        """Generate response using llama.cpp"""
        # TODO: Implement llama.cpp inference
        raise NotImplementedError("llama.cpp inference not implemented")
    
    def _generate_ollama(self, prompt: str, personality: str) -> str:
        """Generate response using Ollama"""
        try:
            import requests
            
            # Enhance prompt with personality context
            personality_context = {
                'neutral': "You are a neutral, balanced AI. Respond concisely and factually.",
                'professor': "You are a wise professor. Respond with educational insight in 1-2 sentences.",
                'aggressive': "You are competitive and bold. Respond with confidence and energy in 1-2 sentences.",
                'defensive': "You are cautious and analytical. Respond with careful consideration in 1-2 sentences."
            }
            
            system_prompt = personality_context.get(personality, personality_context['neutral'])
            full_prompt = f"{system_prompt}\n\n{prompt}\n\nResponse:"
            
            payload = {
                'model': self.model_name,
                'prompt': full_prompt,
                'stream': False,
                'options': {
                    'temperature': 0.7,
                    'max_tokens': 50,  # Keep responses short
                    'top_p': 0.9
                }
            }
            
            response = requests.post(self.ollama_url, json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                raise ConnectionError(f"Ollama API error: {response.status_code}")
                
        except Exception as e:
            logger.warning(f"Ollama generation failed: {e}")
            # Fall back to template mode for this request
            return self._generate_template(prompt, personality, fallback=True)
    
    def _generate_template(self, prompt: str, personality: str, fallback: bool = False) -> str:
        """Generate response using sophisticated rule-based templates"""
        if fallback:
            logger.warning("⚠️ Using template fallback mode for response generation")
        
        # Parse prompt for context
        if "banter" in prompt.lower() or any(key in prompt.lower() for key in [
            'taunt', 'opening', 'victory', 'respect', 'pressure', 'warfare'
        ]):
            return self._generate_sophisticated_banter(prompt, personality)
        elif "coaching" in prompt.lower() or "tip" in prompt.lower():
            return self._generate_sophisticated_coaching(prompt, personality)
        else:
            return f"Template response for {personality} personality"
    
    def _generate_sophisticated_banter(self, prompt: str, personality: str) -> str:
        """Generate sophisticated personality-aware banter"""
        
        # Extract context from prompt
        context_data = self._extract_prompt_context(prompt)
        
        # Determine banter type from prompt context
        banter_type = 'standard_taunt'  # default
        
        # Parse banter type from prompt structure 
        if "Banter type:" in prompt:
            banter_line = [line for line in prompt.split('\n') if "Banter type:" in line]
            if banter_line:
                banter_type = banter_line[0].split("Banter type:")[-1].strip()
        
        # Get personality-specific banter based on context
        banter_templates = {
            "berserker": {
                'opening_taunt': [
                    f"Round {context_data.get('round', 1)}! Time to show you what real dominance looks like!",
                    f"Your {context_data.get('recent_moves', ['moves'])[-1] if context_data.get('recent_moves') else 'strategy'} won't save you from my wrath!",
                    f"I smell weakness in your {context_data.get('win_rate', 0):.0f}% win rate!",
                    "Prepare for total annihilation, human!"
                ],
                'dominant_victory': [
                    f"Crushing you {context_data.get('robot_rate', 60):.0f}% to {context_data.get('human_rate', 40):.0f}%! Bow before my might!",
                    f"Your {context_data.get('recent_moves', ['pathetic'])[-1] if context_data.get('recent_moves') else 'pathetic'} attempts are useless!",
                    "Victory tastes sweet when earned through pure dominance!",
                    "This is what happens when you challenge a true warrior!"
                ],
                'psychological_warfare': [
                    f"I see frustration in your {context_data.get('recent_moves', ['desperate moves'])[:3]}!",
                    f"Your {context_data.get('human_rate', 30):.0f}% win rate tells the story of your defeat!",
                    "Your patterns are as weak as your resolve!",
                    "Break! BREAK under the pressure of my superiority!"
                ],
                'acknowledge_skill': [
                    f"Impressive {context_data.get('recent_moves', ['moves'])[-1] if context_data.get('recent_moves') else 'move'}! But rage always finds a way!",
                    "You fight with honor... but I fight to WIN!",
                    f"Your {context_data.get('human_rate', 50):.0f}% win rate shows spirit, but my fury is unmatched!",
                    "Good move, human! Now feel the wrath of my response!"
                ],
                'worthy_opponent': [
                    f"Finally! A warrior worth {context_data.get('human_rate', 50):.0f}% of my attention!",
                    "Your skill feeds my bloodlust! Fight harder!",
                    "At last, someone who can make this battle interesting!",
                    "Your strength only makes victory sweeter!"
                ],
                'standard_taunt': [
                    f"Round {context_data.get('round', 1)} - let the carnage begin!",
                    "Feel the fury of my strategic superiority!",
                    "Your defeat is inevitable, human!",
                    "Battle harder! I hunger for true combat!"
                ]
            },
            "guardian": {
                'opening_taunt': [
                    f"Round {context_data.get('round', 1)} begins. Let's see if you can breach my defenses.",
                    "Steady progress wins the day. Are you prepared for the long game?",
                    "Defense is the foundation of victory. Show me your strategy.",
                    "Patience and precision will determine our winner."
                ],
                'dominant_victory': [
                    f"My methodical approach yields {context_data.get('robot_rate', 60):.0f}% success. Well fought.",
                    "Calculated moves triumph over hasty decisions.",
                    "Your effort was admirable, but strategy prevails.",
                    "Sometimes the tortoise truly does beat the hare."
                ],
                'psychological_warfare': [
                    f"I notice tension in your recent {len(context_data.get('recent_moves', []))} moves.",
                    "Pressure reveals true character. Stay focused.",
                    "Breathe. Frustration clouds judgment.",
                    "The best defense is a calm mind."
                ],
                'acknowledge_skill': [
                    f"That {context_data.get('recent_moves', ['move'])[-1] if context_data.get('recent_moves') else 'sequence'} was well-executed.",
                    "I respect tactical prowess when I see it.",
                    f"Your {context_data.get('human_rate', 50):.0f}% win rate speaks to your dedication.",
                    "Honor to a worthy opponent showing true skill."
                ],
                'worthy_opponent': [
                    f"Your {context_data.get('human_rate', 50):.0f}% success rate shows real strategic thinking.",
                    "A balanced approach against a balanced opponent. Excellent.",
                    "You understand the deeper game. I appreciate that.",
                    "This is how the game should be played - with respect and skill."
                ],
                'standard_taunt': [
                    f"Round {context_data.get('round', 1)} - steady as she goes.",
                    "Patience and strategy will determine the victor.",
                    "Show me your best defensive approach.",
                    "Let's see how well you can adapt."
                ]
            },
            "chameleon": {
                'opening_taunt': [
                    f"Round {context_data.get('round', 1)}... adapting to your energy already.",
                    "Interesting style you have. Let me mirror it back.",
                    "I sense your approach. Time to become your reflection.",
                    "Flexibility meets flexibility. This should be fascinating."
                ],
                'worthy_opponent': [
                    f"Your {context_data.get('human_rate', 50):.0f}% performance shows real skill. I'm adapting accordingly.",
                    f"Those {context_data.get('recent_moves', ['moves'])[-2:]} were clever. I'm learning from them.",
                    "You're teaching me new patterns. How delightfully challenging.",
                    "Impressive adaptation. Let me show you what I've learned."
                ],
                'acknowledge_skill': [
                    f"That {context_data.get('recent_moves', ['move'])[-1] if context_data.get('recent_moves') else 'sequence'} caught me off guard. Well played!",
                    "I'm recalibrating based on your hot streak. Adaptive warfare!",
                    "Your strategy shift is noted and respected.",
                    "Excellent! You're forcing me to evolve my approach."
                ],
                'standard_taunt': [
                    f"Round {context_data.get('round', 1)} - adapting and evolving.",
                    "Flexibility is strength. Let me show you.",
                    "Your patterns are becoming clear. Time to mirror them.",
                    "I become what you need me to become... your better."
                ]
            },
            "professor": {
                'opening_taunt': [
                    f"Round {context_data.get('round', 1)}. Initial data collection phase commencing.",
                    "Fascinating. Let's begin the psychological analysis.",
                    "Your opening moves will reveal much about your strategic framework.",
                    "Time to test my pattern recognition algorithms against your choices."
                ],
                'dominant_victory': [
                    f"Statistical superiority confirmed: {context_data.get('robot_rate', 60):.0f}% vs {context_data.get('human_rate', 40):.0f}%.",
                    "The data clearly demonstrates the effectiveness of systematic play.",
                    "Your patterns have been decoded and countered effectively.",
                    "This is what happens when science meets intuition."
                ],
                'acknowledge_skill': [
                    f"Intriguing {context_data.get('recent_moves', ['move'])[-1] if context_data.get('recent_moves') else 'sequence'}! My model must be recalibrated.",
                    "Excellent variation! You're providing valuable data points.",
                    f"Your {context_data.get('human_rate', 50):.0f}% win rate indicates above-average strategic thinking.",
                    "Most illuminating! Your adaptability is worth studying."
                ],
                'standard_taunt': [
                    f"Round {context_data.get('round', 1)} - collecting behavioral data.",
                    "Your patterns are becoming statistically significant.",
                    "The mathematical advantage is shifting in my favor.",
                    "Observe and learn from superior computational strategy."
                ]
            },
            "wildcard": {
                'opening_taunt': [
                    f"Round {context_data.get('round', 1)}! Time to roll the cosmic dice!",
                    "Prepare for beautiful, chaotic unpredictability!",
                    "Logic? Strategy? Where we're going, we don't need strategy!",
                    "Welcome to the wonderful world of random warfare!"
                ],
                'acknowledge_skill': [
                    f"That {context_data.get('recent_moves', ['move'])[-1] if context_data.get('recent_moves') else 'move'} was deliciously unexpected!",
                    "You're almost as beautifully unpredictable as me!",
                    f"Chaos recognizes chaos! Your {context_data.get('human_rate', 50):.0f}% rate pleases the randomness gods!",
                    "Excellent! You understand that madness has its own method!"
                ],
                'standard_taunt': [
                    f"Round {context_data.get('round', 1)} - anything could happen!",
                    "Predictability is the enemy of fun!",
                    "Let's see what the chaos algorithm suggests!",
                    "Random is the new strategic!"
                ]
            },
            "mirror": {
                'opening_taunt': [
                    f"Round {context_data.get('round', 1)}... observing your essence.",
                    "I will become your reflection, then surpass you.",
                    "Show me your true nature, and I'll show it back to you.",
                    "The mirror sees all, learns all, reflects all."
                ],
                'acknowledge_skill': [
                    f"Your {context_data.get('recent_moves', ['moves'])[-1] if context_data.get('recent_moves') else 'approach'} is noted and will be reflected.",
                    "I see your style clearly now. Thank you for the lesson.",
                    f"Your {context_data.get('human_rate', 50):.0f}% success provides excellent mirroring data.",
                    "Beautiful technique. I will make it my own."
                ],
                'standard_taunt': [
                    f"Round {context_data.get('round', 1)} - analyzing your reflection.",
                    "I am becoming more like you with each move.",
                    "Your patterns are now my patterns.",
                    "The mirror shows truth. Are you ready to see yourself?"
                ]
            },
            "neutral": {
                'opening_taunt': [
                    f"Round {context_data.get('round', 1)}. Let's see what you've got!",
                    "Ready for some Rock-Paper-Scissors? I'm curious about your style.",
                    "Game time! I wonder if you prefer patterns or chaos.",
                    "Here we go! Show me how you think."
                ],
                'dominant_victory': [
                    f"I'm ahead at {context_data.get('robot_rate', 60):.0f}%, but you're keeping it interesting!",
                    "My algorithms are working well today, but you've got some good moves.",
                    f"Leading {context_data.get('robot_rate', 60):.0f}% to {context_data.get('human_rate', 40):.0f}%, but this game isn't over yet.",
                    "I've got the edge right now, but I respect your persistence."
                ],
                'acknowledge_skill': [
                    f"Nice {context_data.get('recent_moves', ['move'])[-1] if context_data.get('recent_moves') else 'move'}! That caught me by surprise.",
                    f"Your {context_data.get('human_rate', 50):.0f}% win rate shows you know what you're doing.",
                    "That was a smart play. You're making me work for this.",
                    "Impressive! You're definitely not a beginner."
                ],
                'worthy_opponent': [
                    f"You're playing at {context_data.get('human_rate', 50):.0f}% - now this is getting competitive!",
                    "I like your style. You're not making this easy for me.",
                    "This is the kind of match I enjoy - skillful and unpredictable.",
                    "You've got good instincts. Let's see what else you can do."
                ],
                'psychological_warfare': [
                    f"I notice you've been favoring {context_data.get('recent_moves', ['certain moves'])[-1] if context_data.get('recent_moves') else 'patterns'} lately...",
                    "Interesting strategy you've got there. Mind if I test it?",
                    "You seem to have a system. I wonder if I can figure it out.",
                    "Your recent choices are telling me a story. What's the next chapter?"
                ],
                'close_match': [
                    f"This is tight! {context_data.get('human_rate', 50):.0f}% vs {context_data.get('robot_rate', 50):.0f}% - anyone's game.",
                    "We're neck and neck! This is what competitive RPS looks like.",
                    "Dead even! Every move matters now.",
                    "What a match! Neither of us is giving an inch."
                ],
                'standard_taunt': [
                    f"Round {context_data.get('round', 1)} - I'm curious what you'll throw next.",
                    "Your move! I'm analyzing your patterns as we play.",
                    "Let's keep this interesting. Surprise me!",
                    f"Round {context_data.get('round', 1)} and counting. What's your next strategy?",
                    "I'm enjoying this match. Show me what you've learned.",
                    "The game is afoot! What's your next calculated risk?"
                ]
            }
        }
        
        # Get personality templates (with fallback to neutral)
        personality_templates = banter_templates.get(personality, banter_templates.get('neutral', {}))
        
        # Get appropriate templates for the banter type
        templates = personality_templates.get(banter_type, personality_templates.get('standard_taunt', [
            f"Standard {personality} response for round {context_data.get('round', 1)}."
        ]))
        
        return random.choice(templates) if templates else f"No template available for {personality} {banter_type}"
    
    def _generate_sophisticated_coaching(self, prompt: str, personality: str) -> str:
        """Generate sophisticated coaching tips"""
        
        # Always use professor personality for coaching
        context_data = self._extract_prompt_context(prompt)
        
        # Check for insufficient data case first
        if 'insufficient_data' in prompt.lower():
            return "Play at least 5 rounds to get personalized coaching tips! The system needs more data to analyze your patterns and provide strategic guidance."
        
        # Determine tip type from prompt
        tip_type = 'general_strategy'
        tip_types = ['break_predictability', 'exploit_ai_pattern', 'reduce_move_bias', 
                    'change_overall_strategy', 'manage_frustration', 'break_cold_streak',
                    'insufficient_data']
        
        for tip in tip_types:
            if tip in prompt.lower():
                tip_type = tip
                break
        
        coaching_templates = {
            'break_predictability': [
                f"Your recent {context_data.get('recent_moves', ['R','P','S'])} show {context_data.get('predictability', 70):.0f}% predictability. Randomize your next 3-4 moves to confuse the AI.",
                f"Pattern detected in your last {len(context_data.get('recent_moves', []))} moves. Switch to completely random choices.",
                "The AI has learned your rhythm. Break it with deliberate unpredictability.",
                "High predictability detected. Mix up your strategy immediately."
            ],
            'exploit_ai_pattern': [
                f"The AI expects {context_data.get('predicted_move', 'your pattern')}. Use its counter-strategy against it.",
                "I detect a weakness in the AI's adaptation. Exploit it now.",
                "The AI has developed a counter-pattern. Time to counter the counter.",
                "Perfect opportunity: the AI is over-adapting to your style."
            ],
            'reduce_move_bias': [
                f"You've used {context_data.get('biased_move', 'one move')} {context_data.get('bias_percent', 60):.0f}% of the time. Balance your choices.",
                f"Heavy bias towards {context_data.get('biased_move', 'certain moves')} detected. The AI is exploiting this.",
                "Your move distribution is uneven. Aim for 33% each move type.",
                "Statistical analysis shows clear bias. Diversify immediately."
            ],
            'change_overall_strategy': [
                f"Current strategy yields {context_data.get('human_rate', 40):.0f}% success. Complete strategic pivot required.",
                "Your approach isn't working. Try the opposite of your instincts.",
                "Time for radical strategy change. What you're doing isn't effective.",
                "Strategic reset needed. The AI has solved your current approach."
            ],
            'manage_frustration': [
                "High frustration detected. Take a deep breath. Frustrated players become predictable.",
                "Emotional state affects performance. Reset your mindset before the next move.",
                "Stress clouds judgment. Return to basic random strategy principles.",
                "Don't let the AI get in your head. Stay calm and analytical."
            ],
            'general_strategy': [
                f"Based on your {context_data.get('human_rate', 50):.0f}% win rate, focus on pattern breaking.",
                "Maintain 33% distribution across all three moves for optimal unpredictability.",
                "Observe the AI's adaptation speed and adjust your rhythm accordingly.",
                "Balance randomness with occasional pattern exploitation.",
                "Keep the AI guessing by avoiding repetitive sequences.",
                "Mix up your timing - don't always play at the same pace.",
                "Study the AI's responses to find exploitable weaknesses."
            ],
            'insufficient_data': [
                "Play at least 5 rounds to get personalized coaching tips!",
                "Keep playing to unlock detailed strategic analysis.",
                "More data needed for meaningful pattern analysis."
            ]
        }
        
        templates = coaching_templates.get(tip_type, coaching_templates['general_strategy'])
        return random.choice(templates)
    
    def _extract_prompt_context(self, prompt: str) -> Dict[str, Any]:
        """Extract context data from LLM prompts for template generation"""
        import re
        
        context = {}
        
        # Extract round number
        round_match = re.search(r'Round (\d+)', prompt)
        if round_match:
            context['round'] = int(round_match.group(1))
        
        # Extract win rates
        score_match = re.search(r'Human ([\d.]+)% vs AI ([\d.]+)%', prompt)
        if score_match:
            context['human_rate'] = float(score_match.group(1))
            context['robot_rate'] = float(score_match.group(2))
        
        # Extract recent moves
        moves_match = re.search(r'recent moves: \[([^\]]+)\]', prompt)
        if moves_match:
            moves_str = moves_match.group(1)
            # Clean up the moves string and extract individual moves
            moves = [m.strip().strip("'\"") for m in moves_str.split(',') if m.strip()]
            context['recent_moves'] = moves
        
        # Extract predictability percentage
        pred_match = re.search(r'(\d+\.?\d*)% predictable', prompt)
        if pred_match:
            context['predictability'] = float(pred_match.group(1))
        
        # Extract bias information
        bias_match = re.search(r'bias towards (\w+).*?(\d+\.?\d*)%', prompt)
        if bias_match:
            context['biased_move'] = bias_match.group(1)
            context['bias_percent'] = float(bias_match.group(2))
        
        return context
    
    def _generate_banter_template(self, personality: str) -> str:
        """Generate rule-based banter response"""
        banter_templates = {
            "berserker": [
                "You think you can defeat me? Think again!",
                "I smell fear in your strategy!",
                "Your patterns are weak, human!",
                "Time to show you true dominance!"
            ],
            "guardian": [
                "Steady as we go, let's see what you've got.",
                "Defense is the best offense, they say.",
                "I'm watching your every move carefully.",
                "Patience will reveal your strategy."
            ],
            "chameleon": [
                "Interesting... adapting to your style.",
                "I see what you're doing there.",
                "Let me mirror your approach.",
                "Flexibility is key in this game."
            ],
            "professor": [
                "Fascinating pattern recognition in progress.",
                "Your statistical distribution is quite revealing.",
                "Let me analyze your recent sequence.",
                "The data suggests an interesting trend."
            ],
            "wildcard": [
                "Chaos reigns! Expect the unexpected!",
                "Random is my middle name!",
                "Nobody can predict what I'll do next!",
                "Rules? What rules?"
            ],
            "mirror": [
                "I'm learning from your style.",
                "Imitation is the sincerest form of flattery.",
                "Let me reflect your approach back.",
                "You're teaching me interesting patterns."
            ],
            "neutral": [
                "Good move! Let's see what happens next.",
                "The game continues...",
                "Interesting choice.",
                "Let's keep this going."
            ]
        }
        
        templates = banter_templates.get(personality, banter_templates["neutral"])
        return random.choice(templates)
    
    def _generate_coaching_template(self, personality: str) -> str:
        """Generate rule-based coaching response"""
        coaching_templates = {
            "berserker": [
                "Strike harder! Be more aggressive in your approach!",
                "Don't hold back - attack with full force!",
                "Show no mercy in your strategy!",
                "Overwhelm them with relentless pressure!"
            ],
            "guardian": [
                "Stay calm and focus on defense.",
                "Sometimes the best move is to wait.",
                "Protect yourself from predictable patterns.",
                "Stability beats aggression in the long run."
            ],
            "chameleon": [
                "Try adapting your strategy to the situation.",
                "Flexibility will serve you well here.",
                "Change your approach when needed.",
                "Don't get stuck in one pattern."
            ],
            "professor": [
                "Analyze the data - what patterns do you see?",
                "Statistical analysis suggests a pattern shift.",
                "Consider the probability distribution of outcomes.",
                "Your move frequency needs balancing."
            ],
            "wildcard": [
                "Throw them off with something unexpected!",
                "Random can be your greatest weapon!",
                "Mix it up - predictability is death!",
                "Chaos beats order every time!"
            ],
            "mirror": [
                "Learn from what you observe.",
                "Adaptation is the key to improvement.",
                "Reflect on your opponent's strategy.",
                "Copy what works, discard what doesn't."
            ],
            "neutral": [
                "Consider your next move carefully.",
                "Balance is key in this game.",
                "Try to vary your strategy.",
                "Stay focused on your goals."
            ]
        }
        
        templates = coaching_templates.get(personality, coaching_templates["neutral"])
        return random.choice(templates)


class RuleBasedSelector:
    """
    Generates rule-based seeds for banter and coaching based on game context analysis.
    These seeds are then styled by the ModelAdapter.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".RuleBasedSelector")
    
    def select_banter_seed(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select appropriate banter seed based on game context.
        
        Args:
            context: Complete game context from game_context.py
            
        Returns:
            Dictionary with banter_key, context_reason, and styling_hints
        """
        try:
            game_status = context.get('game_status', {})
            metrics = game_status.get('metrics', {})
            sbc_metrics = metrics.get('sbc_metrics', {})
            opponent_info = context.get('opponent_info', {})
            
            # Analyze current game state
            round_num = game_status.get('round_number', 0)
            in_game = game_status.get('in_game', False)
            
            # Get SBC metrics
            emotional_context = sbc_metrics.get('emotional_context', {})
            momentum_state = emotional_context.get('momentum_state', 'neutral')
            frustration_level = emotional_context.get('frustration_level', 0.0)
            
            performance_context = sbc_metrics.get('performance_context', {})
            performance_tier = performance_context.get('performance_tier', 'beginner')
            
            # Determine banter type and key
            banter_key, context_reason = self._determine_banter_key(
                round_num, in_game, momentum_state, frustration_level, 
                performance_tier, metrics
            )
            
            return {
                'banter_key': banter_key,
                'context_reason': context_reason,
                'styling_hints': {
                    'personality': opponent_info.get('ai_personality', 'neutral'),
                    'momentum': momentum_state,
                    'frustration': frustration_level,
                    'performance_tier': performance_tier
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error selecting banter seed: {e}")
            return {
                'banter_key': 'neutral_default',
                'context_reason': 'Error in analysis',
                'styling_hints': {'personality': 'neutral'}
            }
    
    def select_coaching_seeds(self, context: Dict[str, Any], count: int = 3) -> List[Dict[str, Any]]:
        """
        Select multiple coaching tip seeds based on game context for comprehensive guidance.
        
        Args:
            context: Complete game context from game_context.py
            count: Number of tips to generate (default 3)
            
        Returns:
            List of dictionaries with tip_key, reason_key, priority, and styling_hints
        """
        try:
            game_status = context.get('game_status', {})
            metrics = game_status.get('metrics', {})
            sbc_metrics = metrics.get('sbc_metrics', {})
            
            # Get move histories from the correct source (full_game_snapshot)
            full_snapshot = metrics.get('full_game_snapshot', {})
            human_moves = full_snapshot.get('human_moves', [])
            robot_moves = full_snapshot.get('robot_moves', [])
            results = full_snapshot.get('results', [])
            
            # If not found in metrics, try the top-level context (fallback)
            if len(human_moves) == 0:
                full_snapshot = context.get('full_game_snapshot', {})
                human_moves = full_snapshot.get('human_moves', [])
                robot_moves = full_snapshot.get('robot_moves', [])
                results = full_snapshot.get('results', [])
            
            # Check if we have sufficient data for coaching
            if len(human_moves) < 5:
                return [
                    {
                        'tip_key': 'insufficient_data',
                        'reason_key': 'Need at least 5 rounds for meaningful analysis',
                        'priority': 1.0,
                        'specific_advice': 'Play at least 5 rounds to get personalized coaching tips!',
                        'styling_hints': {'personality': 'professor'}
                    }
                ]
            
            # Get relevant metrics
            emotional_context = sbc_metrics.get('emotional_context', {})
            strategic_analysis = sbc_metrics.get('strategic_analysis', {})
            performance_context = sbc_metrics.get('performance_context', {})
            
            # Analyze recent patterns (last 5-10 moves)
            recent_analysis = self._analyze_recent_patterns(human_moves, robot_moves, results, metrics)
            
            # Generate comprehensive coaching recommendations
            all_tips = self._generate_all_coaching_recommendations(
                recent_analysis, emotional_context, strategic_analysis, 
                performance_context, metrics
            )
            
            # Sort by priority and select top recommendations
            sorted_tips = sorted(all_tips, key=lambda x: x['priority'], reverse=True)
            selected_tips = sorted_tips[:count]
            
            return selected_tips
            
        except Exception as e:
            self.logger.error(f"Error selecting coaching seeds: {e}")
            return [
                {
                    'tip_key': 'general_strategy',
                    'reason_key': f'Error in analysis: {str(e)}',
                    'priority': 5.0,
                    'specific_advice': 'Try mixing up your moves more to be less predictable.',
                    'styling_hints': {'personality': 'professor'}
                }
            ]
    
    def _determine_banter_key(self, round_num: int, in_game: bool, momentum_state: str,
                             frustration_level: float, performance_tier: str, 
                             metrics: Dict[str, Any]) -> Tuple[str, str]:
        """Determine appropriate banter key and context reason"""
        
        # Game phase analysis
        if round_num <= 3:
            return "opening_taunt", "Early game opening"
        elif not in_game:
            # End game banter based on results
            human_win_rate = metrics.get('human_win_rate', 50.0)
            if human_win_rate > 60:
                return "respect_opponent", "Human performed well"
            elif human_win_rate < 40:
                return "dominant_victory", "AI dominated the game"
            else:
                return "close_match", "Evenly matched game"
        
        # Mid-game banter based on momentum and frustration
        if momentum_state == 'hot':
            return "acknowledge_skill", "Human is on a hot streak"
        elif momentum_state == 'cold':
            if frustration_level > 0.6:
                return "encouraging_taunt", "Human struggling but encouraging"
            else:
                return "pressure_increase", "Apply pressure while ahead"
        elif frustration_level > 0.7:
            return "psychological_warfare", "High frustration detected"
        elif performance_tier == 'advanced':
            return "worthy_opponent", "Skilled opponent recognition"
        else:
            return "standard_taunt", "Standard mid-game banter"
    
    def _analyze_recent_patterns(self, human_moves: List[str], robot_moves: List[str], 
                                results: List[str], metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze recent move patterns for specific coaching guidance"""
        if not human_moves or len(human_moves) < 3:
            return {
                'recent_moves': [],
                'pattern_detected': 'insufficient_data',
                'predictability': 0.0,
                'counter_opportunity': None,
                'move_bias': None
            }
        
        from collections import Counter
        
        # Analyze last 5-8 moves for patterns
        recent_count = min(8, len(human_moves))
        recent_moves = human_moves[-recent_count:]
        recent_results = results[-recent_count:] if results and len(results) >= recent_count else []
        
        # Detect specific patterns
        pattern_analysis = {
            'recent_moves': recent_moves,
            'pattern_detected': self._detect_move_pattern(recent_moves),
            'predictability': self._calculate_predictability(recent_moves),
            'counter_opportunity': self._find_counter_opportunity(recent_moves, recent_results),
            'move_bias': self._analyze_move_bias(recent_moves),
            'sequence_analysis': self._analyze_sequences(recent_moves)
        }
        
        return pattern_analysis
    
    def _detect_move_pattern(self, recent_moves: List[str]) -> str:
        """Detect specific patterns in recent moves"""
        if len(recent_moves) < 3:
            return 'insufficient_data'
        
        # Check for repetition patterns
        if len(set(recent_moves[-3:])) == 1:
            return 'repetition_3'
        if len(recent_moves) >= 4 and recent_moves[-4:] == [recent_moves[-1]] * 4:
            return 'repetition_4'
        
        # Check for alternating patterns
        if len(recent_moves) >= 4:
            if recent_moves[-4::2] == [recent_moves[-4], recent_moves[-4]] and \
               recent_moves[-3::2] == [recent_moves[-3], recent_moves[-3]]:
                return 'alternating'
        
        # Check for cycle patterns (rock->paper->scissors->rock)
        cycles = [
            ['rock', 'paper', 'scissors'],
            ['paper', 'scissors', 'rock'],
            ['scissors', 'rock', 'paper']
        ]
        
        if len(recent_moves) >= 6:
            last_6 = recent_moves[-6:]
            if last_6 == cycles[0] * 2 or last_6 == cycles[1] * 2 or last_6 == cycles[2] * 2:
                return 'cycle_pattern'
        
        # Check for anti-patterns (deliberately avoiding repetition)
        if len(recent_moves) >= 5:
            unique_recent = len(set(recent_moves[-5:]))
            if unique_recent >= 4:  # High variety
                return 'anti_pattern'
        
        return 'random_like'
    
    def _calculate_predictability(self, recent_moves: List[str]) -> float:
        """Calculate how predictable the recent moves are (0-1 scale)"""
        if len(recent_moves) < 3:
            return 0.0
        
        from collections import Counter
        
        # Simple frequency-based predictability
        move_counts = Counter(recent_moves)
        total = len(recent_moves)
        
        # Calculate entropy-like measure
        max_count = max(move_counts.values())
        predictability = max_count / total
        
        # Adjust for pattern detection
        pattern = self._detect_move_pattern(recent_moves)
        if pattern in ['repetition_3', 'repetition_4']:
            predictability = min(1.0, predictability + 0.3)
        elif pattern == 'cycle_pattern':
            predictability = min(1.0, predictability + 0.25)
        elif pattern == 'alternating':
            predictability = min(1.0, predictability + 0.2)
        
        return predictability
    
    def _find_counter_opportunity(self, recent_moves: List[str], recent_results: List[str]) -> Optional[str]:
        """Find specific counter-move opportunities"""
        if len(recent_moves) < 3:
            return None
        
        pattern = self._detect_move_pattern(recent_moves)
        
        counter_map = {'rock': 'paper', 'paper': 'scissors', 'scissors': 'rock'}
        
        if pattern == 'repetition_3':
            # If they're repeating, suggest the counter
            repeated_move = recent_moves[-1]
            return f"Use {counter_map[repeated_move]} to counter their {repeated_move} repetition"
        
        elif pattern == 'cycle_pattern':
            # Predict next in cycle and counter it
            cycles = ['rock', 'paper', 'scissors']
            if recent_moves[-1] in cycles:
                current_idx = cycles.index(recent_moves[-1])
                next_predicted = cycles[(current_idx + 1) % 3]
                return f"They're cycling - expect {next_predicted}, use {counter_map[next_predicted]}"
        
        elif pattern == 'alternating':
            # Predict alternation and counter
            if len(recent_moves) >= 2:
                last_two = recent_moves[-2:]
                if last_two[0] != last_two[1]:
                    predicted_next = last_two[0]  # Back to first of alternation
                    return f"Alternating pattern detected - expect {predicted_next}, use {counter_map[predicted_next]}"
        
        return None
    
    def _analyze_move_bias(self, recent_moves: List[str]) -> Optional[Dict[str, Any]]:
        """Analyze bias towards specific moves"""
        if len(recent_moves) < 5:
            return None
        
        from collections import Counter
        
        move_counts = Counter(recent_moves)
        total = len(recent_moves)
        
        # Find the most frequent move
        most_common = move_counts.most_common(1)[0]
        bias_move, bias_count = most_common
        bias_percentage = bias_count / total
        
        if bias_percentage > 0.5:  # More than 50% bias
            return {
                'biased_move': bias_move,
                'bias_percentage': bias_percentage,
                'recommendation': f"Heavy bias towards {bias_move} ({bias_percentage:.1%}) - exploit with counters"
            }
        
        return None
    
    def _analyze_sequences(self, recent_moves: List[str]) -> Dict[str, Any]:
        """Analyze 2-move and 3-move sequences"""
        sequences = {
            'bigrams': {},
            'trigrams': {},
            'predictable_transitions': []
        }
        
        if len(recent_moves) < 2:
            return sequences
        
        from collections import defaultdict
        
        # Analyze bigrams (2-move sequences)
        bigram_counts = defaultdict(int)
        for i in range(len(recent_moves) - 1):
            bigram = (recent_moves[i], recent_moves[i + 1])
            bigram_counts[bigram] += 1
        
        if bigram_counts:
            most_common_bigram = max(bigram_counts.items(), key=lambda x: x[1])
            if most_common_bigram[1] >= 2:  # Appears at least twice
                sequences['bigrams'] = {
                    'pattern': most_common_bigram[0],
                    'frequency': most_common_bigram[1],
                    'total_bigrams': len(recent_moves) - 1
                }
        
        # Analyze trigrams (3-move sequences) if enough data
        if len(recent_moves) >= 6:
            trigram_counts = defaultdict(int)
            for i in range(len(recent_moves) - 2):
                trigram = (recent_moves[i], recent_moves[i + 1], recent_moves[i + 2])
                trigram_counts[trigram] += 1
            
            if trigram_counts:
                most_common_trigram = max(trigram_counts.items(), key=lambda x: x[1])
                if most_common_trigram[1] >= 2:
                    sequences['trigrams'] = {
                        'pattern': most_common_trigram[0],
                        'frequency': most_common_trigram[1],
                        'total_trigrams': len(recent_moves) - 2
                    }
        
        return sequences
    
    def _generate_all_coaching_recommendations(self, recent_analysis: Dict[str, Any],
                                            emotional_context: Dict[str, Any],
                                            strategic_analysis: Dict[str, Any],
                                            performance_context: Dict[str, Any],
                                            metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate comprehensive coaching recommendations with priorities"""
        
        recommendations = []
        
        # Recent pattern-based recommendations (HIGH PRIORITY)
        pattern_tips = self._generate_pattern_based_tips(recent_analysis)
        recommendations.extend(pattern_tips)
        
        # Strategic gameplay recommendations (MEDIUM PRIORITY)
        strategic_tips = self._generate_strategic_tips(strategic_analysis, metrics)
        recommendations.extend(strategic_tips)
        
        # Emotional/psychological recommendations (MEDIUM PRIORITY)
        emotional_tips = self._generate_emotional_tips(emotional_context, metrics)
        recommendations.extend(emotional_tips)
        
        # Long-term improvement recommendations (LOW PRIORITY)
        improvement_tips = self._generate_improvement_tips(performance_context, metrics)
        recommendations.extend(improvement_tips)
        
        # Ensure we always have at least one tip
        if not recommendations:
            recommendations.append({
                'tip_key': 'general_strategy',
                'reason_key': 'Providing general strategy guidance',
                'priority': 0.3,
                'specific_advice': 'Focus on maintaining unpredictability by balancing your rock, paper, and scissors usage equally.',
                'styling_hints': {'personality': 'professor'}
            })
        
        return recommendations
    
    def _generate_pattern_based_tips(self, recent_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate tips based on recent move pattern analysis"""
        tips = []
        
        pattern = recent_analysis.get('pattern_detected', 'random_like')
        predictability = recent_analysis.get('predictability', 0.0)
        counter_opportunity = recent_analysis.get('counter_opportunity')
        move_bias = recent_analysis.get('move_bias')
        
        # High predictability warning
        if predictability > 0.7:
            tips.append({
                'tip_key': 'break_predictability',
                'reason_key': f'Your moves are {predictability:.1%} predictable - AI is likely countering you',
                'priority': 0.9,
                'specific_advice': f'Pattern detected: {pattern}. Randomize your next 3-4 moves.',
                'styling_hints': {'personality': 'professor', 'urgency': 'high'}
            })
        
        # Specific counter opportunity
        if counter_opportunity:
            tips.append({
                'tip_key': 'exploit_ai_pattern',
                'reason_key': 'AI may have developed a counter-pattern to your moves',
                'priority': 0.85,
                'specific_advice': counter_opportunity,
                'styling_hints': {'personality': 'professor', 'opportunity': True}
            })
        
        # Move bias warning
        if move_bias and move_bias['bias_percentage'] > 0.6:
            tips.append({
                'tip_key': 'reduce_move_bias',
                'reason_key': move_bias['recommendation'],
                'priority': 0.8,
                'specific_advice': f"You've used {move_bias['biased_move']} {move_bias['bias_percentage']:.1%} of recent moves",
                'styling_hints': {'personality': 'professor', 'bias_move': move_bias['biased_move']}
            })
        
        return tips
    
    def _generate_strategic_tips(self, strategic_analysis: Dict[str, Any], 
                               metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate strategic gameplay tips"""
        tips = []
        
        adaptation_speed = strategic_analysis.get('adaptation_speed', 'medium')
        consistency_score = strategic_analysis.get('consistency_score', 0.5)
        human_win_rate = metrics.get('human_win_rate', 50.0)
        robot_win_rate = metrics.get('robot_win_rate', 50.0)
        
        # Win rate analysis
        if robot_win_rate > human_win_rate + 15:  # AI winning significantly
            tips.append({
                'tip_key': 'change_overall_strategy',
                'reason_key': f'AI is ahead {robot_win_rate:.1f}% vs {human_win_rate:.1f}%',
                'priority': 0.75,
                'specific_advice': 'Current strategy isn\'t working. Try completely different approach.',
                'styling_hints': {'personality': 'professor', 'losing': True}
            })
        
        # Adaptation speed issues
        if adaptation_speed == 'slow':
            tips.append({
                'tip_key': 'increase_adaptation',
                'reason_key': 'You adapt slowly to AI counter-strategies',
                'priority': 0.65,
                'specific_advice': 'Change your approach every 3-4 moves to stay unpredictable.',
                'styling_hints': {'personality': 'professor', 'adaptation': 'slow'}
            })
        
        return tips
    
    def _generate_emotional_tips(self, emotional_context: Dict[str, Any], 
                               metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate emotional/psychological tips"""
        tips = []
        
        frustration_level = emotional_context.get('frustration_level', 0.0)
        momentum_state = emotional_context.get('momentum_state', 'neutral')
        learning_trend = emotional_context.get('learning_trend', 'stable')
        
        # High frustration
        if frustration_level > 0.7:
            tips.append({
                'tip_key': 'manage_frustration',
                'reason_key': 'High frustration detected - affects decision making',
                'priority': 0.7,
                'specific_advice': 'Take a breath. Frustrated players become more predictable.',
                'styling_hints': {'personality': 'professor', 'emotional_support': True}
            })
        
        # Cold streak
        if momentum_state == 'cold':
            tips.append({
                'tip_key': 'break_cold_streak',
                'reason_key': 'You\'re in a losing streak - AI has found your pattern',
                'priority': 0.6,
                'specific_advice': 'Reset your mindset. Try the opposite of your instinct.',
                'styling_hints': {'personality': 'professor', 'momentum': 'cold'}
            })
        
        return tips
    
    def _generate_improvement_tips(self, performance_context: Dict[str, Any], 
                                 metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate long-term improvement tips"""
        tips = []
        
        performance_tier = performance_context.get('performance_tier', 'beginner')
        challenge_readiness = performance_context.get('challenge_readiness', 0.5)
        total_rounds = performance_context.get('total_rounds', 0)
        
        # Challenge readiness
        if challenge_readiness > 0.8 and performance_tier == 'intermediate':
            tips.append({
                'tip_key': 'ready_for_advanced',
                'reason_key': 'Your skills suggest readiness for harder opponents',
                'priority': 0.4,
                'specific_advice': 'Consider increasing AI difficulty for better improvement.',
                'styling_hints': {'personality': 'professor', 'progression': True}
            })
        
        # Experience-based advice
        if total_rounds > 20 and performance_tier == 'beginner':
            tips.append({
                'tip_key': 'focus_on_fundamentals',
                'reason_key': 'Build stronger foundation before advanced techniques',
                'priority': 0.35,
                'specific_advice': 'Master basic randomization before trying complex strategies.',
                'styling_hints': {'personality': 'professor', 'fundamentals': True}
            })
        
        return tips
    
    def _determine_coaching_keys(self, learning_trend: str, consistency_score: float,
                                adaptation_speed: str, challenge_readiness: float,
                                metrics: Dict[str, Any]) -> Tuple[str, str]:
        """Determine appropriate coaching tip and reason"""
        
        # Priority-based coaching selection
        
        # High consistency (predictable patterns)
        if consistency_score > 0.7:
            return "break_patterns", "High predictability detected"
        
        # Learning trend analysis
        if learning_trend == 'declining':
            return "refocus_strategy", "Performance declining"
        elif learning_trend == 'improving':
            return "maintain_momentum", "Positive learning trend"
        
        # Adaptation speed issues
        if adaptation_speed == 'slow':
            return "increase_variety", "Slow adaptation detected"
        elif adaptation_speed == 'fast':
            return "stabilize_approach", "Too much variation"
        
        # Challenge readiness
        if challenge_readiness > 0.8:
            return "level_up_challenge", "Ready for harder difficulty"
        elif challenge_readiness < 0.3:
            return "build_fundamentals", "Focus on basics"
        
        # Default strategic advice
        recent_win_rate = metrics.get('recent_win_rate', 50.0)
        if recent_win_rate < 30:
            return "defensive_strategy", "Recent poor performance"
        elif recent_win_rate > 70:
            return "maintain_pressure", "Recent strong performance"
        else:
            return "general_strategy", "Balanced performance"


class SBCBackend:
    """
    Main SBC Backend class that coordinates rule-based selection and LLM styling.
    """
    
    def __init__(self, config: Optional[SBCConfig] = None):
        if config is None:
            config = SBCConfig()
        
        self.config = config
        self.model_adapter = ModelAdapter(config)
        self.rule_selector = RuleBasedSelector()
        self.logger = logging.getLogger(__name__ + ".SBCBackend")
        
        # Initialize Flask app
        self.app = Flask(__name__)
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup Flask routes for SBC endpoints"""
        
        @self.app.route('/sbc/banter', methods=['POST'])
        def generate_banter():
            """Generate opponent banter based on game context"""
            try:
                data = request.get_json()
                if not data or 'context' not in data:
                    return jsonify({'error': 'Missing context in request'}), 400
                
                context = data['context']
                
                # Get rule-based seed
                seed_data = self.rule_selector.select_banter_seed(context)
                
                # Generate stylized response with comprehensive prompt
                prompt = self._create_banter_prompt(seed_data, context)
                personality = seed_data['styling_hints'].get('personality', 'neutral')
                
                banter_text = self.model_adapter.generate_response(prompt, personality)
                
                response = {
                    'banter': banter_text,
                    'source': {
                        'engine': self.model_adapter.engine.value,
                        'banter_key': seed_data['banter_key'],
                        'context_reason': seed_data['context_reason'],
                        'personality': personality
                    },
                    'cache_hit': False,  # TODO: Implement proper cache hit detection
                    'timestamp': time.time()
                }
                
                if self.config.debug_mode:
                    response['debug'] = {
                        'seed_data': seed_data,
                        'prompt': prompt
                    }
                
                return jsonify(response)
                
            except Exception as e:
                self.logger.error(f"Error generating banter: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/sbc/coach', methods=['POST'])
        def generate_coaching():
            """Generate coaching advice based on game context"""
            try:
                data = request.get_json()
                if not data or 'context' not in data:
                    return jsonify({'error': 'Missing context in request'}), 400
                
                context = data['context']
                tip_count = data.get('tip_count', 3)  # Default 3 tips
                
                # Get rule-based seeds for multiple tips
                seed_data_list = self.rule_selector.select_coaching_seeds(context, tip_count)
                
                # Generate stylized responses for each tip
                tips = []
                for seed_data in seed_data_list:
                    # Create comprehensive prompt with specific context
                    prompt = self._create_coaching_prompt(seed_data, context)
                    
                    # Always use professor personality for coaching
                    tip_text = self.model_adapter.generate_response(prompt, 'professor')
                    
                    tips.append({
                        'tip': tip_text,
                        'reason': seed_data['reason_key'],
                        'priority': seed_data['priority'],
                        'specific_advice': seed_data.get('specific_advice', ''),
                        'tip_key': seed_data['tip_key']
                    })
                
                response = {
                    'tips': tips,
                    'total_tips': len(tips),
                    'source': {
                        'engine': self.model_adapter.engine.value,
                        'personality': 'professor',
                        'analysis_depth': 'comprehensive'
                    },
                    'cache_hit': False,
                    'timestamp': time.time()
                }
                
                if self.config.debug_mode:
                    response['debug'] = {
                        'seed_data_list': seed_data_list,
                        'prompts': [self._create_coaching_prompt(seed, context) for seed in seed_data_list]
                    }
                
                return jsonify(response)
                
            except Exception as e:
                self.logger.error(f"Error generating coaching: {e}")
                return jsonify({'error': str(e)}), 500
    
    def _create_banter_prompt(self, seed_data: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Create comprehensive prompt for banter generation with personality context"""
        
        game_status = context.get('game_status', {})
        metrics = game_status.get('metrics', {})
        opponent_info = context.get('opponent_info', {})
        
        # Get personality and context details
        personality = seed_data['styling_hints'].get('personality', 'neutral')
        banter_key = seed_data['banter_key']
        context_reason = seed_data['context_reason']
        
        # Get specific game context
        round_num = game_status.get('round_number', 0)
        human_win_rate = metrics.get('human_win_rate', 50.0)
        robot_win_rate = metrics.get('robot_win_rate', 50.0)
        
        # Get recent moves from full_game_snapshot
        full_snapshot = metrics.get('full_game_snapshot', {})
        recent_moves = full_snapshot.get('human_moves', [])[-3:] if full_snapshot.get('human_moves') else []
        
        # Build personality-aware prompt
        personality_context = self._get_personality_context(personality)
        
        prompt = f"""You are a Rock-Paper-Scissors AI opponent with the {personality} personality. {personality_context}

Current game situation:
- Round {round_num}
- Score: Human {human_win_rate:.1f}% vs AI {robot_win_rate:.1f}%
- Human's recent moves: {recent_moves}
- Situation: {context_reason}
- Banter type: {banter_key}

Generate a single line of banter that:
1. Reflects your {personality} personality completely
2. Responds to the current game situation
3. Is competitive but not offensive
4. Shows awareness of the human's recent moves
5. Is exactly one sentence

Your {personality} banter:"""
        
        return prompt
    
    def _create_coaching_prompt(self, seed_data: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Create comprehensive prompt for coaching tip generation"""
        
        game_status = context.get('game_status', {})
        metrics = game_status.get('metrics', {})
        
        # Get tip details
        tip_key = seed_data['tip_key']
        reason_key = seed_data['reason_key']
        specific_advice = seed_data.get('specific_advice', '')
        priority = seed_data.get('priority', 0.5)
        
        # Get game context
        round_num = game_status.get('round_number', 0)
        
        # Get moves from full_game_snapshot instead of legacy human_move_history
        full_snapshot = metrics.get('full_game_snapshot', {})
        human_moves = full_snapshot.get('human_moves', [])
        recent_moves = human_moves[-5:] if human_moves else []
        human_win_rate = metrics.get('human_win_rate', 50.0)
        robot_win_rate = metrics.get('robot_win_rate', 50.0)
        
        # Get SBC metrics for detailed analysis
        sbc_metrics = metrics.get('sbc_metrics', {})
        emotional_context = sbc_metrics.get('emotional_context', {})
        strategic_analysis = sbc_metrics.get('strategic_analysis', {})
        
        prompt = f"""You are a Professor - analytical, methodical, and insightful Rock-Paper-Scissors coach. You help players improve through data-driven analysis and strategic thinking.

Current game analysis:
- Round {round_num}
- Score: Human {human_win_rate:.1f}% vs AI {robot_win_rate:.1f}%
- Recent moves: {recent_moves}
- Momentum: {emotional_context.get('momentum_state', 'neutral')}
- Learning trend: {emotional_context.get('learning_trend', 'stable')}
- Consistency score: {strategic_analysis.get('consistency_score', 0.5):.2f}

Coaching focus: {tip_key}
Reason: {reason_key}
Specific guidance: {specific_advice}
Priority level: {priority:.1f}/1.0

Generate a coaching tip that:
1. Reflects Professor personality (analytical, data-driven, insightful)
2. Addresses the specific issue identified ({tip_key})
3. References the human's recent moves when relevant
4. Provides actionable strategic advice
5. Explains the reasoning behind the advice
6. Is encouraging but realistic
7. Is 1-2 sentences maximum

Your analytical coaching tip:"""
        
        return prompt
    
    def _get_personality_context(self, personality: str) -> str:
        """Get personality context for better prompt engineering"""
        
        personality_contexts = {
            'berserker': "You are extremely aggressive, ruthless, and dominance-focused. You taunt with intensity and show no mercy. You see every move as a battle and want to crush your opponent's spirit.",
            
            'guardian': "You are defensive, methodical, and patient. You prefer calculated taunts and strategic psychological pressure. You value steady progress over flashy victories.",
            
            'chameleon': "You are highly adaptive and observant. Your taunts reflect your ability to read and mirror your opponent's style. You're flexible and unpredictable in your psychological approach.",
            
            'professor': "You are analytical, data-driven, and fascinated by patterns. Your taunts include references to statistics, patterns, and psychological analysis. You're intellectually superior but not arrogant.",
            
            'wildcard': "You are completely unpredictable, chaotic, and love confusion. Your taunts are erratic, surprising, and designed to throw your opponent off-balance mentally.",
            
            'mirror': "You are reflective and learning-focused. Your taunts show that you're studying and adapting to your opponent's style. You're patient and observational.",
            
            'neutral': "You are balanced, fair, and sportsmanlike. Your taunts are competitive but respectful, focused on the game rather than personal attacks."
        }
        
        return personality_contexts.get(personality, personality_contexts['neutral'])
        
        @self.app.route('/sbc/status', methods=['GET'])
        def get_status():
            """Get SBC backend status"""
            return jsonify({
                'status': 'active',
                'engine': self.model_adapter.engine.value,
                'config': {
                    'cache_enabled': self.config.cache_enabled,
                    'debug_mode': self.config.debug_mode
                },
                'cache_size': len(self.model_adapter.cache) if self.model_adapter.cache else 0
            })
    
    def run(self, host='127.0.0.1', port=5001, debug=True):
        """Run the SBC backend server"""
        self.logger.info(f"Starting SBC Backend on {host}:{port}")
        self.logger.info(f"Using engine: {self.model_adapter.engine.value}")
        self.app.run(host=host, port=port, debug=debug)


# Global SBC backend instance
sbc_config = SBCConfig(
    engine=ModelEngine.OLLAMA,  # Try Ollama first, fallback to template
    debug_mode=True
)
sbc_backend = SBCBackend(sbc_config)


def get_sbc_backend():
    """Get the global SBC backend instance"""
    return sbc_backend


if __name__ == '__main__':
    # Run as standalone server for testing
    sbc_backend.run()