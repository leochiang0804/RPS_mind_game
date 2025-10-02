"""
LangChain-based AI Coach Implementation
Provides LLM-powered coaching using LangChain framework with optimized prompts
"""

import os
import json
import time
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# LangChain imports with graceful fallback
try:
    from langchain.llms.base import LLM
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.schema import BaseOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("ℹ️ LangChain not installed. AI coaching will use mock responses.")
    LANGCHAIN_AVAILABLE = False
    # Mock classes for fallback
    class LLM: pass

# Trained model import with graceful fallback
try:
    from trained_coach_wrapper import TrainedCoachWrapper
    TRAINED_MODEL_AVAILABLE = True
except ImportError:
    print("ℹ️ Trained model not available.")
    TRAINED_MODEL_AVAILABLE = False
    TrainedCoachWrapper = None
    class PromptTemplate: pass
    class LLMChain: pass
    class ConversationBufferWindowMemory: pass
    class BaseOutputParser: pass


@dataclass
class CoachingResponse:
    """Structured coaching response"""
    tips: List[str]
    insights: Dict[str, Any]
    educational_content: Dict[str, Any]
    behavioral_analysis: Dict[str, Any]
    confidence_level: float
    response_type: str


class CoachingOutputParser(BaseOutputParser):
    """Parse LLM output into structured coaching response"""
    
    def parse(self, text: str) -> CoachingResponse:
        """Parse LLM text output into structured format"""
        try:
            # Try to parse as JSON first
            if text.strip().startswith('{'):
                data = json.loads(text)
                return CoachingResponse(**data)
        except:
            pass
        
        # Fallback: parse unstructured text (enhanced for natural language)
        return self._parse_unstructured_text(text)
    
    def _parse_unstructured_text(self, text: str) -> CoachingResponse:
        """Enhanced parsing for natural language responses from real LLMs"""
        if not text or len(text.strip()) < 10:
            return self._create_fallback_response()
        
        # Clean and prepare text
        text = text.strip()
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Extract coaching tips using natural language processing
        tips = self._extract_tips_from_text(text)
        insights = self._extract_insights_from_text(text)
        educational_content = self._extract_educational_content(text)
        behavioral_analysis = self._extract_behavioral_analysis(text)
        
        return CoachingResponse(
            tips=tips,
            insights=insights,
            educational_content=educational_content,
            behavioral_analysis=behavioral_analysis,
            confidence_level=0.8,  # Higher confidence for natural language
            response_type='natural_language_parsed'
        )
    
    def _extract_tips_from_text(self, text: str) -> List[str]:
        """Extract actionable tips from natural language text"""
        tips = []
        
        # Look for common tip patterns in natural language
        tip_patterns = [
            r"[Tt]ry\s+([^.!?]+[.!?])",  # "Try doing X."
            r"[Ss]uggest(?:ion)?[^\w]*([^.!?]+[.!?])",  # "I suggest X."
            r"[Rr]ecommend[^\w]*([^.!?]+[.!?])",  # "I recommend X."
            r"[Ss]hould\s+([^.!?]+[.!?])",  # "You should X."
            r"[Cc]onsider\s+([^.!?]+[.!?])",  # "Consider X."
            r"[Ff]ocus\s+on\s+([^.!?]+[.!?])",  # "Focus on X."
            r"[Aa]void\s+([^.!?]+[.!?])",  # "Avoid X."
            r"[Ii]mprove\s+([^.!?]+[.!?])",  # "Improve X."
        ]
        
        import re
        for pattern in tip_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                clean_tip = match.strip()
                if len(clean_tip) > 10 and clean_tip not in tips:
                    tips.append(clean_tip)
        
        # If no tips found with patterns, extract from numbered/bulleted lists
        if not tips:
            tips = self._extract_from_lists(text)
        
        # If still no tips, extract from sentences containing key action words
        if not tips:
            tips = self._extract_action_sentences(text)
        
        # Limit to top 3 most relevant tips
        return tips[:3] if tips else ["Keep practicing and stay unpredictable!"]
    
    def _extract_from_lists(self, text: str) -> List[str]:
        """Extract tips from numbered or bulleted lists"""
        tips = []
        lines = text.split('\n')
        
        import re
        for line in lines:
            line = line.strip()
            # Match numbered lists (1., 2., etc.) or bulleted lists (-, *, etc.)
            list_match = re.match(r'^(?:\d+\.|\*|\-|\•)\s*(.+)', line)
            if list_match:
                tip = list_match.group(1).strip()
                if len(tip) > 10:
                    tips.append(tip)
        
        return tips
    
    def _extract_action_sentences(self, text: str) -> List[str]:
        """Extract sentences containing actionable advice"""
        sentences = re.split(r'[.!?]+', text)
        tips = []
        
        action_keywords = [
            'try', 'avoid', 'focus', 'practice', 'improve', 'change', 'adjust',
            'increase', 'decrease', 'vary', 'mix', 'random', 'pattern', 'strategy'
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 15:  # Reasonable sentence length
                # Check if sentence contains action keywords
                if any(keyword in sentence.lower() for keyword in action_keywords):
                    tips.append(sentence + '.')
        
        return tips
    
    def _extract_insights_from_text(self, text: str) -> Dict[str, Any]:
        """Extract strategic insights from natural language"""
        insights = {}
        
        # Look for performance mentions
        import re
        win_rate_match = re.search(r'(\d+)%\s*win\s*rate', text.lower())
        if win_rate_match:
            insights['performance_assessment'] = f"Win rate: {win_rate_match.group(1)}%"
        
        # Look for pattern analysis
        pattern_keywords = ['pattern', 'predictable', 'random', 'entropy', 'sequence']
        pattern_mentions = [word for word in pattern_keywords if word in text.lower()]
        if pattern_mentions:
            insights['pattern_analysis'] = f"Pattern analysis discussed: {', '.join(pattern_mentions)}"
        
        # Look for strategic assessment
        strategy_keywords = ['strategy', 'ai', 'robot', 'frequency', 'adaptation']
        strategy_mentions = [word for word in strategy_keywords if word in text.lower()]
        if strategy_mentions:
            insights['strategic_situation'] = f"Strategic elements: {', '.join(strategy_mentions)}"
        
        # If no specific insights found, use general assessment
        if not insights:
            insights['general_assessment'] = "Comprehensive analysis of gameplay patterns and strategic opportunities"
        
        return insights
    
    def _extract_educational_content(self, text: str) -> Dict[str, Any]:
        """Extract educational information from response"""
        educational = {}
        
        # Look for theory mentions
        theory_keywords = ['nash', 'equilibrium', 'game theory', 'information theory', 'entropy']
        theory_mentions = [word for word in theory_keywords if word.lower() in text.lower()]
        
        if theory_mentions:
            educational['theory_connection'] = f"Concepts mentioned: {', '.join(theory_mentions)}"
            educational['focus_area'] = 'Game Theory Application'
        else:
            educational['focus_area'] = 'Strategic Improvement'
        
        # Extract learning points from the text
        learning_sentences = []
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if any(word in sentence.lower() for word in ['learn', 'understand', 'remember', 'important']):
                if len(sentence) > 20:
                    learning_sentences.append(sentence)
        
        if learning_sentences:
            educational['learning_point'] = learning_sentences[0]  # Take the first learning point
        else:
            educational['learning_point'] = "Focus on developing unpredictable but strategic gameplay"
        
        return educational
    
    def _extract_behavioral_analysis(self, text: str) -> Dict[str, Any]:
        """Extract behavioral insights from response"""
        behavioral = {}
        
        # Look for decision-making style mentions
        decision_keywords = ['decision', 'thinking', 'approach', 'style', 'cognitive']
        if any(word in text.lower() for word in decision_keywords):
            behavioral['decision_style'] = "Decision-making patterns identified in gameplay"
        else:
            behavioral['decision_style'] = "Strategic approach under analysis"
        
        # Look for improvement areas
        improvement_keywords = ['improve', 'better', 'enhance', 'develop', 'work on']
        improvement_sentences = []
        
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            if any(word in sentence.lower() for word in improvement_keywords):
                improvement_sentences.append(sentence.strip())
        
        if improvement_sentences:
            behavioral['improvement_areas'] = improvement_sentences[0]
        else:
            behavioral['improvement_areas'] = "Continue developing strategic adaptability and pattern awareness"
        
        return behavioral
    
    def _create_fallback_response(self) -> CoachingResponse:
        """Create a fallback response when parsing fails"""
        return CoachingResponse(
            tips=["Keep practicing!", "Stay unpredictable", "Adapt your strategy"],
            insights={'status': 'Parsing failed, using fallback response'},
            educational_content={'focus_area': 'General improvement'},
            behavioral_analysis={'style': 'Unable to analyze'},
            confidence_level=0.3,
            response_type='fallback'
        )


class BasicCoach:
    """Basic coaching algorithm that uses fewer metrics for more meaningful tips"""
    
    def __init__(self):
        self.coaching_style = 'easy'  # Default to easy style
    
    def set_coaching_style(self, style: str):
        """Set coaching style for basic coaching"""
        self.coaching_style = style
    
    def __call__(self, prompt: str) -> str:
        """Generate basic coaching response using simple rules"""
        
        if 'comprehensive' in prompt.lower():
            return self._generate_basic_comprehensive_analysis(prompt)
        else:
            return self._generate_basic_realtime_advice(prompt)
    
    def _extract_key_metrics(self, prompt: str) -> dict:
        """Extract only essential metrics for basic coaching"""
        metrics = {}
        
        import re
        
        # Extract current round
        round_match = re.search(r'Round: (\d+)', prompt)
        metrics['round'] = int(round_match.group(1)) if round_match else 0
        
        # Extract win rate
        win_rate_match = re.search(r'Win Rate: ([\d.]+)%', prompt)
        metrics['win_rate'] = float(win_rate_match.group(1)) / 100 if win_rate_match else 0.33
        
        # Extract predictability
        pred_match = re.search(r'Predictability: ([\d.]+)', prompt)
        metrics['predictability'] = float(pred_match.group(1)) if pred_match else 0.5
        
        # Extract AI strategy
        strategy_match = re.search(r'AI Strategy: (\w+)', prompt)
        metrics['ai_strategy'] = strategy_match.group(1) if strategy_match else 'unknown'
        
        # Extract recent moves
        recent_moves_match = re.search(r'Recent Moves: \[(.*?)\]', prompt)
        if recent_moves_match:
            moves_str = recent_moves_match.group(1)
            metrics['recent_moves'] = [move.strip().strip("'\"") for move in moves_str.split(',') if move.strip()]
        else:
            metrics['recent_moves'] = []
        
        return metrics
    
    def _generate_basic_realtime_advice(self, prompt: str) -> str:
        """Generate basic real-time coaching advice using simple rules"""
        
        metrics = self._extract_key_metrics(prompt)
        
        # Generate exactly 3 simple, actionable tips
        tips = []
        
        # Tip 1: Predictability advice
        if metrics['predictability'] > 0.7:
            tips.append("🎯 You're being too predictable! Mix up your moves more randomly.")
        elif metrics['predictability'] < 0.3:
            tips.append("✨ Great randomness! Keep the AI guessing.")
        else:
            tips.append("👍 Good balance - try to stay unpredictable.")
        
        # Tip 2: Pattern advice based on recent moves
        recent_moves = metrics.get('recent_moves', [])
        if len(recent_moves) >= 2:
            if recent_moves[-1] == recent_moves[-2]:
                tips.append("⚠️ Avoid repeating the same move - AI will exploit this!")
            elif len(set(recent_moves[-3:])) == 1 and len(recent_moves) >= 3:
                tips.append("🔄 Break your repetition pattern immediately!")
            else:
                tips.append("🎲 Keep varying your moves to stay ahead.")
        else:
            tips.append("🎲 Focus on unpredictable move selection.")
        
        # Tip 3: AI strategy counter
        ai_strategy = metrics.get('ai_strategy', 'unknown')
        if ai_strategy == 'frequency':
            tips.append("📊 AI tracks your habits - change your style suddenly!")
        elif ai_strategy == 'cycler':
            tips.append("🔄 AI uses cycles - break its pattern with randomness!")
        elif ai_strategy == 'anti_cycle':
            tips.append("↩️ AI counters cycles - try short sequences instead!")
        else:
            tips.append("🤖 Observe the AI's pattern and adapt your strategy!")
        
        # Return as JSON
        import json
        return json.dumps({
            'tips': tips[:3],  # Ensure exactly 3 tips
            'insights': {
                'pattern_analysis': f"Predictability: {metrics['predictability']:.2f}",
                'strategic_situation': f"Round {metrics['round']}: {ai_strategy} AI detected"
            },
            'educational_content': {
                'focus_area': 'Basic Strategy',
                'learning_point': 'Stay unpredictable and adapt to AI patterns'
            },
            'behavioral_analysis': {
                'decision_style': 'Improving',
                'improvement_areas': 'Pattern recognition and adaptation'
            },
            'confidence_level': 0.85,
            'response_type': f'basic_{self.coaching_style}'
        })
    
    def _generate_basic_comprehensive_analysis(self, prompt: str) -> str:
        """Generate basic comprehensive analysis - still only 3 tips, no natural language"""
        
        # For Basic coaching, comprehensive analysis is just the same 3 tips
        # No natural language output, no complex analysis
        return self._generate_basic_realtime_advice(prompt)
        
        # Generate contextual tips based on actual metrics and style
        tips = []
        
        if self.coaching_style == 'easy':
            # Easy-to-understand style - practical advice
            if metrics['entropy'] < 1.0:
                tips.append("🎯 Mix up your moves more! You're being too predictable.")
            elif metrics['entropy'] > 1.4:
                tips.append("✨ Great randomness! Keep mixing your moves unpredictably.")
            else:
                tips.append("👍 Good move variety - try to stay this unpredictable.")
            
            if metrics['predictability'] > 0.7:
                tips.append("⚠️ The AI is catching your pattern - try a completely different approach.")
            elif metrics['predictability'] < 0.3:
                tips.append("🔥 Excellent! You're keeping the AI guessing.")
            else:
                tips.append("💡 You're doing well - maybe vary your timing between moves.")
            
            if metrics['nash_distance'] > 0.4:
                tips.append("⚖️ Try to use rock, paper, and scissors more equally.")
            elif metrics['nash_distance'] < 0.2:
                tips.append("🎯 Perfect balance! You're playing optimally.")
            else:
                tips.append("👌 Good move distribution - keep this balance.")
            
            # Add specific tactical advice
            if metrics['robot_strategy'] == 'cycler':
                tips.append("🤖 The AI is using cycles - break its pattern by being random.")
            elif metrics['robot_strategy'] == 'frequency':
                tips.append("📊 The AI copies your habits - change your style suddenly.")
            elif metrics['robot_strategy'] == 'anti_cycle':
                tips.append("🔄 The AI is anti-cycling - try short sequences instead.")
            
        else:
            # Scientific style - technical analysis
            if metrics['entropy'] < 1.0:
                tips.append(f"📈 Entropy: {metrics['entropy']:.3f} - Increase randomness to approach maximum entropy (1.585)")
            elif metrics['entropy'] > 1.4:
                tips.append(f"🎲 Excellent entropy: {metrics['entropy']:.3f} - Maintaining near-optimal randomization")
            else:
                tips.append(f"📊 Entropy: {metrics['entropy']:.3f} - Good baseline, room for 15% improvement")
            
            if metrics['predictability'] > 0.7:
                tips.append(f"⚠️ High predictability: {metrics['predictability']:.3f} - Implement counter-exploitation strategy")
            elif metrics['predictability'] < 0.3:
                tips.append(f"🛡️ Low predictability: {metrics['predictability']:.3f} - Excellent defensive play")
            else:
                tips.append(f"📋 Predictability: {metrics['predictability']:.3f} - Moderate pattern detection risk")
            
            if metrics['nash_distance'] > 0.4:
                tips.append(f"⚖️ Nash distance: {metrics['nash_distance']:.3f} - Move toward uniform distribution (1/3 each)")
            elif metrics['nash_distance'] < 0.2:
                tips.append(f"🎯 Nash distance: {metrics['nash_distance']:.3f} - Near-optimal equilibrium play")
            else:
                tips.append(f"📐 Nash distance: {metrics['nash_distance']:.3f} - Acceptable deviation from optimal")
            
            # Technical AI analysis
            if metrics['robot_strategy'] == 'cycler':
                tips.append(f"🤖 AI Strategy: Pattern-cycle exploitation detected - Apply anti-pattern methodology")
            elif metrics['robot_strategy'] == 'frequency':
                tips.append(f"📊 AI Strategy: Frequency analysis mode - Implement move distribution randomization")
            elif metrics['robot_strategy'] == 'anti_cycle':
                tips.append(f"🔄 AI Strategy: Anti-cycle prediction - Use short-sequence disruption tactics")
            
            # Add complexity analysis
            if metrics['complexity'] > 0.6:
                tips.append(f"🧠 Decision complexity: {metrics['complexity']:.3f} - High cognitive load detected")
            elif metrics['complexity'] < 0.3:
                tips.append(f"🧠 Decision complexity: {metrics['complexity']:.3f} - Consider deeper strategic thinking")
        
        # Limit to 3 most relevant tips
        tips = tips[:3]
        
        # Ensure valid JSON response
        response_json = json.dumps({
            'tips': tips,
            'insights': {
                'pattern_analysis': f"Entropy: {metrics['entropy']:.3f}, Predictability: {metrics['predictability']:.3f}",
                'strategic_situation': f"Round {metrics['round']}: {metrics['robot_strategy']} AI strategy detected",
                'performance_trend': f"Win rate: {metrics['win_rate']:.1%} - {'Above' if metrics['win_rate'] > 0.4 else 'Below'} baseline"
            },
            'educational_content': {
                'focus_area': 'Entropy Optimization' if metrics['entropy'] < 1.2 else 'Pattern Disruption',
                'learning_point': f"Your entropy ({metrics['entropy']:.3f}) shows {'good' if metrics['entropy'] > 1.2 else 'developing'} randomization skills",
                'theory_connection': 'Information theory suggests maximum unpredictability through balanced randomness'
            },
            'behavioral_analysis': {
                'decision_style': f"Complexity: {metrics['complexity']:.3f} - {'Strategic' if metrics['complexity'] > 0.5 else 'Intuitive'} approach",
                'adaptation_level': f"Consistency: {metrics['consistency']:.3f} - {'Stable' if metrics['consistency'] > 0.6 else 'Adaptive'} strategy",
                'improvement_areas': 'Focus on entropy optimization and Nash equilibrium approximation'
            },
            'confidence_level': 0.95,
            'response_type': f'enhanced_mock_{self.coaching_style}'
        })
        
        return response_json
    

class LangChainAICoach:
    """
    LangChain-based AI coach with optimized prompts for RPS coaching
    """
    
    def __init__(self):
        # Initialize coaching style first
        self.coaching_style = 'easy'  # Default to easy style
        
        self.llm = self._initialize_llm()
        self.trained_coach = None  # Will be initialized if trained model is selected
        self.output_parser = CoachingOutputParser()
        self.memory = self._initialize_memory()
        
        # Initialize prompt templates
        self.real_time_prompt = self._create_real_time_prompt()
        self.comprehensive_prompt = self._create_comprehensive_prompt()
        
        # Initialize chains
        self.real_time_chain = self._create_real_time_chain()
        self.comprehensive_chain = self._create_comprehensive_chain()
        
        print("✅ LangChain AI Coach initialized successfully")
    
    def _initialize_llm(self) -> Any:
        """Initialize LLM with fallback options - prioritizing fast MockLLM"""
        
        # Use BasicCoach as primary choice for speed and reliability
        print("🚀 Using enhanced BasicCoach for fast, intelligent coaching")
        mock_llm = BasicCoach()
        mock_llm.set_coaching_style(self.coaching_style)
        return mock_llm
        
        # Optional: Try real LLM as backup (commented out for speed)
        # if not LANGCHAIN_AVAILABLE:
        #     print("ℹ️ Using mock LLM for testing")
        #     return MockLLM()
        
        # try:
        #     # Option 1: Try local model (Ollama, LlamaCpp, etc.)
        #     return self._try_local_model()
        # except Exception as e:
        #     print(f"ℹ️ Local model not available: {e}")
            
        # try:
        #     # Option 2: Try OpenAI API (if key available)
        #     return self._try_openai_model()
        # except Exception as e:
        #     print(f"ℹ️ OpenAI model not available: {e}")
        
        # # Fallback: Mock LLM
        # print("ℹ️ Using mock LLM as fallback")
        # mock_llm = MockLLM()
        # mock_llm.set_coaching_style(self.coaching_style)
        # return mock_llm
    
    def _try_local_model(self):
        """Try to initialize local model"""
        try:
            from langchain_ollama import OllamaLLM
        except ImportError:
            try:
                from langchain.llms import Ollama as OllamaLLM
            except ImportError:
                print("⚠️ Ollama not available - LangChain integration missing")
                raise Exception("Ollama LangChain integration not found")
        
        # Try Ollama models in order of preference (smaller = faster)
        models_to_try = ["llama3.2:3b", "llama3.2:1b", "phi3:mini", "qwen2:0.5b"]
        
        for model in models_to_try:
            try:
                llm = OllamaLLM(model=model, temperature=0.7)
                # Test the model with a simple prompt
                test_response = llm.invoke("Hello")
                if test_response and len(test_response.strip()) > 0:
                    print(f"✅ Connected to Ollama {model}")
                    return llm
            except Exception as e:
                print(f"⚠️ {model} not available: {e}")
        
        raise Exception("No Ollama models found. Please install with: ollama pull llama3.2:3b")
    
    def _try_openai_model(self):
        """Try to initialize OpenAI model"""
        from langchain.llms import OpenAI
        
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            return OpenAI(temperature=0.7, max_tokens=512)
        
        raise Exception("OpenAI API key not found")
    
    def _initialize_memory(self):
        """Initialize conversation memory"""
        if LANGCHAIN_AVAILABLE:
            try:
                # Use new memory API if available
                from langchain.memory import ConversationBufferWindowMemory
                return ConversationBufferWindowMemory(k=5, return_messages=True)
            except Exception:
                # Fallback: no memory for now
                return None
        return None
    
    def _create_real_time_prompt(self) -> PromptTemplate:
        """Create optimized prompt template for real-time coaching"""
        
        # Create different templates for different LLM types
        mock_template = """You are an expert Rock-Paper-Scissors coach providing real-time guidance.

GAME CONTEXT:
{context}

COACHING STYLE: {style}

Provide coaching advice in JSON format:
{{
    "tips": ["specific actionable tip 1", "strategic insight tip 2", "educational tip 3"],
    "insights": {{
        "pattern_analysis": "What patterns are detected and their significance",
        "strategic_situation": "Current strategic assessment"
    }},
    "educational_content": {{
        "focus_area": "Primary learning focus",
        "learning_point": "Key educational insight"
    }},
    "behavioral_analysis": {{
        "decision_style": "Assessment of decision-making approach",
        "improvement_areas": "Specific areas for development"
    }},
    "confidence_level": 0.85,
    "response_type": "real_time_{style}"
}}

Keep advice {style_instruction} and immediately actionable."""

        real_template = """You are an expert Rock-Paper-Scissors coach. A player needs your help with their game strategy.

CURRENT SITUATION:
{context}

COACHING APPROACH: Provide {style_instruction} guidance that's helpful and encouraging.

Based on the game data above, please provide personalized coaching advice. Your response MUST be structured using the following clear section headers (in this order):

---
**IMMEDIATE TIPS**
List 2-3 specific things the player can do right now to improve.

**PATTERN INSIGHTS**
Describe what you notice about their playing style, including any habits or tendencies.

**STRATEGIC ADVICE**
Explain how they can adapt to beat the AI's current strategy.

**ENCOURAGEMENT**
Give positive reinforcement about what the player is doing well.
---

Please use these section headers verbatim and keep your advice concise, actionable, and easy to follow. Respond in a natural, conversational way as if you're coaching them in person. Be specific about the numbers and patterns you see, but explain them in an accessible way."""

        # Store both templates for dynamic selection
        self.mock_real_time_template = mock_template
        self.real_real_time_template = real_template

        if LANGCHAIN_AVAILABLE:
            return PromptTemplate(
                input_variables=["context", "style", "style_instruction"],
                template=mock_template  # Default to mock template for PromptTemplate object
            )
        return mock_template
    
    def _create_comprehensive_prompt(self) -> PromptTemplate:
        """Create prompt template for comprehensive analysis"""
        
        # Create different templates for different LLM types
        mock_template = """You are conducting a comprehensive post-game analysis of a Rock-Paper-Scissors session.

SESSION DATA:
{session_data}

COACHING STYLE: {style}

Provide comprehensive analysis in JSON format:
{{
    "psychological_patterns": "Deep analysis of decision-making psychology",
    "strategic_evolution": "How strategy evolved throughout the session",
    "decision_analysis": "Decision-making patterns and insights",
    "information_theory_insights": "Insights from entropy and information metrics",
    "game_theory_assessment": "Nash equilibrium and strategic analysis",
    "cognitive_load_analysis": "Analysis of decision complexity patterns",
    "learning_recommendations": "Specific improvements based on analysis",
    "fascinating_discoveries": "Interesting behavioral insights",
    "educational_summary": "Key learning points and theory connections",
    "personalized_roadmap": "Customized improvement plan"
}}

Make this analysis {style_instruction} and educationally valuable."""

        real_template = """You are an expert Rock-Paper-Scissors coach conducting a comprehensive post-game analysis for a player.

COMPLETE SESSION DATA:
{session_data}

ANALYSIS STYLE: Provide {style_instruction} insights that help the player understand their strategic development.

Your response MUST be structured using the following section headers, in this order. For each section, provide clear, concise, and actionable feedback. Write in a conversational but well-organized and professional tone.

---
**PSYCHOLOGICAL PATTERNS**
Analyze the player's decision-making style, risk tolerance, and any psychological tendencies observed during the session.

**STRATEGIC EVOLUTION**
Describe how the player's strategy developed throughout the session. Note any adaptations, shifts in approach, or learning moments.

**DECISION ANALYSIS**
Identify patterns in the player's choices. Are their moves predictable or random? Highlight any habits or exploitable tendencies.

**PERFORMANCE INSIGHTS**
Interpret key metrics (entropy, predictability, win rate, etc.) and explain what they reveal about the player's gameplay.

**LEARNING OPPORTUNITIES**
Suggest specific areas the player should focus on to improve, based on your analysis.

**FASCINATING DISCOVERIES**
Share any interesting or unexpected patterns, behaviors, or turning points you noticed.

**EDUCATIONAL SUMMARY**
Summarize the main lessons and connect them to relevant game theory or strategy principles.

**PERSONALIZED ROADMAP**
Provide a step-by-step improvement plan tailored to the player's current skill level and playing style.

---

Please use these section headers verbatim and keep your advice well-structured, easy to follow, and actionable. Avoid emojis. Respond as if you are having a thoughtful conversation with the player about their game.
"""

        # Store both templates for dynamic selection
        self.mock_comprehensive_template = mock_template
        self.real_comprehensive_template = real_template

        if LANGCHAIN_AVAILABLE:
            return PromptTemplate(
                input_variables=["session_data", "style", "style_instruction"],
                template=mock_template  # Default to mock template for PromptTemplate object
            )
        return mock_template
    
    def _create_real_time_chain(self):
        """Create chain for real-time coaching using modern LangChain API"""
        if LANGCHAIN_AVAILABLE and hasattr(self.llm, 'invoke') and not isinstance(self.llm, BasicCoach):
            # Use modern RunnableSequence for real LLM
            try:
                # Type ignore because we're handling various LLM types
                return self.real_time_prompt | self.llm  # type: ignore
            except:
                return None
        return None
    
    def _create_comprehensive_chain(self):
        """Create chain for comprehensive analysis using modern LangChain API"""
        if LANGCHAIN_AVAILABLE and hasattr(self.llm, 'invoke') and not isinstance(self.llm, BasicCoach):
            # Use modern RunnableSequence for real LLM
            try:
                # Type ignore because we're handling various LLM types
                return self.comprehensive_prompt | self.llm  # type: ignore
            except:
                return None
        return None
    
    def set_coaching_style(self, style: str) -> None:
        """Set coaching style preference"""
        if style in ['easy', 'scientific']:
            self.coaching_style = style
            # Update MockLLM style if it's being used
            if isinstance(self.llm, BasicCoach):
                self.llm.set_coaching_style(style)
    
    def get_coaching_style(self) -> str:
        """Get current coaching style"""
        return self.coaching_style
    
    def set_llm_type(self, llm_type: str) -> Dict[str, Any]:
        """
        Set the LLM type for coaching
        
        Args:
            llm_type: 'mock' for fast MockLLM, 'real' for actual LLM models, 'trained' for our trained model
            
        Returns:
            Status dictionary
        """
        if llm_type not in ['mock', 'real', 'trained']:
            return {
                'success': False,
                'error': f'Invalid LLM type: {llm_type}. Must be "mock", "real", or "trained"',
                'current_llm_type': self.get_llm_type()
            }
        
        old_type = self.get_llm_type()
        
        try:
            if llm_type == 'mock':
                # Switch to MockLLM
                print("🔄 Switching to BasicCoach for fast responses")
                mock_llm = BasicCoach()
                mock_llm.set_coaching_style(self.coaching_style)
                self.llm = mock_llm
                # Clear trained model when switching away from it
                self.trained_coach = None
                
            elif llm_type == 'real':
                # Try to switch to real LLM
                print("🔄 Attempting to switch to real LLM")
                real_llm = self._try_real_llm()
                if real_llm is not None:
                    self.llm = real_llm
                    # Clear trained model when switching away from it
                    self.trained_coach = None
                else:
                    return {
                        'success': False,
                        'error': 'Real LLM not available. Remaining on current LLM.',
                        'current_llm_type': old_type
                    }
                    
            elif llm_type == 'trained':
                # Switch to our trained model
                print("🔄 Switching to trained RPS coaching model")
                if TRAINED_MODEL_AVAILABLE and TrainedCoachWrapper is not None:
                    trained_coach = TrainedCoachWrapper()
                    trained_coach.set_coaching_style(self.coaching_style)
                    if trained_coach.is_available():
                        self.trained_coach = trained_coach
                        self.llm = trained_coach  # Use trained coach as LLM replacement
                    else:
                        return {
                            'success': False,
                            'error': 'Trained model failed to load. Remaining on current LLM.',
                            'current_llm_type': old_type
                        }
                else:
                    return {
                        'success': False,
                        'error': 'Trained model not available. Install model training dependencies.',
                        'current_llm_type': old_type
                    }
            
            # Update chains with new LLM (except for trained model which has its own interface)
            if llm_type != 'trained':
                self.real_time_chain = self._create_real_time_chain()
                self.comprehensive_chain = self._create_comprehensive_chain()
            
            return {
                'success': True,
                'old_llm_type': old_type,
                'new_llm_type': llm_type,
                'llm_class': type(self.llm).__name__
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to switch LLM type: {str(e)}',
                'current_llm_type': old_type
            }
    
    def get_llm_type(self) -> str:
        """Get current LLM type"""
        if hasattr(self, 'trained_coach') and self.trained_coach is not None:
            return 'trained'
        elif isinstance(self.llm, BasicCoach):
            return 'mock'
        else:
            return 'real'
    
    def _try_real_llm(self):
        """Try to initialize a real LLM"""
        # Try local model first
        try:
            return self._try_local_model()
        except Exception as e:
            print(f"ℹ️ Local model not available: {e}")
        
        # Try OpenAI API
        try:
            return self._try_openai_model()
        except Exception as e:
            print(f"ℹ️ OpenAI model not available: {e}")
        
        # No real LLM available
        print("ℹ️ No real LLM available")
        return None
    
    def generate_coaching_advice(self, comprehensive_metrics: Dict[str, Any], coaching_type: str = 'real_time') -> Dict[str, Any]:
        """
        Generate coaching advice using LangChain
        
        Args:
            comprehensive_metrics: Complete metrics from aggregator
            coaching_type: 'real_time' or 'comprehensive'
            
        Returns:
            Structured coaching advice
        """
        
        try:
            if coaching_type == 'real_time':
                return self._generate_real_time_advice(comprehensive_metrics)
            elif coaching_type == 'comprehensive':
                return self._generate_comprehensive_analysis(comprehensive_metrics)
            else:
                raise ValueError(f"Unknown coaching type: {coaching_type}")
                
        except Exception as e:
            print(f"⚠️ LangChain coaching failed: {e}")
            return self._generate_fallback_advice(comprehensive_metrics, coaching_type)
    
    def _identify_exploitable_patterns(self, core: Dict[str, Any], patterns: Dict[str, Any]) -> str:
        """Identify patterns that can be exploited"""
        recent_moves = core.get('recent_moves', {}).get('human_last_5', [])
        if len(recent_moves) < 3:
            return "Insufficient data for pattern analysis"
        
        # Check for repetition
        if len(set(recent_moves[-3:])) == 1:
            return f"🚨 DANGER: You've played {recent_moves[-1]} three times in a row - highly predictable!"
        
        # Check for simple alternations
        if len(recent_moves) >= 4:
            if recent_moves[-1] != recent_moves[-2] and recent_moves[-2] != recent_moves[-3] and recent_moves[-3] != recent_moves[-4]:
                return "⚠️ WARNING: Alternating pattern detected - mix up your strategy"
        
        predictability = patterns.get('predictability_score', 0.0)
        if predictability > 0.7:
            return f"🎯 EXPLOITABLE: Your predictability is {predictability:.1%} - opponent can easily counter"
        elif predictability > 0.5:
            return f"⚡ MODERATE RISK: Predictability at {predictability:.1%} - consider mixing up moves"
        else:
            return f"✅ GOOD: Low predictability at {predictability:.1%} - keep it up!"

    def _assess_predictability_risk(self, recent_moves: List[str], predictability: float) -> str:
        """Assess the risk level of current predictability"""
        if len(recent_moves) < 2:
            return "Need more moves to assess risk"
        
        move_sequence = " → ".join(recent_moves[-5:])
        
        if predictability > 0.8:
            return f"🔴 HIGH RISK: Sequence {move_sequence} is very predictable ({predictability:.1%})"
        elif predictability > 0.6:
            return f"🟡 MEDIUM RISK: Sequence {move_sequence} shows patterns ({predictability:.1%})"
        elif predictability > 0.4:
            return f"🟢 LOW RISK: Sequence {move_sequence} is reasonably unpredictable ({predictability:.1%})"
        else:
            return f"💎 EXCELLENT: Sequence {move_sequence} is highly unpredictable ({predictability:.1%})"

    def _suggest_next_move(self, core: Dict[str, Any], patterns: Dict[str, Any], ai_behavior: Dict[str, Any]) -> str:
        """Suggest the best next move based on current situation"""
        recent_moves = core.get('recent_moves', {}).get('human_last_5', [])
        robot_strategy = ai_behavior.get('ai_strategy', 'unknown')
        
        if len(recent_moves) == 0:
            return "🎲 Start with any move - no pattern established yet"
        
        last_move = recent_moves[-1]
        move_dist = patterns.get('move_distribution', {})
        
        # Find least used move
        least_used = min(move_dist.items(), key=lambda x: x[1])[0] if move_dist else 'rock'
        
        # Check for repetition
        if len(recent_moves) >= 2 and recent_moves[-1] == recent_moves[-2]:
            return f"🔄 SWITCH: You just played {last_move} twice - try {least_used} to break pattern"
        
        # Counter AI strategy
        if robot_strategy == 'frequency_based':
            return f"🎯 COUNTER: AI uses frequency analysis - play {least_used} (your least used move)"
        elif robot_strategy == 'markov_chain':
            return f"🧠 COUNTER: AI predicts based on sequences - avoid patterns, try {least_used}"
        else:
            return f"🎲 STRATEGIC: Mix it up - {least_used} is your least used move"

    def _identify_opponent_weakness(self, ai_behavior: Dict[str, Any], core: Dict[str, Any]) -> str:
        """Identify exploitable weaknesses in opponent (AI) behavior"""
        robot_strategy = ai_behavior.get('ai_strategy', 'unknown')
        ai_confidence = ai_behavior.get('confidence_history', {}).get('enhanced', [0.5])[-1] if ai_behavior.get('confidence_history', {}).get('enhanced') else 0.5
        
        win_rate = core.get('win_rates', {}).get('human', 0.0)
        
        if robot_strategy == 'frequency_based':
            return "🎯 AI WEAKNESS: Uses frequency analysis - exploit by avoiding your most common moves"
        elif robot_strategy == 'markov_chain':
            return "🧠 AI WEAKNESS: Predicts sequences - break patterns with random moves"
        elif robot_strategy == 'adaptive':
            if ai_confidence < 0.4:
                return "🤖 AI STRUGGLING: Low confidence detected - maintain pressure with unpredictable play"
            else:
                return "🤖 AI ADAPTING: High confidence - switch strategy to confuse its adaptation"
        elif win_rate > 0.6:
            return "💪 WINNING: You're exploiting AI successfully - maintain current unpredictability"
        elif win_rate < 0.4:
            return "🔄 LOSING: AI has found your patterns - drastically change your approach"
        else:
            return "⚖️ BALANCED: Even match - focus on increasing randomness"

    def _generate_pattern_disruption_advice(self, recent_moves: List[str]) -> str:
        """Generate specific advice for disrupting current patterns"""
        if len(recent_moves) < 2:
            return "Play any move to start establishing unpredictability"
        
        # Check for repetition
        if len(set(recent_moves[-3:])) == 1 and len(recent_moves) >= 3:
            last_move = recent_moves[-1]
            other_moves = [m for m in ['rock', 'paper', 'scissors'] if m != last_move]
            return f"⚡ URGENT: Break the {last_move} repetition! Try {other_moves[0]} or {other_moves[1]}"
        
        # Check for simple patterns
        if len(recent_moves) >= 4:
            pattern = recent_moves[-4:]
            if pattern == ['rock', 'paper', 'rock', 'paper']:
                return "🔄 Break the rock-paper alternation! Try scissors"
            elif pattern == ['rock', 'paper', 'scissors', 'rock']:
                return "🎯 Cycling detected! Skip your next expected move"
        
        # General advice based on move distribution
        move_counts = {move: recent_moves.count(move) for move in ['rock', 'paper', 'scissors']}
        most_used = max(move_counts.items(), key=lambda x: x[1])[0]
        least_used = min(move_counts.items(), key=lambda x: x[1])[0]
        
        if move_counts[most_used] > len(recent_moves) * 0.6:
            return f"📊 You're favoring {most_used} too much - try {least_used} more often"
        else:
            return f"✅ Good variety - consider playing {least_used} next to maintain balance"

    def _assess_psychological_state(self, streaks: Dict[str, Any], recent_results: List[str]) -> str:
        """Assess current psychological state and risks"""
        current_streak = streaks.get('current_streak', {})
        streak_type = current_streak.get('type', 'none')
        streak_length = current_streak.get('length', 0)
        
        if streak_type == 'loss' and streak_length >= 3:
            return f"😤 TILT RISK: {streak_length} losses in a row - stay calm, avoid emotional decisions"
        elif streak_type == 'win' and streak_length >= 4:
            return f"😎 CONFIDENCE HIGH: {streak_length} wins - don't get overconfident, maintain focus"
        elif streak_type == 'tie' and streak_length >= 3:
            return f"🤝 STALEMATE: {streak_length} ties - time to shake things up dramatically"
        
        # Analyze recent volatility
        if len(recent_results) >= 5:
            wins = recent_results.count('human')
            if wins <= 1:
                return "😰 STRUGGLING: Very few recent wins - take a breath and refocus strategy"
            elif wins >= 4:
                return "🔥 ON FIRE: Dominating recently - keep the pressure on!"
        
        return "😌 STABLE: Balanced emotional state - good time for strategic thinking"

    def _assess_tactical_situation(self, current_round: int, momentum: Dict[str, Any]) -> str:
        """Assess the current tactical situation"""
        momentum_direction = momentum.get('momentum_direction', 'stable')
        momentum_score = momentum.get('momentum_score', 0.0)
        
        if current_round <= 5:
            return "🌱 EARLY GAME: Establish unpredictability, avoid early patterns"
        elif current_round <= 15:
            return "⚖️ MID GAME: Key phase - adapt based on AI's discovered weaknesses"
        else:
            return "🏁 LATE GAME: Maintain what's working, avoid major strategy changes"
        
        if momentum_direction == 'positive' and momentum_score > 0.6:
            return f"📈 MOMENTUM UP: Riding high - maintain current strategy ({momentum_score:.1%} positive)"
        elif momentum_direction == 'negative' and momentum_score < -0.6:
            return f"📉 MOMENTUM DOWN: Need to turn it around ({abs(momentum_score):.1%} negative)"
        else:
            return f"➡️ MOMENTUM STABLE: Even battle - focus on execution ({momentum_score:.1%})"

    def _identify_adaptation_opportunity(self, ai_behavior: Dict[str, Any], game_phase: str) -> str:
        """Identify opportunities for strategic adaptation"""
        robot_strategy = ai_behavior.get('ai_strategy', 'unknown')
        ai_adaptation = ai_behavior.get('ai_adaptation', {})
        
        if robot_strategy == 'frequency_based':
            return "🎯 ADAPT: AI tracks your frequencies - regularly switch your most common move"
        elif robot_strategy == 'markov_chain':
            return "🧠 ADAPT: AI learns sequences - break patterns every 3-4 moves"
        elif robot_strategy == 'adaptive':
            return "🤖 ADAPT: AI changes strategy - monitor its adjustments and counter-adapt"
        elif game_phase == 'late':
            return "🔄 ADAPT: Late game - AI has learned your patterns, try completely new approach"
        else:
            return "⚡ ADAPT: Stay one step ahead - vary your randomization strategy"

    def _determine_strategic_priority(self, predictability: float, win_rate: float) -> str:
        """Determine the top strategic priority"""
        if predictability > 0.7:
            return "🎯 TOP PRIORITY: Reduce predictability - you're too easy to read"
        elif win_rate < 0.3:
            return "🔄 TOP PRIORITY: Change strategy completely - current approach isn't working"
        elif win_rate > 0.7:
            return "💎 TOP PRIORITY: Maintain unpredictability - don't let success breed complacency"
        elif predictability > 0.5:
            return "⚖️ TOP PRIORITY: Balance randomness with strategic play"
        else:
            return "🚀 TOP PRIORITY: Fine-tune execution - you're on the right track"

    # Helper methods for comprehensive analysis session data
    def _identify_most_exploited_pattern(self, core: Dict[str, Any], patterns: Dict[str, Any]) -> str:
        """Identify the most exploited pattern in the session"""
        pattern_type = patterns.get('pattern_type', 'unknown')
        predictability = patterns.get('predictability_score', 0.0)
        
        if predictability > 0.7:
            return f"High predictability in {pattern_type} patterns ({predictability:.1%})"
        elif pattern_type == 'single_move_repetition':
            return "Repeated single moves too frequently"
        elif pattern_type == 'low_variation':
            return "Limited move variety made you predictable"
        else:
            return "No major exploitable patterns detected"

    def _estimate_early_performance(self, performance: Dict[str, Any]) -> str:
        """Estimate performance in early game"""
        recent_perf = performance.get('recent_performance', {})
        if 'last_10_games' in recent_perf:
            return f"Early rounds showed developing strategy"
        return "Building initial approach"

    def _estimate_late_performance(self, performance: Dict[str, Any]) -> str:
        """Estimate performance in late game"""
        trend = performance.get('recent_performance', {}).get('trend', 'stable')
        if trend == 'improving':
            return "Strong finish with improving play"
        elif trend == 'declining':
            return "Struggled in later rounds"
        else:
            return "Maintained consistent performance"

    def _assess_adaptation_success(self, performance: Dict[str, Any], temporal: Dict[str, Any]) -> str:
        """Assess how well player adapted during session"""
        trend = performance.get('recent_performance', {}).get('trend', 'stable')
        momentum = performance.get('momentum', {}).get('momentum_direction', 'stable')
        
        if trend == 'improving' and momentum == 'positive':
            return "Successfully adapted and improved throughout session"
        elif trend == 'declining':
            return "Struggled to adapt - AI learned your patterns"
        else:
            return "Moderate adaptation - mixed results"

    def _identify_learning_patterns(self, patterns: Dict[str, Any], performance: Dict[str, Any]) -> str:
        """Identify learning and improvement patterns"""
        entropy = patterns.get('entropy_calculation', 0.0)
        trend = performance.get('recent_performance', {}).get('trend', 'stable')
        
        if entropy > 1.2 and trend == 'improving':
            return "Strong learning - improved randomness and performance"
        elif entropy > 1.2:
            return "Good randomness learning - focus on tactical execution"
        elif trend == 'improving':
            return "Tactical improvements - work on unpredictability"
        else:
            return "Learning opportunities identified for next session"

    def _identify_key_turning_point(self, performance: Dict[str, Any], core: Dict[str, Any]) -> str:
        """Identify key turning points in the session"""
        streaks = performance.get('streaks', {})
        longest_win = streaks.get('longest_win_streak', 0)
        longest_loss = streaks.get('longest_loss_streak', 0)
        total_rounds = core.get('current_round', 0)
        
        if longest_win >= 4:
            return f"Mid-session win streak of {longest_win} showed strategic breakthrough"
        elif longest_loss >= 4:
            return f"Extended loss streak of {longest_loss} highlighted need for adaptation"
        elif total_rounds > 15:
            return "Gradual strategic evolution throughout extended session"
        else:
            return "No major turning points in short session"

    def _identify_main_weakness(self, patterns: Dict[str, Any], performance: Dict[str, Any]) -> str:
        """Identify the main weakness that was exploited"""
        predictability = patterns.get('predictability_score', 0.0)
        pattern_type = patterns.get('pattern_type', 'unknown')
        win_rate = performance.get('overall_performance', {}).get('win_rate', 0.0)
        
        if predictability > 0.7:
            return f"High predictability ({predictability:.1%}) due to {pattern_type} patterns"
        elif win_rate < 0.4:
            return "Poor strategic adaptation against AI counter-moves"
        elif pattern_type == 'single_move_repetition':
            return "Tendency to repeat same moves too frequently"
        else:
            return "Inconsistent randomization strategy"

    def _suggest_biggest_improvement(self, patterns: Dict[str, Any], performance: Dict[str, Any]) -> str:
        """Suggest the biggest area for improvement"""
        predictability = patterns.get('predictability_score', 0.0)
        entropy = patterns.get('entropy_calculation', 0.0)
        win_rate = performance.get('overall_performance', {}).get('win_rate', 0.0)
        
        if predictability > 0.6:
            return "Focus on unpredictability - break patterns and increase randomness"
        elif entropy < 1.0:
            return "Improve move variety - aim for more balanced distribution"
        elif win_rate < 0.4:
            return "Study AI patterns and develop better counter-strategies"
        else:
            return "Fine-tune timing and psychological awareness"

    def _extract_strategic_lesson(self, ai_behavior: Dict[str, Any], performance: Dict[str, Any]) -> str:
        """Extract the key strategic lesson from the session"""
        robot_strategy = ai_behavior.get('ai_strategy', 'unknown')
        trend = performance.get('recent_performance', {}).get('trend', 'stable')
        
        if robot_strategy == 'frequency_based' and trend == 'improving':
            return "Successfully countered frequency analysis by varying move patterns"
        elif robot_strategy == 'markov_chain':
            return "Pattern sequences matter - AI learns from move combinations"
        elif robot_strategy == 'adaptive':
            return "Adaptive AI requires constant strategy evolution"
        else:
            return "Maintaining unpredictability is key to success"

    def _assess_consistency_evolution(self, patterns: Dict[str, Any], performance: Dict[str, Any]) -> str:
        """Assess how consistency evolved throughout session"""
        pattern_type = patterns.get('pattern_type', 'unknown')
        trend = performance.get('recent_performance', {}).get('trend', 'stable')
        
        if pattern_type == 'mixed_pattern' and trend == 'stable':
            return "Maintained good strategic consistency throughout"
        elif pattern_type == 'high_variation':
            return "Highly varied play - sometimes too unpredictable for optimal strategy"
        elif pattern_type == 'low_variation':
            return "Too consistent - became predictable over time"
        else:
            return "Evolving strategic approach with room for refinement"

    def _assess_emotional_resilience(self, performance: Dict[str, Any]) -> str:
        """Assess emotional resilience during the session"""
        streaks = performance.get('streaks', {})
        longest_loss = streaks.get('longest_loss_streak', 0)
        momentum = performance.get('momentum', {}).get('momentum_direction', 'stable')
        
        if longest_loss >= 5:
            return "Showed resilience during extended losing streak"
        elif momentum == 'positive':
            return "Strong emotional control - maintained positive momentum"
        elif momentum == 'negative':
            return "Struggled with emotional balance during difficult periods"
        else:
            return "Stable emotional approach throughout session"

    def _generate_real_time_advice(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate real-time coaching advice using ALL comprehensive analytics with LLM-specific strategies"""
        
        print(f"🤖 LLM TYPE: {type(self.llm).__name__} (Style: {self.coaching_style})")
        
        # Extract ALL available metrics for comprehensive analysis
        core = metrics.get('core_game', {})
        patterns = metrics.get('patterns', {})
        advanced = metrics.get('advanced', {})
        psychological = metrics.get('psychological', {})
        ai_behavior = metrics.get('ai_behavior', {})
        performance = metrics.get('performance', {})
        temporal = metrics.get('temporal', {})
        strategic = metrics.get('strategic', {})
        
        # Create COMPREHENSIVE context with ALL metrics plus STRATEGIC INSIGHTS
        full_context = {
            # Core game data
            'current_round': core.get('current_round', 0),
            'total_moves': core.get('total_moves', 0),
            'win_rate': core.get('win_rates', {}).get('human', 0.0),
            'recent_moves': core.get('recent_moves', {}).get('human_last_5', []),
            'recent_results': core.get('results', [])[-5:] if core.get('results') else [],
            
            # Pattern analysis
            'entropy': patterns.get('entropy_calculation', 0.0),
            'predictability': patterns.get('predictability_score', 0.0),
            'move_distribution': patterns.get('move_distribution', {}),
            'sequence_patterns': patterns.get('sequence_patterns', {}),
            'cycling_patterns': patterns.get('cycling_detection', {}),
            
            # Advanced analytics
            'decision_complexity': advanced.get('complexity_metrics', {}).get('decision_complexity', 0.0),
            'strategy_consistency': advanced.get('complexity_metrics', {}).get('strategy_consistency', 0.0),
            'adaptation_rate': advanced.get('complexity_metrics', {}).get('adaptation_rate', 0.0),
            'nash_distance': advanced.get('game_theory_metrics', {}).get('nash_equilibrium_distance', 0.33),
            'exploitability': advanced.get('game_theory_metrics', {}).get('exploitability', 0.0),
            'mutual_information': advanced.get('information_theory', {}).get('mutual_information', 0.0),
            'compression_ratio': advanced.get('information_theory', {}).get('compression_ratio', 0.0),
            
            # Psychological metrics
            'impulsiveness': psychological.get('decision_making_style', {}).get('impulsiveness_indicator', 0.5),
            'consistency_score': psychological.get('decision_making_style', {}).get('consistency_score', 0.5),
            'risk_tolerance': psychological.get('decision_making_style', {}).get('risk_tolerance', {}),
            'emotional_indicators': psychological.get('emotional_indicators', {}),
            'cognitive_patterns': psychological.get('cognitive_patterns', {}),
            
            # AI behavior analysis
            'robot_strategy': ai_behavior.get('ai_strategy', 'unknown'),
            'human_strategy_label': ai_behavior.get('human_strategy_label', 'unknown'),
            'ai_confidence': ai_behavior.get('confidence_history', {}).get('enhanced', [0.5])[-1] if ai_behavior.get('confidence_history', {}).get('enhanced') else 0.5,
            'ai_adaptation': ai_behavior.get('ai_adaptation', {}),
            'prediction_patterns': ai_behavior.get('prediction_patterns', {}),
            
            # Performance data
            'streaks': performance.get('streaks', {}),
            'momentum': performance.get('momentum', {}),
            'recent_performance': performance.get('recent_performance', {}),
            
            # Temporal analysis
            'game_phase': temporal.get('game_phase', 'unknown'),
            'performance_timeline': temporal.get('performance_timeline', {}),
            
            # Strategic context
            'current_strategy_assessment': strategic.get('current_strategy_assessment', {}),
            'strategic_opportunities': strategic.get('strategic_opportunities', {}),
            'strategic_weaknesses': strategic.get('weaknesses', {}),
            'educational_focus': strategic.get('educational_focus', {}),
            
            # NEW STRATEGIC INSIGHTS for better coaching
            'exploitable_patterns': self._identify_exploitable_patterns(core, patterns),
            'predictability_risk': self._assess_predictability_risk(core.get('recent_moves', {}).get('human_last_5', []), patterns.get('predictability_score', 0.0)),
            'next_move_recommendation': self._suggest_next_move(core, patterns, ai_behavior),
            'opponent_weakness': self._identify_opponent_weakness(ai_behavior, core),
            'pattern_disruption_advice': self._generate_pattern_disruption_advice(core.get('recent_moves', {}).get('human_last_5', [])),
            'psychological_state': self._assess_psychological_state(performance.get('streaks', {}), core.get('results', [])[-5:] if core.get('results') else []),
            'tactical_situation': self._assess_tactical_situation(core.get('current_round', 0), performance.get('momentum', {})),
            'adaptation_opportunity': self._identify_adaptation_opportunity(ai_behavior, temporal.get('game_phase', 'unknown')),
            'strategic_priority': self._determine_strategic_priority(patterns.get('predictability_score', 0.0), core.get('win_rates', {}).get('human', 0.0))
        }
        
        print(f"📊 FEEDING {len(full_context)} METRICS TO LLM")
        
        # Choose strategy based on LLM type
        if hasattr(self, 'trained_coach') and self.trained_coach is not None:
            return self._generate_trained_model_advice(metrics)  # Use original metrics for trained model
        elif isinstance(self.llm, BasicCoach):
            return self._generate_basic_coach_advice(full_context)
        else:
            return self._generate_real_llm_advice(full_context)
    
    def _generate_trained_model_advice(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate advice using our trained RPS coaching model"""
        print("🤖 Using Trained RPS Coaching Model")
        
        if self.trained_coach is None:
            return {
                'error': 'Trained model not available',
                'tips': ['Trained model not loaded'],
                'insights': {},
                'confidence_level': 0.0,
                'response_type': 'error'
            }
        
        try:
            # Use the trained coach's generate_coaching_advice method
            advice = self.trained_coach.generate_coaching_advice(
                comprehensive_metrics=metrics,
                coaching_type='real_time'
            )
            
            return advice
            
        except Exception as e:
            print(f"❌ Trained model error: {e}")
            return {
                'error': f'Trained model failed: {str(e)}',
                'tips': ['Try switching to a different LLM type'],
                'insights': {},
                'confidence_level': 0.0,
                'response_type': 'error'
            }
    
    def _generate_basic_coach_advice(self, full_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate advice using BasicCoach with simplified logic and only 3 tips"""
        print("🎯 Using Basic coaching algorithm")
        
        # Create simplified prompt for BasicCoach with only essential data
        simplified_prompt = f"""
Basic coaching request for Rock-Paper-Scissors player.

ESSENTIAL GAME DATA:
Round: {full_context['current_round']}
Win Rate: {full_context['win_rate']:.1%}
Recent Moves: {full_context['recent_moves']}
Recent Results: {full_context['recent_results']}

KEY METRICS:
Predictability: {full_context['predictability']:.3f} (0=random, 1=predictable)
Move Distribution: {full_context['move_distribution']}
Current Pattern: {full_context.get('pattern_type', 'mixed')}
AI Strategy: {full_context['robot_strategy']}
"""

        response_text = self._safe_llm_call(simplified_prompt)
        
        # Parse BasicCoach response (should be JSON)
        try:
            parsed_response = self.output_parser.parse(response_text)
            return {
                'tips': parsed_response.tips,
                'insights': parsed_response.insights,
                'educational_content': parsed_response.educational_content,
                'behavioral_analysis': parsed_response.behavioral_analysis,
                'confidence_level': parsed_response.confidence_level,
                'response_type': parsed_response.response_type,
                'llm_type': 'Basic',
                'metrics_count': 6,  # Only using 6 essential metrics
                'raw_response': response_text[:200] if isinstance(response_text, str) else str(response_text)[:200]
            }
        except Exception as e:
            print(f"⚠️ Basic coaching response parsing failed: {e}")
            return self._generate_fallback_advice({}, 'real_time')
    def _safe_llm_call(self, prompt: str) -> str:
        """Safely call LLM with proper error handling"""
        try:
            # Handle different LLM types with their respective calling methods
            if isinstance(self.llm, BasicCoach):
                return self.llm(prompt)
            elif hasattr(self.llm, 'invoke'):
                # Modern LangChain LLMs use invoke
                response = self.llm.invoke(prompt)
                return str(response) if response else ""
            elif hasattr(self.llm, '__call__'):
                # Fallback to direct call
                response = self.llm(prompt)
                return str(response) if response else ""
            else:
                # Last resort
                return "Unable to generate response with current LLM"
        except Exception as e:
            print(f"⚠️ LLM call failed: {e}")
            return f"Error generating response: {str(e)}"
        
        # Parse MockLLM response (should be JSON)
        try:
            parsed_response = self.output_parser.parse(response_text)
            return {
                'tips': parsed_response.tips,
                'insights': parsed_response.insights,
                'educational_content': parsed_response.educational_content,
                'behavioral_analysis': parsed_response.behavioral_analysis,
                'confidence_level': parsed_response.confidence_level,
                'response_type': parsed_response.response_type,
                'llm_type': 'MockLLM',
                'metrics_count': len(full_context),
                'raw_response': response_text[:200] if isinstance(response_text, str) else str(response_text)[:200]
            }
        except Exception as e:
            print(f"⚠️ MockLLM response parsing failed: {e}")
            return self._generate_fallback_advice({}, 'real_time')
    
    def _generate_real_llm_advice(self, full_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate advice using Real LLM with natural language format"""
        print(f"🌐 Using Real LLM: {type(self.llm).__name__}")
        
        # Create natural language prompt optimized for real LLMs
        style_instruction = "scientific and detailed" if self.coaching_style == "scientific" else "easy-to-understand and encouraging"
        
        natural_prompt = f"""You are an expert Rock-Paper-Scissors coach helping a player improve their game. Please provide helpful, {style_instruction} advice.

PLAYER'S CURRENT SITUATION:
• Round {full_context['current_round']} with {full_context['win_rate']:.1%} win rate
• Recent moves: {full_context['recent_moves']}
• Entropy (randomness): {full_context['entropy']:.3f} out of 1.585 maximum
• Predictability: {full_context['predictability']:.3f} (lower is better)  
• Nash distance: {full_context['nash_distance']:.3f} (measures optimal play balance)
• AI opponent using: {full_context['robot_strategy']} strategy
• Decision complexity: {full_context['decision_complexity']:.3f}
• Pattern consistency: {full_context['strategy_consistency']:.3f}

ANALYSIS CONTEXT:
The player's move distribution shows {full_context['move_distribution']}. Their recent performance trend is {full_context.get('momentum', {})}.

Please provide coaching advice that includes:
1. Immediate tactical tips for the next few moves
2. Strategic insights about their current playing patterns  
3. Specific ways to improve their randomness and unpredictability
4. How to counter the AI's {full_context['robot_strategy']} strategy

Be conversational and encouraging while being specific about the numbers and what they mean."""

        # Actually call the LLM with the prompt
        try:
            response_text = self.llm(natural_prompt)
            print(f"📝 Real LLM Response Preview: {response_text[:100]}...")
            
            # Return structured response for real LLM
            return {
                'tips': [response_text[:500]],  # First 500 chars as main tip
                'insights': {
                    'ai_insights': {
                        'analysis': response_text,
                        'strategic_assessment': f"Based on {full_context['current_round']} rounds of play"
                    }
                },
                'educational_content': {
                    'explanation': response_text,
                    'key_concepts': f"Entropy: {full_context['entropy']:.3f}, Predictability: {full_context['predictability']:.3f}"
                },
                'behavioral_analysis': {
                    'playing_style': f"Pattern type: {full_context.get('pattern_type', 'mixed')}",
                    'recommendations': response_text[-200:] if len(response_text) > 200 else response_text
                },
                'confidence_level': 'high',
                'response_type': 'real_time',
                'llm_type': type(self.llm).__name__,
                'raw_response': response_text[:300],  # More text for natural language
                'natural_language_full': response_text[:500]  # Full response for natural language display
            }
        except Exception as e:
            print(f"⚠️ Real LLM call failed: {e}")
            # Fallback to fallback advice
            fallback_response = f"Based on your {full_context['current_round']} rounds with {full_context['win_rate']:.1%} win rate: Try to be more unpredictable. Your entropy is {full_context['entropy']:.3f} - aim for higher randomness. The AI is using {full_context['robot_strategy']} strategy, so vary your patterns more."
            return {
                'tips': [fallback_response],
                'insights': {'ai_insights': {'analysis': fallback_response}},
                'educational_content': {'explanation': fallback_response},
                'behavioral_analysis': {'playing_style': fallback_response},
                'confidence_level': 'medium',
                'response_type': 'real_time_fallback',
                'llm_type': type(self.llm).__name__,
                'raw_response': fallback_response[:300],
                'natural_language_full': fallback_response
            }

    def _generate_mockllm_advice(self, full_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate advice using MockLLM with enhanced strategic context"""
        print("🧠 Using Enhanced MockLLM with ALL metrics + Strategic Insights")
        
        # Create strategic prompt with actionable insights
        strategic_prompt = f"""
STRATEGIC COACHING REQUEST: {self.coaching_style} style real-time advice

🎮 GAME STATUS:
Round: {full_context['current_round']}/{full_context['total_moves']}
Win Rate: {full_context['win_rate']:.1%}
Recent Moves: {full_context['recent_moves']}
Recent Results: {full_context['recent_results']}

🎯 STRATEGIC INSIGHTS:
{full_context['strategic_priority']}
{full_context['exploitable_patterns']}
{full_context['predictability_risk']}

💡 TACTICAL RECOMMENDATIONS:
Next Move: {full_context['next_move_recommendation']}
Pattern Fix: {full_context['pattern_disruption_advice']}
Opponent Weakness: {full_context['opponent_weakness']}

🧠 PSYCHOLOGICAL STATE:
{full_context['psychological_state']}
{full_context['tactical_situation']}

⚡ ADAPTATION OPPORTUNITY:
{full_context['adaptation_opportunity']}

📊 ADVANCED METRICS:
Entropy: {full_context['entropy']:.4f} (max=1.585, randomness measure)
Predictability: {full_context['predictability']:.4f} (lower=better)
Nash Distance: {full_context['nash_distance']:.4f} (optimal=0)
Robot Strategy: {full_context['robot_strategy']}
Complexity: {full_context['decision_complexity']:.4f}
"""

        response_text = self._safe_llm_call(strategic_prompt)
        
        # Parse MockLLM response (should be JSON)
        try:
            parsed_response = self.output_parser.parse(response_text)
            return {
                'tips': parsed_response.tips,
                'insights': parsed_response.insights,
                'educational_content': parsed_response.educational_content,
                'behavioral_analysis': parsed_response.behavioral_analysis,
                'confidence_level': parsed_response.confidence_level,
                'response_type': parsed_response.response_type,
                'llm_type': 'Enhanced_MockLLM',
                'metrics_count': len(full_context),
                'strategic_insights': {
                    'priority': full_context['strategic_priority'],
                    'next_move': full_context['next_move_recommendation'],
                    'pattern_risk': full_context['exploitable_patterns'],
                    'opponent_weakness': full_context['opponent_weakness']
                },
                'raw_response': response_text[:200] if isinstance(response_text, str) else str(response_text)[:200]
            }
        except Exception as e:
            print(f"⚠️ Enhanced MockLLM response parsing failed: {e}")
            return self._generate_fallback_advice({}, 'real_time')
    
    def _generate_comprehensive_analysis(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive post-game analysis"""
        
        # Extract comprehensive metrics
        core = metrics.get('core_game', {})
        patterns = metrics.get('patterns', {})
        performance = metrics.get('performance', {})
        psychological = metrics.get('psychological', {})
        ai_behavior = metrics.get('ai_behavior', {})
        
        # Create enhanced session data summary with strategic insights
        temporal_metrics = metrics.get('temporal', {})
        session_data = f"""
📊 SESSION OVERVIEW:
Total Rounds: {core.get('current_round', 0)}
Final Performance: {core.get('win_rates', {}).get('human', 0.0):.1%} win rate
Session Duration: {temporal_metrics.get('session_duration', 0):.0f} seconds

🎯 PATTERN ANALYSIS:
Pattern Type: {patterns.get('pattern_type', 'unknown')} patterns detected
Final Entropy: {patterns.get('entropy_calculation', 0.0):.3f} (max 1.585)
Final Predictability: {patterns.get('predictability_score', 0.0):.3f} (lower is better)
Most Exploited Pattern: {self._identify_most_exploited_pattern(core, patterns)}

📈 PERFORMANCE EVOLUTION:
Starting Performance: {self._estimate_early_performance(performance)}
Ending Performance: {self._estimate_late_performance(performance)}
Best Streak: {performance.get('streaks', {}).get('longest_win_streak', 0)} wins
Worst Streak: {performance.get('streaks', {}).get('longest_loss_streak', 0)} losses
Overall Trend: {performance.get('recent_performance', {}).get('trend', 'stable')}

🧠 STRATEGIC DEVELOPMENT:
AI Strategy Faced: {ai_behavior.get('ai_strategy', 'unknown')}
Player Adaptation: {self._assess_adaptation_success(performance, temporal_metrics)}
Learning Indicators: {self._identify_learning_patterns(patterns, performance)}
Key Turning Point: {self._identify_key_turning_point(performance, core)}

🎓 EDUCATIONAL INSIGHTS:
Main Weakness Exploited: {self._identify_main_weakness(patterns, performance)}
Biggest Improvement Area: {self._suggest_biggest_improvement(patterns, performance)}
Strategic Lesson Learned: {self._extract_strategic_lesson(ai_behavior, performance)}

💡 PSYCHOLOGICAL PROFILE:
Decision Making Style: {psychological.get('decision_making_style', {})}
Consistency Throughout: {self._assess_consistency_evolution(patterns, performance)}
Emotional Resilience: {self._assess_emotional_resilience(performance)}
"""
        
        # Use BasicCoach directly for fast, reliable responses
        if isinstance(self.llm, BasicCoach):
            # Create simple prompt for BasicCoach comprehensive analysis
            prompt_text = f"Comprehensive analysis for {self.coaching_style} style:\n{session_data}"
            response_text = self.llm(prompt_text)
        else:
            # Fallback to LangChain approach (if real LLM is used)
            try:
                style_instruction = "scientifically rigorous" if self.coaching_style == "scientific" else "accessible and easy-to-understand"
                input_vars = {
                    "session_data": session_data,
                    "style": self.coaching_style,
                    "style_instruction": style_instruction
                }
                if LANGCHAIN_AVAILABLE and self.comprehensive_chain:
                    response_text = self.comprehensive_chain.invoke(input_vars)
                else:
                    response_text = "Unable to generate comprehensive analysis"
            except Exception as e:
                print(f"⚠️ LangChain comprehensive analysis failed: {e}")
                # Fall back to BasicCoach approach
                basic_coach = BasicCoach()
                basic_coach.set_coaching_style(self.coaching_style)
                prompt_text = f"Comprehensive analysis for {self.coaching_style} style:\n{session_data}"
                response_text = basic_coach(prompt_text)
        
        # Parse response and structure for frontend compatibility
        try:
            if isinstance(response_text, str) and response_text.strip().startswith('{'):
                parsed_response = json.loads(response_text)
                
                # Structure the response to match frontend expectations
                return {
                    # Basic analysis fields (always shown)
                    'psychological_patterns': parsed_response.get('psychological_patterns', ''),
                    'strategic_evolution': parsed_response.get('strategic_evolution', ''),
                    'learning_recommendations': parsed_response.get('learning_recommendations', ''),
                    'educational_summary': parsed_response.get('educational_summary', ''),
                    'tips': [parsed_response.get('learning_recommendations', 'Focus on strategic improvement')],
                    
                    # Advanced analytics fields (shown when Advanced Analytics is enabled)
                    'insights': {
                        'ai_insights': {
                            'decision_analysis': parsed_response.get('decision_analysis', ''),
                            'fascinating_discoveries': parsed_response.get('fascinating_discoveries', ''),
                            'information_theory_insights': parsed_response.get('information_theory_insights', ''),
                            'game_theory_assessment': parsed_response.get('game_theory_assessment', '')
                        }
                    },
                    'educational_content': {
                        'theoretical_insights': parsed_response.get('educational_summary', ''),
                        'strategy_fundamentals': parsed_response.get('strategic_evolution', ''),
                        'improvement_methodology': parsed_response.get('learning_recommendations', '')
                    },
                    'behavioral_analysis': {
                        'psychological_profile': parsed_response.get('psychological_patterns', ''),
                        'decision_making_patterns': parsed_response.get('decision_analysis', ''),
                        'cognitive_load_analysis': parsed_response.get('cognitive_load_analysis', ''),
                        'adaptive_behaviors': parsed_response.get('fascinating_discoveries', '')
                    },
                    
                    # Additional metadata
                    'confidence_level': 'high',
                    'response_type': 'comprehensive',
                    'llm_type': type(self.llm).__name__,
                    'natural_language_full': str(response_text)[:500],
                    'raw_response': str(response_text)[:300]
                }
            else:
                # Handle unstructured response with enhanced structure
                response_str = str(response_text) if response_text else "Analysis not available"
                return {
                    'psychological_patterns': response_str[:200],
                    'strategic_evolution': 'Analysis based on complete session data',
                    'learning_recommendations': 'Focus on areas identified in analysis',
                    'educational_summary': 'Comprehensive review completed',
                    'tips': ['Review your patterns', 'Focus on strategic balance', 'Practice counter-strategies'],
                    'insights': {
                        'ai_insights': {
                            'decision_analysis': 'Patterns observed throughout gameplay',
                            'fascinating_discoveries': 'Strategic insights from session data',
                            'information_theory_insights': 'Entropy and predictability insights',
                            'game_theory_assessment': 'Strategic assessment findings'
                        }
                    },
                    'educational_content': {
                        'theoretical_insights': 'Game theory and strategic thinking concepts',
                        'strategy_fundamentals': 'Core Rock-Paper-Scissors strategic principles',
                        'improvement_methodology': 'Systematic approach to skill development'
                    },
                    'behavioral_analysis': {
                        'psychological_profile': 'Decision-making style analysis',
                        'decision_making_patterns': 'Choice patterns and tendencies',
                        'cognitive_load_analysis': 'Decision-making complexity analysis',
                        'adaptive_behaviors': 'Learning and adaptation indicators'
                    },
                    'confidence_level': 'medium',
                    'response_type': 'comprehensive_fallback',
                    'llm_type': type(self.llm).__name__,
                    'natural_language_full': response_str[:500],
                    'raw_response': response_str[:300]
                }
                
        except Exception as e:
            print(f"⚠️ Comprehensive analysis parsing failed: {e}")
            return self._generate_fallback_advice(metrics, 'comprehensive')
    
    def generate_comprehensive_analysis(self, comprehensive_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive analysis (public method)"""
        return self._generate_comprehensive_analysis(comprehensive_metrics)
    
    def _generate_fallback_advice(self, metrics: Dict[str, Any], coaching_type: str) -> Dict[str, Any]:
        """Generate fallback advice when LLM fails"""
        
        core = metrics.get('core_game', {})
        patterns = metrics.get('patterns', {})
        
        if coaching_type == 'real_time':
            return {
                'tips': [
                    "Focus on varying your move patterns to stay unpredictable.",
                    "Observe the AI's responses and adapt your strategy accordingly.",
                    "Try the opposite of what you just played to break patterns."
                ],
                'insights': {
                    'pattern_detected': patterns.get('pattern_type', 'unknown'),
                    'predictability': patterns.get('predictability_score', 0.0)
                },
                'educational_content': {
                    'focus': 'pattern_breaking',
                    'current_round': core.get('current_round', 0)
                },
                'behavioral_analysis': {
                    'style': 'fallback_analysis'
                },
                'confidence_level': 0.5,
                'response_type': 'fallback'
            }
        else:
            return self._generate_fallback_comprehensive_analysis(metrics)
    
    def _generate_fallback_comprehensive_analysis(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback comprehensive analysis"""
        
        core = metrics.get('core_game', {})
        patterns = metrics.get('patterns', {})
        performance = metrics.get('performance', {})
        
        return {
            'psychological_patterns': f"Played {core.get('current_round', 0)} rounds with {patterns.get('pattern_type', 'unknown')} pattern tendencies",
            'strategic_evolution': f"Performance shows {performance.get('recent_performance', {}).get('trend', 'stable')} trend",
            'decision_analysis': "Analysis based on move patterns and performance metrics",
            'learning_recommendations': "Focus on increasing unpredictability and pattern awareness",
            'fascinating_discoveries': "Interesting adaptation patterns observed throughout the session",
            'educational_summary': "Session provides insights into decision-making patterns and strategic thinking"
        }


# Global instance (lazy loaded)
_langchain_coach_instance = None

def get_langchain_coach():
    """Get global LangChain coach instance (lazy loaded)"""
    global _langchain_coach_instance
    if _langchain_coach_instance is None:
        _langchain_coach_instance = LangChainAICoach()
    return _langchain_coach_instance
