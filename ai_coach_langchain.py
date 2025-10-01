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


class MockLLM:
    """Enhanced Mock LLM for testing when LangChain is not available - uses all advanced analytics"""
    
    def __init__(self):
        self.coaching_style = 'easy'  # Default to easy style
    
    def set_coaching_style(self, style: str):
        """Set coaching style for mock responses"""
        self.coaching_style = style
    
    def __call__(self, prompt: str) -> str:
        """Generate sophisticated mock response based on prompt content and analytics"""
        
        if 'real_time' in prompt.lower() or 'Round:' in prompt or 'COACHING REQUEST:' in prompt:
            return self._generate_mock_realtime_advice(prompt)
        elif 'comprehensive' in prompt.lower() or 'Total Rounds:' in prompt:
            return self._generate_mock_comprehensive_analysis(prompt)
        else:
            return self._generate_mock_general_advice(prompt)
    
    def _extract_metrics_from_prompt(self, prompt: str) -> dict:
        """Extract key metrics from the prompt for intelligent mock responses"""
        metrics = {}
        
        # Extract key values using simple parsing
        import re
        
        # Extract current round (handles both "Round: 15/25" and "Current Round: 15" formats)
        round_match = re.search(r'Round: (\d+)', prompt)
        metrics['round'] = int(round_match.group(1)) if round_match else 0
        
        # Extract win rate
        win_rate_match = re.search(r'Win Rate: ([\d.]+)%', prompt)
        metrics['win_rate'] = float(win_rate_match.group(1)) / 100 if win_rate_match else 0.33
        
        # Extract entropy (handles "Entropy: X.XXXX" format)
        entropy_match = re.search(r'Entropy: ([\d.]+)', prompt)
        metrics['entropy'] = float(entropy_match.group(1)) if entropy_match else 1.0
        
        # Extract predictability (handles "Predictability: X.XXXX" format)
        pred_match = re.search(r'Predictability: ([\d.]+)', prompt)
        metrics['predictability'] = float(pred_match.group(1)) if pred_match else 0.5
        
        # Extract strategy
        strategy_match = re.search(r'Robot Strategy: (\w+)', prompt)
        metrics['robot_strategy'] = strategy_match.group(1) if strategy_match else 'unknown'
        
        # Extract decision complexity
        complexity_match = re.search(r'Decision Complexity: ([\d.]+)', prompt)
        metrics['complexity'] = float(complexity_match.group(1)) if complexity_match else 0.5
        
        # Extract consistency
        consistency_match = re.search(r'Strategy Consistency: ([\d.]+)', prompt)
        metrics['consistency'] = float(consistency_match.group(1)) if consistency_match else 0.5
        
        # Extract Nash distance
        nash_match = re.search(r'Nash Distance: ([\d.]+)', prompt)
        metrics['nash_distance'] = float(nash_match.group(1)) if nash_match else 0.33
        
        return metrics
        
        # Extract predictability
        pred_match = re.search(r'Predictability Score: ([\d.]+)', prompt)
        metrics['predictability'] = float(pred_match.group(1)) if pred_match else 0.5
        
        # Extract strategy
        strategy_match = re.search(r'Robot Strategy: (\w+)', prompt)
        metrics['robot_strategy'] = strategy_match.group(1) if strategy_match else 'unknown'
        
        # Extract decision complexity
        complexity_match = re.search(r'Decision Complexity: ([\d.]+)', prompt)
        metrics['complexity'] = float(complexity_match.group(1)) if complexity_match else 0.5
        
        # Extract consistency
        consistency_match = re.search(r'Strategy Consistency: ([\d.]+)', prompt)
        metrics['consistency'] = float(consistency_match.group(1)) if consistency_match else 0.5
        
        # Extract Nash distance
        nash_match = re.search(r'Nash Equilibrium Distance: ([\d.]+)', prompt)
        metrics['nash_distance'] = float(nash_match.group(1)) if nash_match else 0.33
        
        return metrics
    
    def _generate_mock_realtime_advice(self, prompt: str) -> str:
        """Generate sophisticated mock real-time coaching advice using extracted analytics"""
        
        metrics = self._extract_metrics_from_prompt(prompt)
        
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
    
    def _generate_mock_comprehensive_analysis(self, prompt: str) -> str:
        """Generate sophisticated mock comprehensive analysis using all advanced analytics"""
        
        metrics = self._extract_metrics_from_prompt(prompt)
        
        if self.coaching_style == 'easy':
            # Easy-to-understand comprehensive analysis
            psychological_patterns = f"""🧠 **Your Playing Style Analysis:**
Your decision-making shows {'consistent patterns' if metrics['consistency'] > 0.6 else 'adaptive flexibility'}. 
You've played {metrics['round']} rounds with {metrics['entropy']:.2f} randomness level 
({'excellent variety!' if metrics['entropy'] > 1.2 else 'room to mix things up more'})."""

            strategic_evolution = f"""📈 **How You've Improved:**
Starting strategy: Mixed approach → Current: {'Pattern-aware' if metrics['complexity'] > 0.5 else 'Intuitive play'}
Your win rate of {metrics['win_rate']:.1%} shows {'strong performance' if metrics['win_rate'] > 0.4 else 'room for improvement'}.
The AI adapted its strategy to {metrics['robot_strategy']}, but you {'handled it well' if metrics['nash_distance'] < 0.35 else 'can learn to counter it better'}."""

            decision_analysis = f"""🎯 **Your Decision Making:**
Predictability: {metrics['predictability']:.2f} ({'Great!' if metrics['predictability'] < 0.5 else 'Work on this!'})
Move balance: {'Well distributed' if metrics['nash_distance'] < 0.35 else 'Could be more balanced'}
Timing: {'Thoughtful' if metrics['complexity'] > 0.4 else 'Quick and intuitive'}"""

            learning_recommendations = f"""📚 **What to Focus on Next:**
1. {'Keep up the great randomness!' if metrics['entropy'] > 1.2 else 'Practice mixing moves more randomly'}
2. {'Maintain your unpredictability' if metrics['predictability'] < 0.5 else 'Work on breaking patterns faster'}
3. {'You understand the AI well' if metrics['robot_strategy'] != 'unknown' else 'Watch for AI patterns'}"""

            fascinating_discoveries = f"""🔍 **Cool Insights About Your Play:**
• You have {'high' if metrics['complexity'] > 0.6 else 'moderate'} strategic awareness
• Your consistency score ({metrics['consistency']:.2f}) suggests {'stable' if metrics['consistency'] > 0.6 else 'evolving'} decision-making
• Distance from perfect play: {metrics['nash_distance']:.2f} (closer to 0 is better)"""

            educational_summary = f"""🎓 **Key Lessons:**
Rock-Paper-Scissors is about balancing randomness with strategy. Your entropy of {metrics['entropy']:.2f} 
shows {'excellent' if metrics['entropy'] > 1.2 else 'developing'} randomization skills. 
Keep practicing unpredictability while staying balanced!"""

        else:
            # Scientific comprehensive analysis
            psychological_patterns = f"""🧠 **Cognitive Decision Analysis:**
Decision complexity index: {metrics['complexity']:.4f} (cognitive load indicator)
Consistency coefficient: {metrics['consistency']:.4f} (behavioral stability measure)
Pattern awareness correlation: {(1-metrics['predictability']):.4f} (counter-exploitation effectiveness)
Entropy coefficient: {metrics['entropy']:.4f}/{1.585:.3f} ({(metrics['entropy']/1.585)*100:.1f}% of theoretical maximum)"""

            strategic_evolution = f"""📊 **Strategic Development Trajectory:**
Initial state: Random baseline → Current: {metrics['robot_strategy']}-aware adaptation
Performance metrics: {metrics['win_rate']:.4f} win rate (μ=0.333 for random play)
Nash equilibrium deviation: {metrics['nash_distance']:.4f} (σ from uniform distribution)
Strategic complexity evolution: {metrics['complexity']:.4f} (decision tree depth indicator)"""

            decision_analysis = f"""⚖️ **Game Theory Analysis:**
Information entropy: {metrics['entropy']:.4f} bits (Shannon entropy measure)
Predictability coefficient: {metrics['predictability']:.4f} (inverse entropy correlation)
Nash distance: {metrics['nash_distance']:.4f} (optimal mixed strategy deviation)
Exploitability index: Estimated {metrics['predictability']*100:.1f}% (pattern exploitation vulnerability)"""

            learning_recommendations = f"""🔬 **Optimization Recommendations:**
1. Entropy target: Achieve >1.400 for near-optimal randomization
2. Nash distance: Minimize to <0.200 for equilibrium approximation  
3. Predictability: Maintain <0.300 for anti-exploitation defense
4. Complexity: Balance cognitive load with strategic depth"""

            fascinating_discoveries = f"""🔬 **Statistical Behavioral Analysis:**
• Entropy-complexity correlation: {metrics['entropy']*metrics['complexity']:.4f} (strategic sophistication index)
• Anti-pattern effectiveness: {(1-metrics['predictability'])*100:.1f}% (counter-prediction success rate)
• Cognitive load distribution: {metrics['complexity']:.4f} (decision-making intensity measure)
• Strategic adaptation rate: {metrics['consistency']:.4f} (behavioral flexibility coefficient)"""

            educational_summary = f"""📖 **Information Theory Summary:**
Your Shannon entropy of {metrics['entropy']:.4f} represents {(metrics['entropy']/1.585)*100:.1f}% of maximum possible randomness.
Nash equilibrium analysis shows {metrics['nash_distance']:.4f} deviation from optimal mixed strategy.
Predictability analysis indicates {metrics['predictability']*100:.1f}% pattern vulnerability.
Cognitive complexity index suggests {'high-level' if metrics['complexity'] > 0.6 else 'developing'} strategic thinking."""

        return json.dumps({
            'psychological_patterns': psychological_patterns,
            'strategic_evolution': strategic_evolution,
            'decision_analysis': decision_analysis,
            'information_theory_insights': f"Entropy: {metrics['entropy']:.3f}, Information content: {(metrics['entropy']/1.585)*100:.1f}% of maximum",
            'game_theory_assessment': f"Nash distance: {metrics['nash_distance']:.3f}, Exploitability: {metrics['predictability']*100:.1f}%",
            'cognitive_load_analysis': f"Decision complexity: {metrics['complexity']:.3f}, Processing depth: {'High' if metrics['complexity'] > 0.6 else 'Moderate'}",
            'learning_recommendations': learning_recommendations,
            'fascinating_discoveries': fascinating_discoveries,
            'educational_summary': educational_summary,
            'personalized_roadmap': f"Phase 1: {'Maintain randomness' if metrics['entropy'] > 1.2 else 'Increase entropy'} → Phase 2: Optimize Nash distance → Phase 3: Master anti-exploitation"
        })
    
    def _generate_mock_general_advice(self, prompt: str) -> str:
        """Generate mock general advice with analytical depth"""
        return "Focus on maintaining unpredictability (entropy > 1.2) while approximating Nash equilibrium through balanced move distributions."


class LangChainAICoach:
    """
    LangChain-based AI coach with optimized prompts for RPS coaching
    """
    
    def __init__(self):
        # Initialize coaching style first
        self.coaching_style = 'easy'  # Default to easy style
        
        self.llm = self._initialize_llm()
        self.output_parser = CoachingOutputParser()
        self.memory = self._initialize_memory()
        
        # Initialize prompt templates
        self.real_time_prompt = self._create_real_time_prompt()
        self.comprehensive_prompt = self._create_comprehensive_prompt()
        
        # Initialize chains
        self.real_time_chain = self._create_real_time_chain()
        self.comprehensive_chain = self._create_comprehensive_chain()
        
        print("✅ LangChain AI Coach initialized successfully")
    
    def _initialize_llm(self) -> LLM:
        """Initialize LLM with fallback options - prioritizing fast MockLLM"""
        
        # Use MockLLM as primary choice for speed and reliability
        print("🚀 Using enhanced MockLLM for fast, intelligent coaching")
        mock_llm = MockLLM()
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

Based on the game data above, please provide personalized coaching advice. Include:

1. **Immediate Tips**: 2-3 specific things they can do right now to improve
2. **Pattern Insights**: What you notice about their playing style  
3. **Strategic Advice**: How they can adapt to beat the AI
4. **Encouragement**: Positive reinforcement about what they're doing well

Please respond in a natural, conversational way as if you're coaching them in person. Be specific about the numbers and patterns you see, but explain them in an accessible way."""

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

        real_template = """You are conducting a comprehensive post-game analysis of a Rock-Paper-Scissors session.

COMPLETE SESSION DATA:
{session_data}

ANALYSIS STYLE: Provide {style_instruction} insights that help the player understand their strategic development.

Please provide a thorough post-game analysis covering:

**🧠 Psychological Patterns**: What does their decision-making reveal about their playing style and cognitive approach?

**📈 Strategic Evolution**: How did their strategy develop throughout the session? What adaptations did they make?

**🎯 Decision Analysis**: What patterns emerge from their choices? Are they predictable or effectively random?

**📊 Performance Insights**: How do the metrics (entropy, predictability, win rate) tell the story of their gameplay?

**🎓 Learning Opportunities**: What specific areas should they focus on to improve?

**🔍 Fascinating Discoveries**: What interesting or unexpected patterns emerged from their play?

**📚 Educational Summary**: Key lessons and connections to game theory/strategy principles.

**🛤️ Personalized Roadmap**: Specific next steps tailored to their current skill level and playing style.

Please write this as if you're having a thoughtful conversation with the player about their game."""

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
        if LANGCHAIN_AVAILABLE:
            # Use modern RunnableSequence instead of deprecated LLMChain
            return self.real_time_prompt | self.llm
        return None
    
    def _create_comprehensive_chain(self):
        """Create chain for comprehensive analysis using modern LangChain API"""
        if LANGCHAIN_AVAILABLE:
            # Use modern RunnableSequence instead of deprecated LLMChain
            return self.comprehensive_prompt | self.llm
        return None
    
    def set_coaching_style(self, style: str) -> None:
        """Set coaching style preference"""
        if style in ['easy', 'scientific']:
            self.coaching_style = style
            # Update MockLLM style if it's being used
            if isinstance(self.llm, MockLLM):
                self.llm.set_coaching_style(style)
    
    def get_coaching_style(self) -> str:
        """Get current coaching style"""
        return self.coaching_style
    
    def set_llm_type(self, llm_type: str) -> Dict[str, Any]:
        """
        Set the LLM type for coaching
        
        Args:
            llm_type: 'mock' for fast MockLLM, 'real' for actual LLM models
            
        Returns:
            Status dictionary
        """
        if llm_type not in ['mock', 'real']:
            return {
                'success': False,
                'error': f'Invalid LLM type: {llm_type}. Must be "mock" or "real"',
                'current_llm_type': self.get_llm_type()
            }
        
        old_type = self.get_llm_type()
        
        try:
            if llm_type == 'mock':
                # Switch to MockLLM
                print("🔄 Switching to MockLLM for fast responses")
                mock_llm = MockLLM()
                mock_llm.set_coaching_style(self.coaching_style)
                self.llm = mock_llm
                
            elif llm_type == 'real':
                # Try to switch to real LLM
                print("🔄 Attempting to switch to real LLM")
                real_llm = self._try_real_llm()
                if real_llm is not None:
                    self.llm = real_llm
                else:
                    return {
                        'success': False,
                        'error': 'Real LLM not available. Remaining on current LLM.',
                        'current_llm_type': old_type
                    }
            
            # Update chains with new LLM
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
        if isinstance(self.llm, MockLLM):
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
        
        # Create COMPREHENSIVE context with ALL metrics
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
            'robot_strategy': ai_behavior.get('current_strategy', 'unknown'),
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
            'educational_focus': strategic.get('educational_focus', {})
        }
        
        print(f"📊 FEEDING {len(full_context)} METRICS TO LLM")
        
        # Choose strategy based on LLM type
        if isinstance(self.llm, MockLLM):
            return self._generate_mockllm_advice(full_context)
        else:
            return self._generate_real_llm_advice(full_context)
    
    def _generate_mockllm_advice(self, full_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate advice using MockLLM with JSON format"""
        print("🧠 Using Enhanced MockLLM with ALL metrics")
        
        # Create comprehensive prompt for MockLLM with ALL data
        comprehensive_prompt = f"""
COACHING REQUEST: {self.coaching_style} style real-time advice

COMPLETE GAME DATA:
Round: {full_context['current_round']}/{full_context['total_moves']}
Win Rate: {full_context['win_rate']:.1%}
Recent Moves: {full_context['recent_moves']}
Recent Results: {full_context['recent_results']}

PATTERN ANALYSIS:
Entropy: {full_context['entropy']:.4f} (max=1.585, randomness measure)
Predictability: {full_context['predictability']:.4f} (lower=better)
Move Distribution: {full_context['move_distribution']}
Sequence Patterns: {full_context['sequence_patterns']}

ADVANCED ANALYTICS:
Decision Complexity: {full_context['decision_complexity']:.4f}
Strategy Consistency: {full_context['strategy_consistency']:.4f}
Adaptation Rate: {full_context['adaptation_rate']:.4f}
Nash Distance: {full_context['nash_distance']:.4f} (optimal=0)
Exploitability: {full_context['exploitability']:.4f}
Mutual Information: {full_context['mutual_information']:.4f}
Compression Ratio: {full_context['compression_ratio']:.4f}

PSYCHOLOGICAL PROFILE:
Impulsiveness: {full_context['impulsiveness']:.4f}
Consistency: {full_context['consistency_score']:.4f}
Risk Tolerance: {full_context['risk_tolerance']}
Emotional State: {full_context['emotional_indicators']}
Cognitive Patterns: {full_context['cognitive_patterns']}

AI BEHAVIOR:
Robot Strategy: {full_context['robot_strategy']}
AI Confidence: {full_context['ai_confidence']:.4f}
AI Adaptation: {full_context['ai_adaptation']}
Prediction Patterns: {full_context['prediction_patterns']}

PERFORMANCE CONTEXT:
Streaks: {full_context['streaks']}
Momentum: {full_context['momentum']}
Game Phase: {full_context['game_phase']}
Strategic Assessment: {full_context['current_strategy_assessment']}
"""
    def _safe_llm_call(self, prompt: str) -> str:
        """Safely call LLM with proper error handling"""
        try:
            # Handle different LLM types with their respective calling methods
            if isinstance(self.llm, MockLLM):
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
        """Generate advice using MockLLM with JSON format"""
        print("🧠 Using Enhanced MockLLM with ALL metrics")
        
        # Create comprehensive prompt for MockLLM with ALL data
        comprehensive_prompt = f"""
COACHING REQUEST: {self.coaching_style} style real-time advice

COMPLETE GAME DATA:
Round: {full_context['current_round']}/{full_context['total_moves']}
Win Rate: {full_context['win_rate']:.1%}
Recent Moves: {full_context['recent_moves']}
Recent Results: {full_context['recent_results']}

PATTERN ANALYSIS:
Entropy: {full_context['entropy']:.4f} (max=1.585, randomness measure)
Predictability: {full_context['predictability']:.4f} (lower=better)
Move Distribution: {full_context['move_distribution']}
Sequence Patterns: {full_context['sequence_patterns']}

ADVANCED ANALYTICS:
Decision Complexity: {full_context['decision_complexity']:.4f}
Strategy Consistency: {full_context['strategy_consistency']:.4f}
Adaptation Rate: {full_context['adaptation_rate']:.4f}
Nash Distance: {full_context['nash_distance']:.4f} (optimal=0)
Exploitability: {full_context['exploitability']:.4f}
Mutual Information: {full_context['mutual_information']:.4f}
Compression Ratio: {full_context['compression_ratio']:.4f}

PSYCHOLOGICAL PROFILE:
Impulsiveness: {full_context['impulsiveness']:.4f}
Consistency: {full_context['consistency_score']:.4f}
Risk Tolerance: {full_context['risk_tolerance']}
Emotional State: {full_context['emotional_indicators']}
Cognitive Patterns: {full_context['cognitive_patterns']}

AI BEHAVIOR:
Robot Strategy: {full_context['robot_strategy']}
AI Confidence: {full_context['ai_confidence']:.4f}
AI Adaptation: {full_context['ai_adaptation']}
Prediction Patterns: {full_context['prediction_patterns']}

PERFORMANCE CONTEXT:
Streaks: {full_context['streaks']}
Momentum: {full_context['momentum']}
Game Phase: {full_context['game_phase']}
Strategic Assessment: {full_context['current_strategy_assessment']}
"""
        response_text = self._safe_llm_call(comprehensive_prompt)
        
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
        
        # Create distinct coaching personalities based on style
        if self.coaching_style == "scientific":
            personality_prompt = """You are Dr. GameMaster, a professional gaming strategist and cognitive scientist specializing in competitive game theory. You speak with authority about mathematical concepts, statistical analysis, and advanced strategic principles. You use precise terminology, reference academic concepts, and provide detailed analytical breakdowns. Your tone is professional, insightful, and intellectually stimulating."""
            
            coaching_approach = """
COACHING APPROACH: Provide comprehensive, data-driven analysis with:
• Deep statistical insights with specific numerical targets
• Game theory and information theory explanations  
• Advanced strategic concepts and terminology
• Detailed step-by-step optimization recommendations
• Each tip should be 2-3 sentences with specific metrics and reasoning
• Reference entropy thresholds, Nash equilibrium principles, and exploitability theory"""
            
        else:  # easy style
            personality_prompt = """You are Coach Sam, a friendly and enthusiastic Rock-Paper-Scissors mentor who loves helping players improve. You speak in an encouraging, accessible way using simple language and gaming metaphors. You're like a supportive friend who happens to be really good at strategy games. Your tone is warm, motivational, and easy to understand."""
            
            coaching_approach = """
COACHING APPROACH: Provide encouraging, practical guidance with:
• Clear, actionable advice using everyday language
• Encouraging motivation and positive reinforcement
• Simple explanations of complex concepts using analogies
• Fun, engaging tips that build confidence
• Each tip should be 2-3 sentences focusing on practical actions and encouragement
• Use gaming metaphors and friendly language"""
        
        natural_prompt = f"""{personality_prompt}

{coaching_approach}

PLAYER'S CURRENT SITUATION:
• Round {full_context['current_round']} with {full_context['win_rate']:.1%} win rate
• Recent moves: {full_context['recent_moves']}
• Entropy (randomness): {full_context['entropy']:.3f} out of 1.585 maximum
• Predictability: {full_context['predictability']:.3f} (lower is better)  
• Nash distance: {full_context['nash_distance']:.3f} (measures optimal play balance)
• AI opponent using: {full_context['robot_strategy']} strategy
• Decision complexity: {full_context['decision_complexity']:.3f}
• Pattern consistency: {full_context['strategy_consistency']:.3f}

DETAILED ANALYSIS CONTEXT:
The player's move distribution shows {full_context['move_distribution']}. Their recent performance trend is {full_context.get('momentum', {})}. Current psychological indicators suggest {full_context.get('impulsiveness', 0.5):.3f} impulsiveness level and {full_context.get('consistency_score', 0.5):.3f} consistency score.

Please provide comprehensive coaching advice with 4-5 detailed tips that include:
1. Immediate tactical recommendations for the next 2-3 moves (be specific about which moves to consider)
2. Pattern analysis insights about their current playing style (reference the specific metrics)
3. Strategic optimization advice for improving randomness and unpredictability (give specific targets)
4. Counter-strategy recommendations for the AI's {full_context['robot_strategy']} approach (explain the reasoning)
5. Long-term strategic development advice based on their performance profile

Make each tip substantial (2-3 sentences) with specific reasoning, and maintain your coaching personality throughout. Reference the actual numbers and explain what they mean in your coaching style."""

        try:
            # Use safe LLM call method
            response_text = self._safe_llm_call(natural_prompt)
            
            print(f"📝 Real LLM Response Preview: {str(response_text)[:100]}...")
            
            # Parse natural language response
            parsed_response = self.output_parser.parse(str(response_text))
            
            return {
                'tips': parsed_response.tips,
                'insights': parsed_response.insights,
                'educational_content': parsed_response.educational_content,
                'behavioral_analysis': parsed_response.behavioral_analysis,
                'confidence_level': parsed_response.confidence_level,
                'response_type': parsed_response.response_type,
                'llm_type': type(self.llm).__name__,
                'metrics_count': len(full_context),
                'raw_response': str(response_text)[:300],  # More text for natural language
                'natural_language_full': str(response_text),  # Store full response
                'coaching_personality': 'Dr. GameMaster (Scientific)' if self.coaching_style == 'scientific' else 'Coach Sam (Friendly)'
            }
            
        except Exception as e:
            print(f"⚠️ Real LLM coaching failed: {e}")
            # Fallback to a simple natural language attempt
            try:
                style_instruction = "scientific and detailed" if self.coaching_style == "scientific" else "easy-to-understand and encouraging"
                simple_prompt = f"Give coaching advice for rock-paper-scissors. The player has {full_context['win_rate']:.1%} win rate and entropy {full_context['entropy']:.3f}. Be helpful and {style_instruction}."
                fallback_response = self._safe_llm_call(simple_prompt)
                
                parsed_fallback = self.output_parser.parse(str(fallback_response))
                return {
                    'tips': parsed_fallback.tips,
                    'insights': {'error_recovery': 'Used simplified prompt due to parsing issues'},
                    'educational_content': parsed_fallback.educational_content,
                    'behavioral_analysis': parsed_fallback.behavioral_analysis,
                    'confidence_level': 0.6,
                    'response_type': 'fallback_natural_language',
                    'llm_type': type(self.llm).__name__,
                    'raw_response': str(fallback_response)[:200]
                }
            except Exception as e2:
                print(f"⚠️ Fallback also failed: {e2}")
                return self._generate_fallback_advice({}, 'real_time')
    
    def _generate_comprehensive_analysis(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive post-game analysis"""
        
        # Extract comprehensive metrics
        core = metrics.get('core_game', {})
        patterns = metrics.get('patterns', {})
        performance = metrics.get('performance', {})
        psychological = metrics.get('psychological', {})
        ai_behavior = metrics.get('ai_behavior', {})
        
        # Create session data summary
        session_data = f"""
Total Rounds: {core.get('current_round', 0)}
Performance: {core.get('win_rates', {}).get('human', 0.0):.1%} win rate
Pattern Analysis: {patterns.get('pattern_type', 'unknown')} patterns detected
Entropy: {patterns.get('entropy_calculation', 0.0):.3f}
Predictability: {patterns.get('predictability_score', 0.0):.3f}
AI Strategy: {ai_behavior.get('current_strategy', 'unknown')}
Decision Style: {psychological.get('decision_making_style', {})}
"""
        
        # Use MockLLM directly for fast, reliable responses
        if isinstance(self.llm, MockLLM):
            # Create simple prompt for MockLLM comprehensive analysis
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
                # Fall back to MockLLM approach
                mock_llm = MockLLM()
                mock_llm.set_coaching_style(self.coaching_style)
                prompt_text = f"Comprehensive analysis for {self.coaching_style} style:\n{session_data}"
                response_text = mock_llm(prompt_text)
        
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