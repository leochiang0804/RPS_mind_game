#!/usr/bin/env python3
"""
Trained RPS Coach Model Wrapper

Integrates the custom-trained RPS coaching model into the main application.
Converts comprehensive metrics to the 4-metric format the model expects.
"""

import os
import sys
import time
import torch
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add model_training directory to path
model_training_path = os.path.join(os.path.dirname(__file__), 'model_training')
sys.path.append(model_training_path)

try:
    from simple_train_coach import SimpleRPSCoachModel
except ImportError as e:
    print(f"⚠️ Could not import trained model: {e}")
    SimpleRPSCoachModel = None


class TrainedCoachWrapper:
    """Wrapper for the trained RPS coaching model"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.vocab = None
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.coaching_style = 'easy'
        
        # Default model path
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__), 
                'model_training', 
                'models', 
                'simple_rps_coach.pth'
            )
        
        self.model_path = model_path
        self._load_model()
    
    def _load_model(self):
        """Load the trained model"""
        try:
            if not os.path.exists(self.model_path):
                print(f"⚠️ Model file not found: {self.model_path}")
                return False
            
            if SimpleRPSCoachModel is None:
                print("⚠️ SimpleRPSCoachModel class not available")
                return False
            
            # Initialize model architecture
            self.model = SimpleRPSCoachModel(vocab_size=40, hidden_size=256)
            
            # Load trained weights
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Load vocabulary from checkpoint
            self.vocab = checkpoint.get('vocab', self._build_vocab())
            
            print(f"✅ Trained model loaded successfully from {self.model_path}")
            print(f"   Device: {self.device}")
            print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load trained model: {e}")
            return False
    
    def _build_vocab(self) -> Dict[str, int]:
        """Build vocabulary (same as training)"""
        vocab = {"<pad>": 0, "<unk>": 1, "<start>": 2, "<end>": 3}
        
        rps_words = [
            "rock", "paper", "scissors", "pattern", "strategy", "win", "lose", "tie",
            "entropy", "random", "predictable", "adapt", "change", "repeat", "cycle",
            "frequency", "markov", "ai", "player", "game", "move", "sequence", "advice",
            "coaching", "tip", "suggestion", "observation", "analysis", "behavior",
            "psychology", "tendency", "bias", "counter", "effective", "improve", "learn"
        ]
        
        for word in rps_words:
            if word not in vocab:
                vocab[word] = len(vocab)
                
        return vocab
    
    def _convert_comprehensive_metrics_to_simple(self, comprehensive_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Convert 65+ comprehensive metrics to the 4 metrics the model expects"""
        
        # Extract core metrics
        core = comprehensive_metrics.get('core_game', {})
        patterns = comprehensive_metrics.get('patterns', {})
        advanced = comprehensive_metrics.get('advanced', {})
        
        # Calculate entropy (randomness measure)
        entropy = patterns.get('entropy', 1.0)
        if entropy is None or entropy == 0:
            entropy = 1.0
        
        # Get win rate
        win_rates = core.get('win_rates', {})
        win_rate = win_rates.get('human', 0.5)
        if win_rate is None:
            win_rate = 0.5
        
        # Calculate pattern strength (how predictable the player is)
        pattern_strength = 1.0 - min(entropy / 1.585, 1.0)  # Inverse of normalized entropy
        
        # Get adaptation rate (how much player is adapting)
        adaptation_rate = advanced.get('complexity_metrics', {}).get('adaptation_rate', 0.5)
        if adaptation_rate is None:
            adaptation_rate = 0.5
        
        return {
            'entropy': float(entropy),
            'win_rate': float(win_rate),
            'pattern_strength': float(pattern_strength),
            'adaptation_rate': float(adaptation_rate)
        }
    
    def _extract_basic_metrics_from_game_state(self, game_state: Dict[str, Any]) -> Dict[str, float]:
        """Extract basic metrics directly from game state if comprehensive metrics not available"""
        
        human_moves = game_state.get('human_moves', [])
        results = game_state.get('results', [])
        
        if not human_moves:
            return {
                'entropy': 1.0,
                'win_rate': 0.5,
                'pattern_strength': 0.5,
                'adaptation_rate': 0.5
            }
        
        # Calculate basic entropy
        move_counts = {'rock': 0, 'paper': 0, 'scissors': 0}
        for move in human_moves[-10:]:  # Last 10 moves
            if move in move_counts:
                move_counts[move] += 1
        
        total = sum(move_counts.values())
        if total == 0:
            entropy = 1.0
        else:
            probs = [count/total for count in move_counts.values() if count > 0]
            entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        
        # Calculate win rate
        wins = results.count('win') if results else 0
        total_games = len(results) if results else 1
        win_rate = wins / total_games
        
        # Simple pattern strength (based on entropy)
        pattern_strength = 1.0 - min(entropy / 1.585, 1.0)
        
        # Simple adaptation rate (based on recent performance change)
        recent_results = results[-5:] if len(results) >= 5 else results
        early_results = results[:5] if len(results) >= 10 else results[:len(results)//2] if results else []
        
        if early_results and recent_results:
            early_win_rate = early_results.count('win') / len(early_results)
            recent_win_rate = recent_results.count('win') / len(recent_results)
            adaptation_rate = abs(recent_win_rate - early_win_rate)
        else:
            adaptation_rate = 0.5
        
        return {
            'entropy': float(entropy),
            'win_rate': float(win_rate),
            'pattern_strength': float(pattern_strength),
            'adaptation_rate': float(adaptation_rate)
        }
    
    def generate_coaching_advice(self, comprehensive_metrics: Optional[Dict[str, Any]] = None, 
                               game_state: Optional[Dict[str, Any]] = None,
                               coaching_type: str = 'real_time') -> Dict[str, Any]:
        """Generate coaching advice using the trained model"""
        
        if self.model is None:
            return {
                'error': 'Trained model not available',
                'tips': ['Model not loaded - check model file'],
                'insights': {},
                'confidence_level': 0.0,
                'response_type': 'error'
            }
        
        try:
            start_time = time.time()
            
            # Convert metrics to simple format
            if comprehensive_metrics:
                simple_metrics = self._convert_comprehensive_metrics_to_simple(comprehensive_metrics)
            elif game_state:
                simple_metrics = self._extract_basic_metrics_from_game_state(game_state)
            else:
                # Default metrics
                simple_metrics = {
                    'entropy': 1.0,
                    'win_rate': 0.5,
                    'pattern_strength': 0.5,
                    'adaptation_rate': 0.5
                }
            
            # Prepare model input
            dummy_text = "player strategy analysis coaching"
            words = dummy_text.split()
            if self.vocab:
                token_ids = [self.vocab.get(word, 1) for word in words]
            else:
                token_ids = [1] * len(words)  # Use unknown tokens if vocab not loaded
            input_tokens = torch.tensor([token_ids], dtype=torch.long).to(self.device)
            
            # Create metrics tensor
            metrics_tensor = torch.tensor([[
                simple_metrics['entropy'],
                simple_metrics['win_rate'],
                simple_metrics['pattern_strength'], 
                simple_metrics['adaptation_rate']
            ]], dtype=torch.float32).to(self.device)
            
            # Generate prediction
            with torch.no_grad():
                outputs = self.model(input_tokens, metrics_tensor)
                predicted_category = int(torch.argmax(outputs, dim=-1).item())
                confidence = float(torch.softmax(outputs, dim=-1).max().item())
            
            # Convert prediction to coaching advice
            coaching_advice = self._category_to_advice(predicted_category, simple_metrics)
            
            response_time = time.time() - start_time
            
            return {
                'tips': coaching_advice['tips'],
                'insights': {
                    'predicted_category': predicted_category,
                    'category_name': coaching_advice['category_name'],
                    'model_confidence': confidence,
                    'input_metrics': simple_metrics,
                    'coaching_focus': coaching_advice['focus']
                },
                'educational_content': {
                    'focus_area': coaching_advice['educational_focus'],
                    'learning_point': coaching_advice['learning_point'],
                    'theory_connection': coaching_advice['theory']
                },
                'behavioral_analysis': {
                    'player_tendency': coaching_advice['tendency'],
                    'improvement_area': coaching_advice['improvement'],
                    'strategic_assessment': coaching_advice['assessment']
                },
                'confidence_level': confidence,
                'response_type': f'trained_model_{self.coaching_style}',
                'llm_type': 'TrainedRPSCoach',
                'performance': {
                    'response_time_ms': response_time * 1000,
                    'model_size_mb': 9.0,
                    'inference_speed': 'fast'
                },
                'natural_language_full': coaching_advice['natural_language']
            }
            
        except Exception as e:
            print(f"❌ Trained model inference failed: {e}")
            return {
                'error': f'Model inference failed: {str(e)}',
                'tips': ['Try again - model had an issue'],
                'insights': {},
                'confidence_level': 0.0,
                'response_type': 'error'
            }
    
    def _category_to_advice(self, category: int, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Convert model prediction category to detailed coaching advice"""
        
        win_rate = metrics['win_rate']
        entropy = metrics['entropy']
        pattern_strength = metrics['pattern_strength']
        
        if category == 0:  # Vary strategy
            return {
                'category_name': 'Strategy Variation',
                'tips': [
                    f"Your current strategy shows patterns (strength: {pattern_strength:.2f})",
                    "Try varying your move selection more randomly",
                    "Break up predictable sequences to confuse the AI"
                ],
                'focus': 'Increasing unpredictability',
                'educational_focus': 'Randomization Theory',
                'learning_point': f"Your entropy ({entropy:.3f}) shows room for more randomness",
                'theory': 'Game theory suggests optimal play involves unpredictable mixed strategies',
                'tendency': 'Shows some predictable patterns',
                'improvement': 'Focus on random timing and move selection',
                'assessment': f"Pattern strength {pattern_strength:.2f} indicates exploitable tendencies",
                'natural_language': f"I notice you have some predictable patterns in your play (pattern strength: {pattern_strength:.2f}). The AI might be exploiting this, which could explain your {win_rate:.1%} win rate. Try varying your strategy more - mix up your timing and create false patterns to keep the AI guessing."
            }
        
        elif category == 1:  # Pattern detection
            return {
                'category_name': 'Pattern Breaking',
                'tips': [
                    f"Strong pattern detected (strength: {pattern_strength:.2f}) - AI likely adapted",
                    "Break your current pattern immediately",
                    "Create deliberate false patterns to mislead the AI"
                ],
                'focus': 'Pattern disruption',
                'educational_focus': 'Pattern Recognition Theory',
                'learning_point': f"High pattern strength ({pattern_strength:.2f}) makes you vulnerable",
                'theory': 'Predictable patterns allow opponents to exploit your tendencies',
                'tendency': 'Strong pattern formation',
                'improvement': 'Immediate pattern breaking',
                'assessment': f"Win rate {win_rate:.1%} suggests AI has learned your patterns",
                'natural_language': f"I've detected a strong pattern in your play (strength: {pattern_strength:.2f}). With your current {win_rate:.1%} win rate, it's likely the AI has caught on to this pattern and is exploiting it. You need to break this pattern immediately - try completely different sequences or even create false patterns to throw off the AI's predictions."
            }
        
        elif category == 2:  # Randomness encouragement
            return {
                'category_name': 'Randomness Excellence',
                'tips': [
                    f"Excellent randomness! Your entropy is {entropy:.3f}",
                    "Keep up the unpredictable play style",
                    "Your move distribution is working well"
                ],
                'focus': 'Maintaining randomness',
                'educational_focus': 'Optimal Randomization',
                'learning_point': f"Your entropy ({entropy:.3f}) shows excellent unpredictability",
                'theory': 'High entropy play approaches Nash equilibrium strategy',
                'tendency': 'Excellent random play',
                'improvement': 'Maintain current approach',
                'assessment': f"Win rate {win_rate:.1%} with high randomness shows good strategic play",
                'natural_language': f"Great job! Your play shows excellent randomness (entropy: {entropy:.3f}) which is exactly what you want in Rock-Paper-Scissors. With a {win_rate:.1%} win rate and low pattern strength ({pattern_strength:.2f}), you're playing strategically sound. Keep up this unpredictable style - it's making you very difficult for the AI to read and exploit."
            }
        
        elif category == 3:  # Adaptation coaching
            return {
                'category_name': 'Counter-Adaptation',
                'tips': [
                    f"AI is adapting (your adaptation rate: {metrics['adaptation_rate']:.2f})",
                    "Counter-adapt by changing your meta-strategy",
                    "Switch between different playing styles"
                ],
                'focus': 'Strategic adaptation',
                'educational_focus': 'Adaptive Game Theory',
                'learning_point': f"Adaptation rate {metrics['adaptation_rate']:.2f} shows dynamic play",
                'theory': 'In adaptive games, meta-strategy evolution is crucial',
                'tendency': 'Adaptive player',
                'improvement': 'Meta-level strategy switching',
                'assessment': f"Win rate {win_rate:.1%} suggests need for better counter-adaptation",
                'natural_language': f"The AI is actively adapting to your play, and your adaptation rate of {metrics['adaptation_rate']:.2f} shows you're trying to counter it. With a {win_rate:.1%} win rate, you need to think one level higher - don't just change your moves, change your entire approach. Switch between different playing philosophies to stay ahead of the AI's learning."
            }
        
        else:  # General advice (category 4)
            return {
                'category_name': 'General Strategy',
                'tips': [
                    "Focus on reading the AI's response patterns",
                    "Pay attention to timing and rhythm",
                    "Stay mentally engaged and focused"
                ],
                'focus': 'Overall improvement',
                'educational_focus': 'Game Awareness',
                'learning_point': f"Win rate {win_rate:.1%} with balanced metrics shows steady play",
                'theory': 'Consistent focus and pattern recognition are key to improvement',
                'tendency': 'Balanced strategic approach',
                'improvement': 'Tactical awareness',
                'assessment': f"Entropy {entropy:.3f} and pattern strength {pattern_strength:.2f} show room for optimization",
                'natural_language': f"Your overall play is showing balanced metrics - win rate of {win_rate:.1%}, entropy of {entropy:.3f}, and pattern strength of {pattern_strength:.2f}. Focus on staying mentally engaged and reading how the AI responds to your moves. Pay attention to subtle timing changes and rhythm patterns that might give you an edge."
            }
    
    def set_coaching_style(self, style: str):
        """Set coaching style preference"""
        self.coaching_style = style
    
    def get_llm_type(self) -> str:
        """Return the LLM type identifier"""
        return "trained_model"
    
    def is_available(self) -> bool:
        """Check if the trained model is available"""
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model"""
        if self.model is None:
            return {'available': False, 'error': 'Model not loaded'}
        
        return {
            'available': True,
            'model_path': self.model_path,
            'device': str(self.device),
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'vocab_size': len(self.vocab) if self.vocab else 0,
            'model_size_mb': 9.0,
            'input_metrics': ['entropy', 'win_rate', 'pattern_strength', 'adaptation_rate'],
            'output_categories': 5,
            'training_examples': 2000
        }


# Test function
def test_trained_coach():
    """Test the trained coach wrapper"""
    print("🧪 Testing Trained Coach Wrapper...")
    
    coach = TrainedCoachWrapper()
    
    if not coach.is_available():
        print("❌ Trained coach not available")
        return False
    
    # Test with sample metrics
    test_metrics = {
        'core_game': {
            'win_rates': {'human': 0.3}
        },
        'patterns': {
            'entropy': 0.8
        },
        'advanced': {
            'complexity_metrics': {
                'adaptation_rate': 0.4
            }
        }
    }
    
    advice = coach.generate_coaching_advice(comprehensive_metrics=test_metrics)
    
    print(f"✅ Generated advice: {advice.get('tips', ['No tips'])[0]}")
    print(f"   Confidence: {advice.get('confidence_level', 0):.2f}")
    print(f"   Response time: {advice.get('performance', {}).get('response_time_ms', 0):.1f}ms")
    
    return True


if __name__ == "__main__":
    test_trained_coach()