"""
LSTM Web Integration Module
Provides LSTM model training and prediction capabilities for the web application.
"""

import os
import sys
import json
import random
from typing import List, Dict, Optional, Any
import torch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lstm_unified import LSTMManager

import os
import json
import torch
import numpy as np
from typing import Dict, List, Optional
from lstm_model import TinyLSTM
from move_mapping import normalize_move

class LSTMPredictor:
    """Web-compatible LSTM predictor"""
    
    def __init__(self, model_path: str = "models/lstm/lstm_rps.pth", metadata_path: str = "models/lstm/lstm_rps_metadata.json"):
        self.model = None
        self.metadata = None
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.is_loaded = False
        
    def load_model(self) -> bool:
        """Load the trained LSTM model"""
        try:
            # Check if files exist
            if not os.path.exists(self.model_path):
                print(f"LSTM model not found at {self.model_path}")
                return False
                
            if not os.path.exists(self.metadata_path):
                print(f"LSTM metadata not found at {self.metadata_path}")
                return False
            
            # Load metadata
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            # Create model with saved parameters
            self.model = TinyLSTM(
                vocab_size=self.metadata['vocab_size'],
                embed_dim=self.metadata['embed_dim'],
                hidden_dim=self.metadata['hidden_dim'],
                num_layers=self.metadata['num_layers']
            )
            
            # Load trained weights
            self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
            self.model.eval()
            
            self.is_loaded = True
            print("✅ LSTM model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load LSTM model: {e}")
            return False
    
    def predict(self, move_history: List[str]) -> Dict[str, float]:
        """
        Predict next move probabilities
        Returns standard format (rock/paper/scissors)
        """
        if not self.is_loaded:
            if not self.load_model():
                # Fallback to random predictions in standard format
                return {'rock': 0.33, 'paper': 0.33, 'scissors': 0.34}
        
        try:
            # Normalize input moves and get predictions
            normalized_history = [normalize_move(move) for move in move_history]
            probs = self.model.predict_next_move(normalized_history)
            
            # Return in standard format (the model already returns rock/paper/scissors)
            return probs
        except Exception as e:
            print(f"LSTM prediction error: {e}")
            return {'rock': 0.33, 'paper': 0.33, 'scissors': 0.34}
    
    def get_confidence(self, move_history: List[str]) -> float:
        """Get confidence level for predictions"""
        if not self.is_loaded or len(move_history) < 2:
            return 0.33
        
        try:
            probs = self.predict(move_history)
            # Confidence is the maximum probability
            return max(probs.values())
        except:
            return 0.33
    
    def get_counter_move(self, move_history: List[str]) -> str:
        """Get robot's counter move based on predicted human move"""
        if not self.is_loaded:
            if not self.load_model():
                return random.choice(['rock', 'paper', 'scissors'])
        
        try:
            probs = self.predict(move_history)
            # Get most likely human move
            predicted_human_move = max(probs.items(), key=lambda x: x[1])[0]
            
            # Return counter move (what beats the predicted human move)
            counters = {'rock': 'paper', 'paper': 'scissors', 'scissors': 'rock'}
            counter_move = counters.get(predicted_human_move)
            return counter_move if counter_move is not None else random.choice(['rock', 'paper', 'scissors'])
        except Exception as e:
            print(f"LSTM counter move error: {e}")
            return random.choice(['rock', 'paper', 'scissors'])
    
    def train_on_games(self, game_sequences: List[List[str]], epochs: int = 20) -> bool:
        """Train the model on new game sequences"""
        if not self.is_loaded or self.model is None:
            if not self.load_model():
                return False
        
        try:
            trainer = LSTMTrainer(self.model, learning_rate=0.0001)  # Lower LR for fine-tuning
            trainer.train_on_sequences(game_sequences, epochs=epochs)
            
            # Save updated model
            torch.save(self.model.state_dict(), self.model_path)
            print(f"Model updated and saved to {self.model_path}")
            return True
        except Exception as e:
            print(f"Training error: {e}")
            return False
    
    def get_model_info(self) -> Dict:
        """Get model information for debugging"""
        if not self.metadata:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded" if self.is_loaded else "error",
            "model_type": self.metadata.get('model_type', 'unknown'),
            "created_at": self.metadata.get('created_at', 'unknown'),
            "training_data_size": self.metadata.get('training_data_size', 0),
            "parameters": {
                "vocab_size": self.metadata.get('vocab_size', 3),
                "embed_dim": self.metadata.get('embed_dim', 8),
                "hidden_dim": self.metadata.get('hidden_dim', 24),
                "num_layers": self.metadata.get('num_layers', 2)
            }
        }

# Global instance for web app
lstm_manager = LSTMManager()

def get_lstm_predictor() -> LSTMManager:
    """Get an instance of the unified LSTM manager for web integration"""
    return lstm_manager

def init_lstm_model() -> bool:
    """Initialize LSTM model for web app (loads both PyTorch and ONNX if available)"""
    try:
        lstm_manager.load_model()
        print("✅ PyTorch LSTM model loaded for web app")
        
        # Try to load ONNX for faster inference
        try:
            lstm_manager.load_onnx_model()
            print("✅ ONNX LSTM model loaded for web app")
        except Exception as e:
            print(f"⚠️ ONNX not available, using PyTorch: {e}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to initialize LSTM: {e}")
        return False

if __name__ == "__main__":
    # Test the web integration
    print("🧪 Testing LSTM Web Integration")
    print("=" * 40)
    
    predictor = LSTMPredictor()
    
    if predictor.load_model():
        print("✅ Model loaded successfully")
        
        # Test predictions
        test_history = ['rock', 'rock', 'paper', 'paper', 'scissors']
        
        probs = predictor.predict(test_history)
        print(f"Predictions: {probs}")
        
        confidence = predictor.get_confidence(test_history)
        print(f"Confidence: {confidence:.3f}")
        
        counter = predictor.get_counter_move(test_history)
        print(f"Counter move: {counter}")
        
        info = predictor.get_model_info()
        print(f"Model info: {info}")
        
    else:
        print("❌ Failed to load model")
        print("Run 'python lstm_model.py' first to create the model")