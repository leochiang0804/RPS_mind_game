"""
LSTM Web Integration for Flask App
Provides a web-compatible interface for the LSTM model
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Optional
from lstm_model import TinyLSTM

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
        Compatible with existing strategy interface
        """
        if not self.is_loaded:
            if not self.load_model():
                # Fallback to random predictions
                return {'stone': 0.33, 'paper': 0.33, 'scissor': 0.34}
        
        try:
            return self.model.predict_next_move(move_history)
        except Exception as e:
            print(f"LSTM prediction error: {e}")
            return {'stone': 0.33, 'paper': 0.33, 'scissor': 0.34}
    
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
        """Get the robot's counter move based on prediction"""
        if not move_history:
            return 'paper'  # Default
        
        try:
            # Get human move predictions
            probs = self.predict(move_history)
            
            # Find most likely human move
            predicted_human_move = max(probs.items(), key=lambda x: x[1])[0]
            
            # Return counter move
            counters = {
                'stone': 'paper',
                'paper': 'scissor', 
                'scissor': 'stone'
            }
            
            return counters.get(predicted_human_move, 'paper')
            
        except Exception as e:
            print(f"Counter move error: {e}")
            return 'paper'
    
    def fine_tune(self, recent_moves: List[str], epochs: int = 5):
        """
        Fine-tune the model on recent moves
        Called every 5 rounds as per the development plan
        """
        if not self.is_loaded or len(recent_moves) < 6:
            return
        
        try:
            from lstm_model import LSTMTrainer
            trainer = LSTMTrainer(self.model, learning_rate=0.0001)  # Lower LR for fine-tuning
            trainer.fine_tune(recent_moves, epochs=epochs, seq_length=5)
            
            # Save updated model
            torch.save(self.model.state_dict(), self.model_path)
            print(f"✅ LSTM model fine-tuned on {len(recent_moves)} recent moves")
            
        except Exception as e:
            print(f"Fine-tuning error: {e}")
    
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
lstm_predictor = LSTMPredictor()

def get_lstm_predictor() -> LSTMPredictor:
    """Get the global LSTM predictor instance"""
    return lstm_predictor

def init_lstm_model() -> bool:
    """Initialize LSTM model for web app"""
    return lstm_predictor.load_model()

if __name__ == "__main__":
    # Test the web integration
    print("🧪 Testing LSTM Web Integration")
    print("=" * 40)
    
    predictor = LSTMPredictor()
    
    if predictor.load_model():
        print("✅ Model loaded successfully")
        
        # Test predictions
        test_history = ['stone', 'stone', 'paper', 'paper', 'scissor']
        
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