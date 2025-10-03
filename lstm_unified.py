#!/usr/bin/env python3
"""
Comprehensive LSTM Training and Management System
Handles all LSTM model operations: training, optimization, loading, and inference.
Consolidates lstm_model.py, lstm_optimizer.py, and optimized_lstm_loader.py.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import random
import onnx
import onnxruntime as ort
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime

class TinyLSTM(nn.Module):
    """
    Basic LSTM model for RPS prediction
    Architecture: configurable embed/hidden/layers
    """
    
    def __init__(self, vocab_size=3, embed_dim=8, hidden_dim=24, num_layers=2, dropout=0.1):
        super(TinyLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use last timestep
        last_output = lstm_out[:, -1, :]
        
        # Project to vocabulary
        output = self.output_proj(last_output)
        
        return output

class EnhancedLSTM(nn.Module):
    """Enhanced LSTM model with attention, pattern recognition, and ensemble prediction"""
    
    def __init__(self, vocab_size=3, embed_dim=16, hidden_dim=48, num_layers=2, dropout=0.1):
        super(EnhancedLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional encoding for sequence awareness
        self.positional_encoding = nn.Parameter(torch.randn(1000, embed_dim) * 0.1)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Multi-head attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # Bidirectional
            num_heads=2,
            dropout=dropout,
            batch_first=True
        )
        
        # Pattern recognition with 1D convolution
        self.pattern_conv1d = nn.Conv1d(
            in_channels=hidden_dim * 2,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1
        )
        
        # Dual prediction heads for ensemble
        self.sequence_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        self.pattern_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, vocab_size)
        )
        
        # Ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(2) * 0.5)
        
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # Embedding with positional encoding
        embedded = self.embedding(x)
        if seq_len <= self.positional_encoding.size(0):
            embedded = embedded + self.positional_encoding[:seq_len].unsqueeze(0)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Pattern recognition via convolution
        conv_input = attn_out.transpose(1, 2)
        pattern_features = self.pattern_conv1d(conv_input)
        pattern_features = pattern_features.transpose(1, 2)
        
        # Use last timestep for prediction
        sequence_features = attn_out[:, -1, :]
        pattern_features = pattern_features[:, -1, :]
        
        # Dual predictions
        sequence_pred = self.sequence_head(sequence_features)
        pattern_pred = self.pattern_head(pattern_features)
        
        # Ensemble prediction
        weights = torch.softmax(self.ensemble_weights, dim=0)
        final_pred = weights[0] * sequence_pred + weights[1] * pattern_pred
        
        return final_pred

class LSTMTrainer:
    """Training utilities for LSTM models"""
    
    def __init__(self, model: nn.Module, learning_rate: float = 0.001):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.move_to_num = {'rock': 0, 'paper': 1, 'scissors': 2}
        self.num_to_move = {0: 'rock', 1: 'paper', 2: 'scissors'}
        
    def prepare_sequences(self, moves: List[str], sequence_length: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert move sequences to training data"""
        if len(moves) < sequence_length + 1:
            return torch.empty(0, sequence_length, dtype=torch.long), torch.empty(0, dtype=torch.long)
        
        # Convert moves to numbers
        move_numbers = [self.move_to_num.get(move.lower(), 0) for move in moves]
        
        sequences = []
        targets = []
        
        for i in range(len(move_numbers) - sequence_length):
            seq = move_numbers[i:i + sequence_length]
            target = move_numbers[i + sequence_length]
            sequences.append(seq)
            targets.append(target)
        
        return torch.tensor(sequences, dtype=torch.long), torch.tensor(targets, dtype=torch.long)
    
    def train_on_moves(self, moves: List[str], epochs: int = 100, sequence_length: int = 5) -> Dict[str, float]:
        """Train model on a sequence of moves"""
        sequences, targets = self.prepare_sequences(moves, sequence_length)
        
        if len(sequences) == 0:
            return {'loss': float('inf'), 'accuracy': 0.0}
        
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(sequences)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            with torch.no_grad():
                predicted = torch.argmax(outputs, dim=1)
                correct_predictions += (predicted == targets).sum().item()
                total_predictions += len(targets)
        
        avg_loss = total_loss / epochs
        accuracy = (correct_predictions / (total_predictions * epochs)) * 100 if total_predictions > 0 else 0
        
        return {'loss': avg_loss, 'accuracy': accuracy}

class LSTMManager:
    """Comprehensive LSTM model management system with ONNX support"""
    
    def __init__(self, model_dir: str = "models/lstm"):
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "lstm_rps.pth")
        self.onnx_path = os.path.join(model_dir, "lstm_rps.onnx")
        self.metadata_path = os.path.join(model_dir, "lstm_rps_metadata.json")
        self.backup_path = os.path.join(model_dir, "lstm_rps_backup.pth")
        self.backup_metadata_path = os.path.join(model_dir, "lstm_rps_backup_metadata.json")
        
        self.model = None
        self.onnx_session = None
        self.trainer = None
        self.metadata = None
        self.move_to_num = {'rock': 0, 'paper': 1, 'scissors': 2}
        self.num_to_move = {0: 'rock', 1: 'paper', 2: 'scissors'}
        
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)
    
    def create_tiny_lstm(self, vocab_size=3, embed_dim=8, hidden_dim=24, num_layers=2) -> TinyLSTM:
        """Create a basic TinyLSTM model"""
        return TinyLSTM(vocab_size, embed_dim, hidden_dim, num_layers)
    
    def create_enhanced_lstm(self, vocab_size=3, embed_dim=16, hidden_dim=48, num_layers=2) -> EnhancedLSTM:
        """Create an enhanced LSTM model with attention and pattern recognition"""
        return EnhancedLSTM(vocab_size, embed_dim, hidden_dim, num_layers)
    
    def load_model(self, use_enhanced: bool = True) -> bool:
        """Load existing model from disk"""
        try:
            if not os.path.exists(self.model_path) or not os.path.exists(self.metadata_path):
                print(f"❌ Model files not found at {self.model_path}")
                return False
            
            # Load metadata
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            # Create appropriate model type
            if use_enhanced or self.metadata.get('model_type') == 'enhanced_lstm_optimized':
                self.model = self.create_enhanced_lstm(
                    vocab_size=self.metadata.get('vocab_size', 3),
                    embed_dim=self.metadata.get('embed_dim', 16),
                    hidden_dim=self.metadata.get('hidden_dim', 48),
                    num_layers=self.metadata.get('num_layers', 2)
                )
            else:
                self.model = self.create_tiny_lstm(
                    vocab_size=self.metadata.get('vocab_size', 3),
                    embed_dim=self.metadata.get('embed_dim', 8),
                    hidden_dim=self.metadata.get('hidden_dim', 24),
                    num_layers=self.metadata.get('num_layers', 2)
                )
            
            # Load model weights
            model_state = torch.load(self.model_path, map_location='cpu')
            self.model.load_state_dict(model_state)
            self.model.eval()
            
            # Create trainer
            self.trainer = LSTMTrainer(self.model)
            
            print(f"✅ LSTM model loaded successfully")
            print(f"   Type: {self.metadata.get('model_type', 'basic')}")
            print(f"   Architecture: embed_dim={self.metadata.get('embed_dim')}, hidden_dim={self.metadata.get('hidden_dim')}, num_layers={self.metadata.get('num_layers')}")
            
            if 'performance' in self.metadata:
                perf = self.metadata['performance']
                print(f"   Performance: {perf.get('avg_win_rate', 0):.1f}% win rate, {perf.get('avg_accuracy', 0):.1f}% accuracy")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def save_model(self, model: nn.Module, metadata: Dict[str, Any]) -> bool:
        """Save model and metadata to disk"""
        try:
            # Backup existing model if it exists
            if os.path.exists(self.model_path):
                torch.save(torch.load(self.model_path), self.backup_path)
                print(f"📦 Backed up existing model to {self.backup_path}")
            
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    backup_metadata = json.load(f)
                with open(self.backup_metadata_path, 'w') as f:
                    json.dump(backup_metadata, f, indent=2)
                print(f"📦 Backed up existing metadata to {self.backup_metadata_path}")
            
            # Save new model
            torch.save(model.state_dict(), self.model_path)
            
            # Save metadata
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.model = model
            self.metadata = metadata
            self.trainer = LSTMTrainer(model)
            
            print(f"💾 Model saved successfully to {self.model_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save model: {e}")
            return False
    
    def export_to_onnx(self, model: Optional[nn.Module] = None, sequence_length: int = 10, use_simple_model: bool = True) -> bool:
        """Export PyTorch model to ONNX format for better deployment"""
        try:
            # Use provided model or loaded model
            if model is None:
                if self.model is None:
                    if not self.load_model():
                        print("❌ No model available for ONNX export")
                        return False
                model = self.model
            
            # For ONNX export, create a simplified wrapper model
            if use_simple_model and isinstance(model, EnhancedLSTM):
                print("🔄 Creating ONNX-compatible wrapper model...")
                
                # Create a simple wrapper that mimics the enhanced model's behavior
                class ONNXCompatibleLSTM(nn.Module):
                    def __init__(self, enhanced_model):
                        super().__init__()
                        self.enhanced_model = enhanced_model
                        
                    def forward(self, x):
                        # Use the enhanced model but return just the final prediction
                        with torch.no_grad():
                            return self.enhanced_model(x)
                
                # Test the enhanced model first to ensure it works
                test_input = torch.randint(0, 3, (1, sequence_length), dtype=torch.long)
                try:
                    with torch.no_grad():
                        test_output = model(test_input)
                        print(f"✅ Enhanced model test successful, output shape: {test_output.shape}")
                    
                    # Use the enhanced model directly but with stricter settings
                    export_model = model
                    
                except Exception as e:
                    print(f"⚠️ Enhanced model test failed: {e}")
                    # Fall back to creating a basic LSTM
                    export_model = self.create_tiny_lstm(
                        embed_dim=8, hidden_dim=24, num_layers=1  # Even simpler
                    )
            else:
                export_model = model
            
            # Set model to evaluation mode
            export_model.eval()
            
            # Create dummy input tensor (batch_size=1, sequence_length)
            dummy_input = torch.randint(0, 3, (1, sequence_length), dtype=torch.long)
            
            print(f"🔄 Exporting model to ONNX format...")
            print(f"   Model type: {type(export_model).__name__}")
            print(f"   Input shape: {dummy_input.shape}")
            print(f"   Sequence length: {sequence_length}")
            
            # For enhanced models, we need to be more careful with ONNX export
            if isinstance(export_model, EnhancedLSTM):
                # Try exporting with simplified options
                print("   Using enhanced model with simplified export settings...")
                torch.onnx.export(
                    export_model,
                    dummy_input,
                    self.onnx_path,
                    export_params=True,
                    opset_version=13,
                    do_constant_folding=False,  # Disable for complex models
                    input_names=['sequence'],
                    output_names=['predictions'],
                    verbose=False,
                    training=torch.onnx.TrainingMode.EVAL
                )
            else:
                # Standard export for simple models
                torch.onnx.export(
                    export_model,
                    dummy_input,
                    self.onnx_path,
                    export_params=True,
                    opset_version=13,
                    do_constant_folding=True,
                    input_names=['sequence'],
                    output_names=['predictions'],
                    dynamic_axes={
                        'sequence': {1: 'seq_length'},
                        'predictions': {0: 'batch_size'}
                    }
                )
            
            # Verify ONNX model
            try:
                onnx_model = onnx.load(self.onnx_path)
                onnx.checker.check_model(onnx_model)
                print(f"✅ ONNX model exported and verified: {self.onnx_path}")
                
                # Test ONNX model
                self._test_onnx_model(dummy_input)
                
                return True
                
            except Exception as e:
                print(f"⚠️ ONNX model verification failed: {e}")
                return False
                
        except Exception as e:
            print(f"❌ Failed to export to ONNX: {e}")
            return False
    
    def _test_onnx_model(self, test_input: torch.Tensor) -> bool:
        """Test ONNX model to ensure it works correctly"""
        try:
            # Create ONNX Runtime session
            ort_session = ort.InferenceSession(self.onnx_path)
            
            # Prepare input
            input_name = ort_session.get_inputs()[0].name
            ort_inputs = {input_name: test_input.numpy()}
            
            # Run inference
            ort_outputs = ort_session.run(None, ort_inputs)
            
            # Compare with PyTorch output
            if self.model is not None:
                self.model.eval()
                with torch.no_grad():
                    torch_output = self.model(test_input)
                
                # Check if outputs are close
                pytorch_probs = torch.softmax(torch_output[0], dim=0).numpy()
                onnx_probs = torch.softmax(torch.tensor(ort_outputs[0][0]), dim=0).numpy()
                
                if np.allclose(pytorch_probs, onnx_probs, atol=1e-4):
                    print("✅ ONNX model produces same results as PyTorch model")
                    return True
                else:
                    print("⚠️ ONNX model outputs differ from PyTorch model")
                    print(f"   PyTorch: {pytorch_probs}")
                    print(f"   ONNX: {onnx_probs}")
                    return False
            
            print("✅ ONNX model inference successful")
            return True
            
        except Exception as e:
            print(f"⚠️ ONNX model test failed: {e}")
            return False
    
    def load_onnx_model(self) -> bool:
        """Load ONNX model for inference"""
        try:
            if not os.path.exists(self.onnx_path):
                print(f"❌ ONNX model not found at {self.onnx_path}")
                return False
            
            # Load metadata
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            
            # Create ONNX Runtime session
            self.onnx_session = ort.InferenceSession(self.onnx_path)
            
            print(f"✅ ONNX model loaded successfully: {self.onnx_path}")
            if self.metadata:
                print(f"   Type: {self.metadata.get('model_type', 'unknown')}")
                print(f"   Architecture: embed_dim={self.metadata.get('embed_dim')}, hidden_dim={self.metadata.get('hidden_dim')}, num_layers={self.metadata.get('num_layers')}")
                
                if 'performance' in self.metadata:
                    perf = self.metadata['performance']
                    print(f"   Performance: {perf.get('avg_win_rate', 0):.1f}% win rate, {perf.get('avg_accuracy', 0):.1f}% accuracy")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load ONNX model: {e}")
            return False
    
    def predict_with_onnx(self, moves: List[str]) -> Dict[str, float]:
        """Predict using ONNX model (faster inference)"""
        if not self.onnx_session:
            if not self.load_onnx_model():
                return {'rock': 0.33, 'paper': 0.33, 'scissors': 0.34}
        
        if len(moves) < 2:
            return {'rock': 0.33, 'paper': 0.33, 'scissors': 0.34}
        
        try:
            # Convert moves to numbers
            move_numbers = [self.move_to_num.get(move.lower(), 0) for move in moves]
            
            # ONNX model expects exactly 5 moves - pad or truncate as needed
            sequence_length = 5
            if len(move_numbers) >= sequence_length:
                # Use the last 5 moves
                sequence = move_numbers[-sequence_length:]
            else:
                # Pad with 0s (rock) at the beginning if we don't have enough moves
                sequence = [0] * (sequence_length - len(move_numbers)) + move_numbers
            
            # Prepare input for ONNX
            input_array = np.array([sequence], dtype=np.int64)
            input_name = self.onnx_session.get_inputs()[0].name
            ort_inputs = {input_name: input_array}
            
            # Run inference
            ort_outputs = self.onnx_session.run(None, ort_inputs)
            
            # Convert output to probabilities
            logits = ort_outputs[0][0]
            probabilities = torch.softmax(torch.tensor(logits), dim=0).numpy()
            
            # Convert to dictionary
            result = {}
            for move, idx in self.move_to_num.items():
                result[move] = float(probabilities[idx])
            
            return result
            
        except Exception as e:
            print(f"⚠️ ONNX prediction error: {e}")
            return {'rock': 0.33, 'paper': 0.33, 'scissors': 0.34}
    
    def train_basic_model(self, training_moves: List[str], epochs: int = 1000, 
                         embed_dim: int = 8, hidden_dim: int = 24, num_layers: int = 2) -> Dict[str, Any]:
        """Train a basic TinyLSTM model"""
        print(f"🏋️ Training basic LSTM model...")
        print(f"   Architecture: embed_dim={embed_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}")
        print(f"   Training data: {len(training_moves)} moves")
        
        # Create model
        model = self.create_tiny_lstm(3, embed_dim, hidden_dim, num_layers)
        trainer = LSTMTrainer(model)
        
        # Train
        training_results = trainer.train_on_moves(training_moves, epochs)
        
        # Save model
        metadata = {
            'vocab_size': 3,
            'embed_dim': embed_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'model_type': 'basic_lstm',
            'training_moves': len(training_moves),
            'training_epochs': epochs,
            'training_results': training_results,
            'timestamp': datetime.now().isoformat()
        }
        
        self.save_model(model, metadata)
        return training_results
    
    def optimize_architecture(self) -> Dict[str, Any]:
        """Run architecture optimization to find best LSTM configuration"""
        print("🔬 Starting LSTM architecture optimization...")
        
        # Architecture configurations to test
        architectures = [
            {'embed_dim': 16, 'hidden_dim': 48, 'num_layers': 2},
            {'embed_dim': 16, 'hidden_dim': 64, 'num_layers': 3},
            {'embed_dim': 24, 'hidden_dim': 96, 'num_layers': 3},
            {'embed_dim': 32, 'hidden_dim': 128, 'num_layers': 4}
        ]
        
        # Generate synthetic training data
        print("📊 Generating synthetic training data...")
        training_data = self._generate_synthetic_training_data(4500)
        
        best_config = None
        best_performance = 0
        results = []
        
        for i, config in enumerate(architectures):
            print(f"\n🧪 Testing Architecture {i+1}/{len(architectures)}")
            print(f"   Config: {config}")
            
            # Create and train model
            model = self.create_enhanced_lstm(**config)
            trainer = LSTMTrainer(model, learning_rate=0.001)
            
            # Train on synthetic data
            train_results = trainer.train_on_moves(training_data, epochs=100)
            
            # Evaluate performance
            performance = self._evaluate_architecture(model, training_data)
            
            config_result = {
                'config': config,
                'training_loss': train_results['loss'],
                'training_accuracy': train_results['accuracy'],
                'evaluation_performance': performance
            }
            results.append(config_result)
            
            print(f"   Training Loss: {train_results['loss']:.4f}")
            print(f"   Training Accuracy: {train_results['accuracy']:.1f}%")
            print(f"   Win Rate: {performance['win_rate']:.1f}%")
            
            # Check if this is the best so far
            if performance['win_rate'] > best_performance:
                best_performance = performance['win_rate']
                best_config = config
                
                # Save the best model
                metadata = {
                    'vocab_size': 3,
                    'embed_dim': config['embed_dim'],
                    'hidden_dim': config['hidden_dim'],
                    'num_layers': config['num_layers'],
                    'model_type': 'enhanced_lstm_optimized',
                    'architecture_config': config,
                    'performance': performance,
                    'optimization_timestamp': datetime.now().isoformat()
                }
                
                self.save_model(model, metadata)
                print(f"   🏆 New best model saved!")
        
        print(f"\n🎯 Optimization Complete!")
        print(f"   Best Architecture: {best_config}")
        print(f"   Best Performance: {best_performance:.1f}% win rate")
        
        return {
            'best_config': best_config,
            'best_performance': best_performance,
            'all_results': results
        }
    
    def _generate_synthetic_training_data(self, num_moves: int) -> List[str]:
        """Generate synthetic training data with human-like patterns"""
        moves = ['rock', 'paper', 'scissors']
        synthetic_data = []
        
        # Different pattern types
        patterns = [
            # Cyclic patterns
            lambda i: moves[i % 3],
            lambda i: moves[(i * 2) % 3],
            
            # Frequency-based patterns
            lambda i: 'rock' if i % 4 == 0 else random.choice(moves),
            lambda i: 'paper' if i % 5 < 2 else random.choice(moves),
            
            # Alternating patterns
            lambda i: moves[0] if i % 2 == 0 else moves[1],
            lambda i: moves[i % 2],
            
            # Random with bias
            lambda i: random.choice(['rock', 'rock', 'paper', 'scissors']),
            lambda i: random.choice(moves),
        ]
        
        # Generate moves using different patterns
        current_pattern = patterns[0]  # Initialize with first pattern
        for i in range(num_moves):
            if i % 100 == 0:  # Change pattern every 100 moves
                current_pattern = random.choice(patterns)
            
            move = current_pattern(i)
            synthetic_data.append(move)
        
        return synthetic_data
    
    def _evaluate_architecture(self, model: nn.Module, test_data: List[str]) -> Dict[str, float]:
        """Evaluate architecture performance"""
        model.eval()
        
        wins = 0
        total = 0
        correct_predictions = 0
        
        counters = {'rock': 'paper', 'paper': 'scissors', 'scissors': 'rock'}
        
        # Use middle portion of test data for evaluation
        start_idx = len(test_data) // 4
        end_idx = 3 * len(test_data) // 4
        
        for i in range(start_idx, min(end_idx, len(test_data) - 5)):
            sequence = test_data[i:i+5]
            actual_next = test_data[i+5]
            
            # Convert to tensor
            seq_nums = [self.move_to_num[move] for move in sequence]
            input_tensor = torch.tensor([seq_nums], dtype=torch.long)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs[0], dim=0)
                predicted_idx = int(torch.argmax(probabilities).item())
                predicted_move = self.num_to_move[predicted_idx]
            
            # Check prediction accuracy
            if predicted_move == actual_next:
                correct_predictions += 1
            
            # Check if robot wins
            robot_move = counters[predicted_move]
            if (actual_next == 'rock' and robot_move == 'paper') or \
               (actual_next == 'paper' and robot_move == 'scissors') or \
               (actual_next == 'scissors' and robot_move == 'rock'):
                wins += 1
            
            total += 1
        
        win_rate = (wins / total * 100) if total > 0 else 0
        accuracy = (correct_predictions / total * 100) if total > 0 else 0
        
        return {
            'win_rate': win_rate,
            'accuracy': accuracy,
            'total_games': total
        }
    
    def predict(self, moves: List[str], use_onnx: bool = True) -> Dict[str, float]:
        """Predict next move probabilities (prefers ONNX if available)"""
        # Try ONNX first if requested and available
        if use_onnx and (self.onnx_session or os.path.exists(self.onnx_path)):
            try:
                return self.predict_with_onnx(moves)
            except Exception as e:
                print(f"⚠️ ONNX prediction failed, falling back to PyTorch: {e}")
        
        # Fallback to PyTorch model
        if not self.model:
            if not self.load_model():
                return {'rock': 0.33, 'paper': 0.33, 'scissors': 0.34}
        
        if len(moves) < 2:
            return {'rock': 0.33, 'paper': 0.33, 'scissors': 0.34}
        
        try:
            # Convert moves to numbers
            move_sequence = [self.move_to_num.get(move.lower(), 0) for move in moves]
            input_tensor = torch.tensor([move_sequence], dtype=torch.long)
            
            with torch.no_grad():
                if self.model is not None:
                    outputs = self.model(input_tensor)
                    probabilities = torch.softmax(outputs[0], dim=0)
                else:
                    # Default uniform distribution if no model
                    probabilities = torch.ones(3) / 3
            
            # Convert to dictionary
            result = {}
            for move, idx in self.move_to_num.items():
                result[move] = float(probabilities[idx])
            
            return result
            
        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            return {'rock': 0.33, 'paper': 0.33, 'scissors': 0.34}
    
    def get_winning_move(self, moves: List[str]) -> str:
        """Get the move that beats the predicted move"""
        predictions = self.predict(moves)
        predicted_move = max(predictions.keys(), key=lambda x: predictions[x])
        
        counters = {'rock': 'paper', 'paper': 'scissors', 'scissors': 'rock'}
        return counters.get(predicted_move, 'rock')

# Convenience functions for backward compatibility
def get_lstm_predictor() -> LSTMManager:
    """Get LSTM manager instance (replaces old get_lstm_predictor)"""
    manager = LSTMManager()
    manager.load_model()
    return manager

def get_optimized_lstm_predictor() -> LSTMManager:
    """Get optimized LSTM manager instance"""
    return get_lstm_predictor()

if __name__ == "__main__":
    print("🚀 LSTM Training and Management System")
    print("=" * 50)
    
    # Initialize manager
    manager = LSTMManager()
    
    print("\nAvailable commands:")
    print("1. Load existing model")
    print("2. Train basic model")
    print("3. Optimize architecture")
    print("4. Test prediction")
    print("5. Export to ONNX")
    print("6. Test ONNX model")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == "1":
        if manager.load_model():
            test_moves = ['rock', 'paper', 'scissors', 'rock', 'paper']
            predictions = manager.predict(test_moves)
            winning_move = manager.get_winning_move(test_moves)
            print(f"Test sequence: {test_moves}")
            print(f"Predictions: {predictions}")
            print(f"Recommended move: {winning_move}")
        
    elif choice == "2":
        # Generate some training data
        training_moves = []
        for i in range(500):
            if i % 3 == 0:
                training_moves.append('rock')
            elif i % 3 == 1:
                training_moves.append('paper')
            else:
                training_moves.append('scissors')
        
        results = manager.train_basic_model(training_moves, epochs=500)
        print(f"Training completed: {results}")
        
    elif choice == "3":
        optimization_results = manager.optimize_architecture()
        print("Optimization completed!")
        print(f"Best configuration: {optimization_results['best_config']}")
        print(f"Best performance: {optimization_results['best_performance']:.1f}%")
        
    elif choice == "4":
        if manager.load_model():
            while True:
                moves_input = input("Enter moves (comma-separated, or 'quit'): ").strip()
                if moves_input.lower() == 'quit':
                    break
                
                moves = [m.strip().lower() for m in moves_input.split(',')]
                predictions = manager.predict(moves)
                winning_move = manager.get_winning_move(moves)
                
                print(f"Sequence: {moves}")
                print(f"Predictions: {predictions}")
                print(f"Recommended move: {winning_move}")
                print()
    
    elif choice == "5":
        if manager.load_model():
            sequence_length = int(input("Enter sequence length for ONNX export (default 10): ") or "10")
            if manager.export_to_onnx(sequence_length=sequence_length):
                print("✅ Model successfully exported to ONNX!")
            else:
                print("❌ Failed to export model to ONNX")
        else:
            print("❌ No model available to export")
    
    elif choice == "6":
        if manager.load_onnx_model():
            while True:
                moves_input = input("Enter moves for ONNX test (comma-separated, or 'quit'): ").strip()
                if moves_input.lower() == 'quit':
                    break
                
                moves = [m.strip().lower() for m in moves_input.split(',')]
                
                # Test both PyTorch and ONNX
                pytorch_predictions = manager.predict(moves, use_onnx=False)
                onnx_predictions = manager.predict_with_onnx(moves)
                
                print(f"Sequence: {moves}")
                print(f"PyTorch predictions: {pytorch_predictions}")
                print(f"ONNX predictions: {onnx_predictions}")
                print()
        else:
            print("❌ ONNX model not available")
    
    else:
        print("Invalid choice!")