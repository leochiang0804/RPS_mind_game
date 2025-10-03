"""
Developer Metrics Console for Rock Paper Scissors Game
Comprehensive debugging, monitoring, and performance analysis interface.
"""

import json
import time
import os
import psutil
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from io import BytesIO
import base64
from typing import Dict, List, Tuple, Any, Optional

class PerformanceMonitor:
    """Real-time performance tracking for all game components."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = time.time()
        self.memory_baseline = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
    def track_inference_time(self, model_name: str, duration: float):
        """Track model inference timing."""
        self.metrics[f'{model_name}_inference_time'].append({
            'timestamp': time.time(),
            'duration': duration,
            'datetime': datetime.now().isoformat()
        })
        
    def track_memory_usage(self):
        """Track current memory usage."""
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.metrics['memory_usage'].append({
            'timestamp': time.time(),
            'memory_mb': current_memory,
            'memory_delta': current_memory - self.memory_baseline,
            'datetime': datetime.now().isoformat()
        })
        
    def track_prediction_accuracy(self, model_name: str, was_correct: bool):
        """Track prediction accuracy for each model."""
        self.metrics[f'{model_name}_accuracy'].append({
            'timestamp': time.time(),
            'correct': was_correct,
            'datetime': datetime.now().isoformat()
        })
        
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            'session_duration': time.time() - self.start_time,
            'total_metrics': len(self.metrics),
            'models_tracked': []
        }
        
        for key in self.metrics.keys():
            if '_inference_time' in key:
                model_name = key.replace('_inference_time', '')
                if model_name not in summary['models_tracked']:
                    summary['models_tracked'].append(model_name)
                    
        return summary

class ModelComparator:
    """Advanced model comparison and analysis tools."""
    
    def __init__(self):
        self.model_data = defaultdict(dict)
        
    def add_model_prediction(self, model_name: str, prediction: Any, confidence: float, 
                           actual_result: Optional[str] = None):
        """Add prediction data for analysis."""
        if model_name not in self.model_data:
            self.model_data[model_name] = {
                'predictions': [],
                'confidences': [],
                'accuracies': [],
                'last_updated': None
            }
            
        self.model_data[model_name]['predictions'].append(prediction)
        self.model_data[model_name]['confidences'].append(confidence)
        
        if actual_result:
            is_correct = prediction == actual_result if isinstance(prediction, str) else False
            self.model_data[model_name]['accuracies'].append(is_correct)
            
        self.model_data[model_name]['last_updated'] = datetime.now().isoformat()
        
    def generate_comparison_chart(self) -> str:
        """Generate base64 encoded comparison chart."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Model Performance Dashboard', fontsize=16)
            
            # 1. Accuracy Comparison
            model_names = list(self.model_data.keys())
            accuracies = []
            
            for model in model_names:
                if self.model_data[model]['accuracies']:
                    acc = sum(self.model_data[model]['accuracies']) / len(self.model_data[model]['accuracies']) * 100
                    accuracies.append(acc)
                else:
                    accuracies.append(0)
                    
            from matplotlib.colors import ListedColormap
            import matplotlib.pyplot as plt
            colors = plt.cm.get_cmap('Set3')(np.linspace(0, 1, len(model_names)))
            bars = ax1.bar(model_names, accuracies, color=colors)
            ax1.set_title('Model Accuracy Comparison')
            ax1.set_ylabel('Accuracy (%)')
            ax1.set_ylim(0, 100)
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{acc:.1f}%', ha='center', va='bottom')
            
            # 2. Confidence Distribution
            for i, model in enumerate(model_names):
                if self.model_data[model]['confidences']:
                    confidences = self.model_data[model]['confidences']
                    ax2.hist(confidences, alpha=0.7, label=model, color=colors[i], bins=20)
                    
            ax2.set_title('Confidence Distribution')
            ax2.set_xlabel('Confidence Level')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            
            # 3. Prediction Count
            pred_counts = [len(self.model_data[model]['predictions']) for model in model_names]
            wedges, texts, autotexts = ax3.pie(pred_counts, labels=model_names, autopct='%1.1f%%',
                                             colors=colors, startangle=90)
            ax3.set_title('Prediction Volume Distribution')
            
            # 4. Performance Timeline (last 50 predictions)
            for i, model in enumerate(model_names):
                if self.model_data[model]['accuracies']:
                    recent_acc = self.model_data[model]['accuracies'][-50:]  # Last 50
                    # Moving average
                    if len(recent_acc) >= 5:
                        moving_avg = []
                        for j in range(4, len(recent_acc)):
                            moving_avg.append(sum(recent_acc[j-4:j+1]) / 5 * 100)
                        ax4.plot(range(len(moving_avg)), moving_avg, 
                                label=f'{model} (5-game avg)', color=colors[i], linewidth=2)
                        
            ax4.set_title('Recent Performance Trend (5-Game Moving Average)')
            ax4.set_xlabel('Recent Games')
            ax4.set_ylabel('Accuracy (%)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plot_data = buffer.getvalue()
            buffer.close()
            plt.close(fig)
            
            return base64.b64encode(plot_data).decode()
            
        except Exception as e:
            print(f"Chart generation error: {e}")
            return ""
            
    def get_model_insights(self) -> Dict[str, Any]:
        """Generate detailed insights about model performance."""
        insights = {}
        
        for model_name, data in self.model_data.items():
            model_insights = {
                'total_predictions': len(data['predictions']),
                'avg_confidence': np.mean(data['confidences']) if data['confidences'] else 0,
                'accuracy_rate': (sum(data['accuracies']) / len(data['accuracies']) * 100) if data['accuracies'] else 0,
                'confidence_std': np.std(data['confidences']) if data['confidences'] else 0,
                'last_updated': data['last_updated']
            }
            
            # Prediction distribution analysis
            if data['predictions']:
                if isinstance(data['predictions'][0], dict):
                    # Handle probability distributions
                    all_moves = []
                    for pred in data['predictions']:
                        if isinstance(pred, dict):
                            best_move = max(pred.items(), key=lambda x: x[1])[0]
                            all_moves.append(best_move)
                    prediction_dist = Counter(all_moves)
                else:
                    prediction_dist = Counter(data['predictions'])
                    
                model_insights['prediction_distribution'] = dict(prediction_dist)
                
            # Recent performance trend
            if len(data['accuracies']) >= 10:
                recent_10 = data['accuracies'][-10:]
                older_10 = data['accuracies'][-20:-10] if len(data['accuracies']) >= 20 else []
                
                recent_acc = sum(recent_10) / len(recent_10) * 100
                older_acc = sum(older_10) / len(older_10) * 100 if older_10 else recent_acc
                
                model_insights['recent_trend'] = recent_acc - older_acc
                model_insights['recent_accuracy'] = recent_acc
                
            insights[model_name] = model_insights
            
        return insights

class DeveloperConsole:
    """Main developer console interface combining all monitoring tools."""
    
    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        self.model_comparator = ModelComparator()
        self.session_start = datetime.now()
        self.debug_log = []
        
    def log_debug(self, message: str, level: str = "INFO"):
        """Add debug message to log."""
        self.debug_log.append({
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message
        })
        
        # Keep only last 1000 entries
        if len(self.debug_log) > 1000:
            self.debug_log = self.debug_log[-1000:]
            
    def track_game_move(self, human_move: str, robot_move: str, result: str, 
                       model_predictions: Dict[str, Any], model_confidences: Dict[str, float]):
        """Track a complete game move with all model data."""
        
        # Track prediction accuracy
        for model_name, prediction in model_predictions.items():
            if isinstance(prediction, dict):
                # Handle probability distributions
                predicted_move = max(prediction.items(), key=lambda x: x[1])[0]
            else:
                predicted_move = prediction
                
            was_correct = predicted_move == human_move
            self.performance_monitor.track_prediction_accuracy(model_name, was_correct)
            
            confidence = model_confidences.get(model_name, 0.0)
            self.model_comparator.add_model_prediction(
                model_name, prediction, confidence, human_move
            )
            
        # Log the move
        self.log_debug(f"Game move: H:{human_move} R:{robot_move} Result:{result}")
        
        # Track memory
        self.performance_monitor.track_memory_usage()
        
    def track_model_inference(self, model_name: str, duration: float):
        """Track model inference timing."""
        self.performance_monitor.track_inference_time(model_name, duration)
        
        if duration > 0.1:  # Log slow inferences
            self.log_debug(f"Slow inference: {model_name} took {duration:.3f}s", "WARNING")
            
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate complete developer report."""
        
        report = {
            'session_info': {
                'start_time': self.session_start.isoformat(),
                'duration': str(datetime.now() - self.session_start),
                'total_debug_entries': len(self.debug_log)
            },
            'performance_summary': self.performance_monitor.get_summary(),
            'model_insights': self.model_comparator.get_model_insights(),
            'recent_debug_log': self.debug_log[-20:],  # Last 20 entries
            'system_info': {
                'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
                'cpu_percent': psutil.cpu_percent(),
                'available_memory_mb': psutil.virtual_memory().available / 1024 / 1024
            }
        }
        
        # Add performance warnings
        warnings = []
        
        # Check for memory leaks
        memory_history = self.performance_monitor.metrics.get('memory_usage', [])
        if len(memory_history) >= 10:
            recent_memory = [m['memory_mb'] for m in memory_history[-10:]]
            if max(recent_memory) - min(recent_memory) > 50:  # 50MB increase
                warnings.append("Possible memory leak detected")
                
        # Check for slow models
        for key, values in self.performance_monitor.metrics.items():
            if '_inference_time' in key and values:
                avg_time = sum(v['duration'] for v in values) / len(values)
                if avg_time > 0.05:  # 50ms threshold
                    model_name = key.replace('_inference_time', '')
                    warnings.append(f"Model {model_name} has slow inference ({avg_time:.3f}s avg)")
                    
        report['warnings'] = warnings
        
        return report
        
    def get_comparison_chart(self) -> str:
        """Get base64 encoded model comparison chart."""
        return self.model_comparator.generate_comparison_chart()
        
    def export_session_data(self, filepath: str):
        """Export complete session data for analysis."""
        export_data = {
            'session_start': self.session_start.isoformat(),
            'export_time': datetime.now().isoformat(),
            'performance_metrics': dict(self.performance_monitor.metrics),
            'model_data': dict(self.model_comparator.model_data),
            'debug_log': self.debug_log,
            'comprehensive_report': self.generate_comprehensive_report()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
            
        self.log_debug(f"Session data exported to {filepath}")

# Global console instance
console = DeveloperConsole()

# Utility functions for easy integration
def track_move(human_move: str, robot_move: str, result: str, 
               model_predictions: Dict[str, Any], model_confidences: Dict[str, float]):
    """Quick function to track a game move."""
    console.track_game_move(human_move, robot_move, result, model_predictions, model_confidences)

def track_inference(model_name: str, duration: float):
    """Quick function to track model inference."""
    console.track_model_inference(model_name, duration)

def get_developer_report() -> Dict[str, Any]:
    """Quick function to get developer report."""
    return console.generate_comprehensive_report()

def get_chart() -> str:
    """Quick function to get comparison chart."""
    return console.get_comparison_chart()

if __name__ == "__main__":
    # Test the developer console
    print("🚀 Developer Console Test")
    
    # Simulate some data
    test_predictions = {
        'random': 'paper',
        'frequency': 'rock', 
        'markov': 'scissors'
    }
    
    test_confidences = {
        'random': 0.33,
        'frequency': 0.67,
        'markov': 0.54
    }
    
    # Track a few moves
    for i in range(5):
        track_move('paper', 'rock', 'human_win', test_predictions, test_confidences)
        track_inference('markov', 0.012)
        
    # Generate report
    report = get_developer_report()
    print(f"📊 Generated report with {len(report)} sections")
    print(f"🎯 Tracked {report['performance_summary']['total_metrics']} metrics")
    
    # Export test data
    console.export_session_data('test_session.json')
    print("✅ Developer Console Test Complete!")