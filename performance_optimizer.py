"""
Performance Optimization Suite for Rock Paper Scissors Game
Comprehensive bundle size optimization, timing validation, and resource monitoring.
"""

import os
import time
import json
import gzip
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import psutil
import threading
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np

class BundleSizeAnalyzer:
    """Analyze and optimize bundle sizes for web deployment."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.size_cache = {}
        
    def analyze_file_sizes(self) -> Dict[str, Any]:
        """Analyze file sizes throughout the project."""
        analysis = {
            'total_size': 0,
            'file_breakdown': {},
            'large_files': [],
            'optimization_suggestions': []
        }
        
        # Walk through all files
        for file_path in self.project_root.rglob('*'):
            if file_path.is_file() and not self._should_ignore(file_path):
                size = file_path.stat().st_size
                relative_path = str(file_path.relative_to(self.project_root))
                
                analysis['total_size'] += size
                analysis['file_breakdown'][relative_path] = {
                    'size_bytes': size,
                    'size_mb': size / 1024 / 1024,
                    'type': file_path.suffix
                }
                
                # Flag large files (>1MB)
                if size > 1024 * 1024:
                    analysis['large_files'].append({
                        'path': relative_path,
                        'size_mb': size / 1024 / 1024
                    })
                    
        # Generate optimization suggestions
        analysis['optimization_suggestions'] = self._generate_optimization_suggestions(analysis)
        
        return analysis
    
    def _should_ignore(self, file_path: Path) -> bool:
        """Check if file should be ignored in analysis."""
        ignore_patterns = [
            '__pycache__', '.git', '.DS_Store', 'node_modules',
            '.pyc', '.pyo', '.pyd', 'venv', '.env'
        ]
        
        path_str = str(file_path)
        return any(pattern in path_str for pattern in ignore_patterns)
    
    def _generate_optimization_suggestions(self, analysis: Dict) -> List[str]:
        """Generate suggestions for size optimization."""
        suggestions = []
        
        # Check for large Python files
        large_py_files = [f for f in analysis['large_files'] if f['path'].endswith('.py')]
        if large_py_files:
            suggestions.append(f"Consider splitting large Python files: {[f['path'] for f in large_py_files]}")
            
        # Check for large model files
        model_files = [f for f, data in analysis['file_breakdown'].items() 
                      if any(ext in f for ext in ['.pth', '.pkl', '.joblib', '.onnx'])]
        if model_files:
            total_model_size = sum(analysis['file_breakdown'][f]['size_mb'] for f in model_files)
            if total_model_size > 10:  # 10MB
                suggestions.append(f"Model files are large ({total_model_size:.1f}MB). Consider model compression or lazy loading.")
                
        # Check for static assets
        static_files = [f for f, data in analysis['file_breakdown'].items() 
                       if any(ext in data['type'] for ext in ['.png', '.jpg', '.jpeg', '.css', '.js'])]
        if static_files:
            static_size = sum(analysis['file_breakdown'][f]['size_mb'] for f in static_files)
            if static_size > 5:  # 5MB
                suggestions.append(f"Static assets are large ({static_size:.1f}MB). Consider compression or CDN.")
                
        return suggestions
    
    def compress_file(self, file_path: str) -> Dict[str, Any]:
        """Test compression ratio for a file."""
        path = Path(file_path)
        if not path.exists():
            return {'error': f'File not found: {file_path}'}
            
        original_size = path.stat().st_size
        
        with open(path, 'rb') as f:
            original_data = f.read()
            
        compressed_data = gzip.compress(original_data)
        compressed_size = len(compressed_data)
        
        return {
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compressed_size / original_size,
            'savings_percent': (1 - compressed_size / original_size) * 100,
            'savings_bytes': original_size - compressed_size
        }

class InferenceTimingValidator:
    """Validate and optimize model inference timing."""
    
    def __init__(self):
        self.timing_history = defaultdict(deque)
        self.max_history = 1000
        self.timing_thresholds = {
            'excellent': 0.01,    # 10ms
            'good': 0.05,         # 50ms
            'acceptable': 0.1,    # 100ms
            'slow': 0.5,          # 500ms
            'critical': 1.0       # 1000ms
        }
        
    def time_inference(self, model_name: str, inference_func, *args, **kwargs):
        """Time a model inference and store results."""
        start_time = time.perf_counter()
        try:
            result = inference_func(*args, **kwargs)
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            # Store timing
            self.timing_history[model_name].append({
                'timestamp': time.time(),
                'duration': duration,
                'success': True,
                'datetime': datetime.now().isoformat()
            })
            
            # Keep only recent history
            if len(self.timing_history[model_name]) > self.max_history:
                self.timing_history[model_name].popleft()
                
            return result, duration
            
        except Exception as e:
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            self.timing_history[model_name].append({
                'timestamp': time.time(),
                'duration': duration,
                'success': False,
                'error': str(e),
                'datetime': datetime.now().isoformat()
            })
            
            raise e
    
    def get_timing_analysis(self, model_name: str) -> Dict[str, Any]:
        """Get comprehensive timing analysis for a model."""
        if model_name not in self.timing_history:
            return {'error': f'No timing data for model: {model_name}'}
            
        timings = [t for t in self.timing_history[model_name] if t['success']]
        if not timings:
            return {'error': f'No successful timings for model: {model_name}'}
            
        durations = [t['duration'] for t in timings]
        
        analysis = {
            'model_name': model_name,
            'total_inferences': len(timings),
            'avg_duration': np.mean(durations),
            'median_duration': np.median(durations),
            'min_duration': np.min(durations),
            'max_duration': np.max(durations),
            'std_duration': np.std(durations),
            'p95_duration': np.percentile(durations, 95),
            'p99_duration': np.percentile(durations, 99)
        }
        
        # Categorize performance
        avg_time = analysis['avg_duration']
        if avg_time <= self.timing_thresholds['excellent']:
            analysis['performance_category'] = 'excellent'
        elif avg_time <= self.timing_thresholds['good']:
            analysis['performance_category'] = 'good'
        elif avg_time <= self.timing_thresholds['acceptable']:
            analysis['performance_category'] = 'acceptable'
        elif avg_time <= self.timing_thresholds['slow']:
            analysis['performance_category'] = 'slow'
        else:
            analysis['performance_category'] = 'critical'
            
        # Generate recommendations
        analysis['recommendations'] = self._generate_timing_recommendations(analysis)
        
        return analysis
    
    def _generate_timing_recommendations(self, analysis: Dict) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        category = analysis['performance_category']
        avg_time = analysis['avg_duration']
        
        if category == 'critical':
            recommendations.append("CRITICAL: Model inference is extremely slow. Consider model optimization, caching, or async processing.")
        elif category == 'slow':
            recommendations.append("Model inference is slow. Consider optimizing the model architecture or adding caching.")
        elif category == 'acceptable':
            recommendations.append("Model performance is acceptable but could be improved with optimization.")
            
        # Check for high variance
        if analysis['std_duration'] > analysis['avg_duration'] * 0.5:
            recommendations.append("High variance in inference times detected. Check for resource contention or GC issues.")
            
        # Check for outliers
        if analysis['p99_duration'] > analysis['avg_duration'] * 3:
            recommendations.append("Some inferences are much slower than average. Investigate edge cases or memory issues.")
            
        return recommendations
    
    def get_all_models_summary(self) -> Dict[str, Any]:
        """Get timing summary for all tracked models."""
        summary = {
            'total_models': len(self.timing_history),
            'models': {}
        }
        
        for model_name in self.timing_history.keys():
            analysis = self.get_timing_analysis(model_name)
            if 'error' not in analysis:
                summary['models'][model_name] = {
                    'avg_duration': analysis['avg_duration'],
                    'performance_category': analysis['performance_category'],
                    'total_inferences': analysis['total_inferences']
                }
                
        return summary

class ResourceUsageMonitor:
    """Monitor and optimize resource usage in real-time."""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.resource_history = deque(maxlen=1000)
        self.alerts = []
        self.thresholds = {
            'memory_mb': 500,      # 500MB
            'cpu_percent': 80,     # 80%
            'disk_io_mbps': 50     # 50MB/s
        }
        
    def start_monitoring(self, interval: float = 1.0):
        """Start continuous resource monitoring."""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            
    def _monitor_loop(self, interval: float):
        """Main monitoring loop."""
        last_disk_io = psutil.disk_io_counters()
        
        while self.monitoring:
            try:
                # Get current resource usage
                memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                cpu = psutil.cpu_percent(interval=0.1)
                
                # Calculate disk I/O rate
                current_disk_io = psutil.disk_io_counters()
                if last_disk_io:
                    read_mb = (current_disk_io.read_bytes - last_disk_io.read_bytes) / 1024 / 1024
                    write_mb = (current_disk_io.write_bytes - last_disk_io.write_bytes) / 1024 / 1024
                    total_io_mbps = (read_mb + write_mb) / interval
                else:
                    total_io_mbps = 0
                    
                last_disk_io = current_disk_io
                
                # Store measurement
                measurement = {
                    'timestamp': time.time(),
                    'memory_mb': memory,
                    'cpu_percent': cpu,
                    'disk_io_mbps': total_io_mbps,
                    'datetime': datetime.now().isoformat()
                }
                
                self.resource_history.append(measurement)
                
                # Check thresholds
                self._check_thresholds(measurement)
                
                time.sleep(interval)
                
            except Exception as e:
                print(f"Resource monitoring error: {e}")
                time.sleep(interval)
                
    def _check_thresholds(self, measurement: Dict):
        """Check if resource usage exceeds thresholds."""
        for metric, threshold in self.thresholds.items():
            if measurement[metric] > threshold:
                alert = {
                    'timestamp': measurement['timestamp'],
                    'metric': metric,
                    'value': measurement[metric],
                    'threshold': threshold,
                    'severity': 'high' if measurement[metric] > threshold * 1.5 else 'medium',
                    'datetime': measurement['datetime']
                }
                self.alerts.append(alert)
                
                # Keep only recent alerts
                if len(self.alerts) > 100:
                    self.alerts = self.alerts[-100:]
                    
    def get_current_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
        if not self.resource_history:
            return {'error': 'No monitoring data available'}
            
        latest = self.resource_history[-1]
        
        # Calculate recent averages (last 60 seconds)
        recent_cutoff = time.time() - 60
        recent_data = [r for r in self.resource_history if r['timestamp'] > recent_cutoff]
        
        if recent_data:
            avg_memory = np.mean([r['memory_mb'] for r in recent_data])
            avg_cpu = np.mean([r['cpu_percent'] for r in recent_data])
            avg_io = np.mean([r['disk_io_mbps'] for r in recent_data])
        else:
            avg_memory = latest['memory_mb']
            avg_cpu = latest['cpu_percent']
            avg_io = latest['disk_io_mbps']
            
        return {
            'current': latest,
            'recent_averages': {
                'memory_mb': avg_memory,
                'cpu_percent': avg_cpu,
                'disk_io_mbps': avg_io
            },
            'active_alerts': [a for a in self.alerts if time.time() - a['timestamp'] < 300],  # Last 5 minutes
            'monitoring_duration': time.time() - self.resource_history[0]['timestamp'] if self.resource_history else 0
        }
    
    def get_resource_trends(self) -> Dict[str, Any]:
        """Analyze resource usage trends."""
        if len(self.resource_history) < 10:
            return {'error': 'Insufficient data for trend analysis'}
            
        # Calculate trends over different time windows
        now = time.time()
        windows = {
            '5min': 300,
            '15min': 900,
            '1hour': 3600
        }
        
        trends = {}
        
        for window_name, window_seconds in windows.items():
            cutoff = now - window_seconds
            window_data = [r for r in self.resource_history if r['timestamp'] > cutoff]
            
            if len(window_data) >= 5:  # Need minimum data points
                memory_values = [r['memory_mb'] for r in window_data]
                cpu_values = [r['cpu_percent'] for r in window_data]
                
                # Calculate simple linear trends
                x = np.arange(len(memory_values))
                memory_trend = np.polyfit(x, memory_values, 1)[0]  # Slope
                cpu_trend = np.polyfit(x, cpu_values, 1)[0]
                
                trends[window_name] = {
                    'memory_trend_mb_per_minute': memory_trend * 60,
                    'cpu_trend_percent_per_minute': cpu_trend * 60,
                    'data_points': len(window_data)
                }
                
        return trends

class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self, project_root: str):
        self.bundle_analyzer = BundleSizeAnalyzer(project_root)
        self.timing_validator = InferenceTimingValidator()
        self.resource_monitor = ResourceUsageMonitor()
        self.optimization_history = []
        
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run complete performance analysis."""
        start_time = time.time()
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'analysis_duration': 0,
            'bundle_analysis': {},
            'timing_analysis': {},
            'resource_analysis': {},
            'overall_recommendations': [],
            'optimization_score': 0
        }
        
        try:
            # Bundle size analysis
            print("🔍 Analyzing bundle sizes...")
            analysis['bundle_analysis'] = self.bundle_analyzer.analyze_file_sizes()
            
            # Timing analysis
            print("⏱️ Analyzing inference timing...")
            analysis['timing_analysis'] = self.timing_validator.get_all_models_summary()
            
            # Resource analysis
            print("📊 Analyzing resource usage...")
            if self.resource_monitor.resource_history:
                analysis['resource_analysis'] = self.resource_monitor.get_current_usage()
                analysis['resource_trends'] = self.resource_monitor.get_resource_trends()
            else:
                analysis['resource_analysis'] = {'message': 'Resource monitoring not active'}
                
            # Generate overall recommendations
            analysis['overall_recommendations'] = self._generate_overall_recommendations(analysis)
            
            # Calculate optimization score
            analysis['optimization_score'] = self._calculate_optimization_score(analysis)
            
            analysis['analysis_duration'] = time.time() - start_time
            
            # Store in history
            self.optimization_history.append(analysis)
            
            return analysis
            
        except Exception as e:
            analysis['error'] = str(e)
            analysis['analysis_duration'] = time.time() - start_time
            return analysis
    
    def _generate_overall_recommendations(self, analysis: Dict) -> List[str]:
        """Generate comprehensive optimization recommendations."""
        recommendations = []
        
        # Bundle size recommendations
        if 'bundle_analysis' in analysis and analysis['bundle_analysis'].get('total_size', 0) > 0:
            total_mb = analysis['bundle_analysis']['total_size'] / 1024 / 1024
            if total_mb > 100:  # 100MB
                recommendations.append(f"Project size is large ({total_mb:.1f}MB). Consider removing unused dependencies.")
                
            if analysis['bundle_analysis'].get('optimization_suggestions'):
                recommendations.extend(analysis['bundle_analysis']['optimization_suggestions'])
                
        # Timing recommendations
        if 'timing_analysis' in analysis and 'models' in analysis['timing_analysis']:
            slow_models = [name for name, data in analysis['timing_analysis']['models'].items() 
                          if data.get('performance_category') in ['slow', 'critical']]
            if slow_models:
                recommendations.append(f"Slow models detected: {slow_models}. Consider optimization or caching.")
                
        # Resource recommendations
        if 'resource_analysis' in analysis and 'recent_averages' in analysis['resource_analysis']:
            avg_memory = analysis['resource_analysis']['recent_averages'].get('memory_mb', 0)
            avg_cpu = analysis['resource_analysis']['recent_averages'].get('cpu_percent', 0)
            
            if avg_memory > 200:  # 200MB
                recommendations.append(f"High memory usage ({avg_memory:.1f}MB). Check for memory leaks.")
            if avg_cpu > 70:  # 70%
                recommendations.append(f"High CPU usage ({avg_cpu:.1f}%). Consider optimizing algorithms.")
                
        return recommendations
    
    def _calculate_optimization_score(self, analysis: Dict) -> float:
        """Calculate overall optimization score (0-100)."""
        score = 100.0
        
        # Bundle size penalty
        if 'bundle_analysis' in analysis:
            total_mb = analysis['bundle_analysis'].get('total_size', 0) / 1024 / 1024
            if total_mb > 50:
                score -= min(20, (total_mb - 50) * 0.4)  # Penalty for large bundles
                
        # Timing penalty
        if 'timing_analysis' in analysis and 'models' in analysis['timing_analysis']:
            for model_data in analysis['timing_analysis']['models'].values():
                if model_data.get('performance_category') == 'critical':
                    score -= 15
                elif model_data.get('performance_category') == 'slow':
                    score -= 10
                elif model_data.get('performance_category') == 'acceptable':
                    score -= 5
                    
        # Resource penalty
        if 'resource_analysis' in analysis and 'recent_averages' in analysis['resource_analysis']:
            avg_memory = analysis['resource_analysis']['recent_averages'].get('memory_mb', 0)
            avg_cpu = analysis['resource_analysis']['recent_averages'].get('cpu_percent', 0)
            
            if avg_memory > 300:
                score -= 15
            elif avg_memory > 200:
                score -= 10
                
            if avg_cpu > 80:
                score -= 15
            elif avg_cpu > 60:
                score -= 10
                
        return max(0, min(100, score))
    
    def start_monitoring(self):
        """Start continuous monitoring."""
        self.resource_monitor.start_monitoring()
        
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.resource_monitor.stop_monitoring()
        
    def export_analysis(self, filepath: str):
        """Export performance analysis to file."""
        analysis = self.run_comprehensive_analysis()
        with open(filepath, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)

# Global optimizer instance
optimizer = PerformanceOptimizer(os.path.dirname(__file__))

# Utility functions for easy integration
def time_model_inference(model_name: str, inference_func, *args, **kwargs):
    """Time a model inference."""
    return optimizer.timing_validator.time_inference(model_name, inference_func, *args, **kwargs)

def get_performance_report() -> Dict[str, Any]:
    """Get comprehensive performance report."""
    return optimizer.run_comprehensive_analysis()

def start_performance_monitoring():
    """Start performance monitoring."""
    optimizer.start_monitoring()

def stop_performance_monitoring():
    """Stop performance monitoring."""
    optimizer.stop_monitoring()

if __name__ == "__main__":
    print("🚀 Performance Optimization Suite Test")
    
    # Start monitoring
    start_performance_monitoring()
    print("📊 Resource monitoring started")
    
    # Wait a bit to collect data
    time.sleep(2)
    
    # Run analysis
    report = get_performance_report()
    print(f"✅ Performance analysis complete")
    print(f"📊 Optimization score: {report.get('optimization_score', 0):.1f}/100")
    print(f"⏱️ Analysis took: {report.get('analysis_duration', 0):.2f}s")
    
    if report.get('overall_recommendations'):
        print("💡 Recommendations:")
        for rec in report['overall_recommendations']:
            print(f"  - {rec}")
    
    # Stop monitoring
    stop_performance_monitoring()
    print("✅ Performance Optimization Suite Test Complete!")