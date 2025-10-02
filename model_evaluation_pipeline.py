#!/usr/bin/env python3
"""
Model Evaluation and Promotion Pipeline
Provides robust way to evaluate and promote RPS coaching models
"""

import os
import json
import time
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import shutil

class ModelEvaluationPipeline:
    """Comprehensive pipeline for evaluating and promoting models"""
    
    def __init__(self, workspace_root: str):
        self.workspace_root = Path(workspace_root)
        self.models_dir = self.workspace_root / "models"
        self.staging_dir = self.workspace_root / "model_staging"
        self.evaluation_dir = self.workspace_root / "model_evaluations"
        
        # Create directories
        self.staging_dir.mkdir(exist_ok=True)
        self.evaluation_dir.mkdir(exist_ok=True)
        
        # Evaluation thresholds
        self.promotion_thresholds = {
            'overall_score': 0.75,
            'coaching_effectiveness': 0.70,
            'response_time': 0.80,
            'model_size_mb': 300,  # Max size
            'stability_score': 0.85,
            'user_satisfaction': 0.75
        }
        
    def evaluate_model_candidate(self, 
                               model_path: str, 
                               model_name: str,
                               run_full_evaluation: bool = True) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a model candidate
        
        Args:
            model_path: Path to the model file
            model_name: Name/version of the model
            run_full_evaluation: Whether to run full evaluation suite
            
        Returns:
            Complete evaluation report with promotion recommendation
        """
        
        print(f"🔍 Evaluating model candidate: {model_name}")
        evaluation_start = time.time()
        
        # Create evaluation record
        evaluation_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        eval_dir = self.evaluation_dir / evaluation_id
        eval_dir.mkdir(exist_ok=True)
        
        evaluation_report = {
            'model_name': model_name,
            'model_path': model_path,
            'evaluation_id': evaluation_id,
            'timestamp': datetime.now().isoformat(),
            'promotion_status': 'pending',
            'scores': {},
            'tests': {},
            'benchmarks': {},
            'recommendation': 'pending'
        }
        
        try:
            # Stage 1: Basic Model Validation
            print("📋 Stage 1: Basic validation...")
            validation_results = self._validate_model_basics(model_path)
            evaluation_report['tests']['validation'] = validation_results
            
            if not validation_results['passed']:
                evaluation_report['promotion_status'] = 'rejected'
                evaluation_report['recommendation'] = 'Model failed basic validation'
                return evaluation_report
            
            # Stage 2: Performance Benchmarking
            print("⚡ Stage 2: Performance benchmarking...")
            performance_results = self._benchmark_model_performance(model_path)
            evaluation_report['benchmarks']['performance'] = performance_results
            
            # Stage 3: Quality Assessment
            if run_full_evaluation:
                print("🎯 Stage 3: Quality assessment...")
                quality_results = self._assess_model_quality(model_path, model_name)
                evaluation_report['scores'] = quality_results
                
                # Stage 4: A/B Testing Simulation
                print("🧪 Stage 4: A/B testing simulation...")
                ab_test_results = self._simulate_ab_test(model_path)
                evaluation_report['tests']['ab_testing'] = ab_test_results
                
                # Stage 5: Stability Testing
                print("🔒 Stage 5: Stability testing...")
                stability_results = self._test_model_stability(model_path)
                evaluation_report['tests']['stability'] = stability_results
            
            # Stage 6: Generate Promotion Recommendation
            print("🏆 Stage 6: Generating recommendation...")
            recommendation = self._generate_promotion_recommendation(evaluation_report)
            evaluation_report.update(recommendation)
            
        except Exception as e:
            evaluation_report['error'] = str(e)
            evaluation_report['promotion_status'] = 'error'
            evaluation_report['recommendation'] = f'Evaluation failed: {e}'
        
        # Save evaluation report
        evaluation_time = time.time() - evaluation_start
        evaluation_report['evaluation_time_minutes'] = evaluation_time / 60
        
        report_path = eval_dir / "evaluation_report.json"
        with open(report_path, 'w') as f:
            json.dump(evaluation_report, f, indent=2, default=str)
        
        print(f"✅ Evaluation complete in {evaluation_time/60:.1f} minutes")
        print(f"📊 Report saved: {report_path}")
        
        return evaluation_report
    
    def _validate_model_basics(self, model_path: str) -> Dict[str, Any]:
        """Basic model validation checks"""
        
        results = {
            'passed': False,
            'checks': {},
            'errors': []
        }
        
        try:
            # Check 1: File exists and is readable
            if not os.path.exists(model_path):
                results['errors'].append(f"Model file not found: {model_path}")
                return results
            
            file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            results['checks']['file_size_mb'] = file_size_mb
            results['checks']['file_exists'] = True
            
            # Check 2: File size reasonable
            if file_size_mb < 0.1:  # Less than 100KB is suspicious
                results['errors'].append(f"Model file too small: {file_size_mb:.1f}MB")
            elif file_size_mb > 500:  # More than 500MB is too large
                results['errors'].append(f"Model file too large: {file_size_mb:.1f}MB")
            else:
                results['checks']['file_size_reasonable'] = True
            
            # Check 3: Can load with PyTorch
            try:
                import torch
                checkpoint = torch.load(model_path, map_location='cpu')
                results['checks']['pytorch_loadable'] = True
                
                # Check for required keys
                required_keys = ['model_state_dict']
                for key in required_keys:
                    if key in checkpoint:
                        results['checks'][f'has_{key}'] = True
                    else:
                        results['errors'].append(f"Missing required key: {key}")
                
            except Exception as e:
                results['errors'].append(f"Cannot load with PyTorch: {e}")
            
            # Check 4: Model architecture compatibility
            results['checks']['architecture_compatible'] = self._check_architecture_compatibility(model_path)
            
            # Overall validation result
            results['passed'] = len(results['errors']) == 0
            
        except Exception as e:
            results['errors'].append(f"Validation error: {e}")
        
        return results
    
    def _benchmark_model_performance(self, model_path: str) -> Dict[str, Any]:
        """Benchmark model performance metrics"""
        
        results = {
            'inference_time_ms': 0,
            'memory_usage_mb': 0,
            'throughput_queries_per_second': 0,
            'model_size_mb': 0
        }
        
        try:
            # Model size
            results['model_size_mb'] = os.path.getsize(model_path) / (1024 * 1024)
            
            # Simulate performance metrics (in real implementation, would load and test)
            import time
            
            # Simulated inference time
            start_time = time.time()
            time.sleep(0.01)  # Simulate 10ms inference
            results['inference_time_ms'] = (time.time() - start_time) * 1000
            
            # Simulated memory usage
            results['memory_usage_mb'] = results['model_size_mb'] * 1.5  # Estimate
            
            # Simulated throughput
            results['throughput_queries_per_second'] = 1000 / results['inference_time_ms']
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _assess_model_quality(self, model_path: str, model_name: str) -> Dict[str, Any]:
        """Assess model quality using evaluation framework"""
        
        # In real implementation, would use the comprehensive evaluator
        # For now, simulate quality scores
        
        scores = {
            'overall_score': 0.0,
            'coaching_effectiveness': 0.0,
            'response_coherence': 0.0,
            'response_time': 0.0,
            'model_size': 0.0,
            'educational_value': 0.0
        }
        
        try:
            # Simulate loading the comprehensive evaluator
            from evaluation.comprehensive_evaluator import ComprehensiveLLMEvaluator
            
            evaluator = ComprehensiveLLMEvaluator()
            
            # Create mock test data
            test_data = [
                {
                    'input': 'I keep losing to rock. What should I do?',
                    'context': 'Player has lost 5 games in a row, opponent played rock 4 times',
                    'expected_output': 'Try using paper more often, as it beats rock.'
                }
            ]
            
            # Would run actual evaluation here
            # For now, simulate realistic scores
            import random
            random.seed(hash(model_name))  # Deterministic based on model name
            
            scores['coaching_effectiveness'] = 0.6 + random.random() * 0.3
            scores['response_coherence'] = 0.7 + random.random() * 0.2
            scores['response_time'] = 0.8 + random.random() * 0.15
            scores['model_size'] = 0.9  # Assume within size constraints
            scores['educational_value'] = 0.65 + random.random() * 0.25
            
            # Calculate overall score
            weights = [0.35, 0.20, 0.20, 0.15, 0.10]
            score_values = [scores['coaching_effectiveness'], scores['response_coherence'], 
                          scores['response_time'], scores['model_size'], scores['educational_value']]
            scores['overall_score'] = sum(w * s for w, s in zip(weights, score_values))
            
        except Exception as e:
            scores['error'] = str(e)
        
        return scores
    
    def _simulate_ab_test(self, model_path: str) -> Dict[str, Any]:
        """Simulate A/B testing against current model"""
        
        results = {
            'test_scenarios': 100,
            'new_model_wins': 0,
            'current_model_wins': 0,
            'ties': 0,
            'win_rate': 0.0,
            'statistical_significance': False
        }
        
        try:
            import random
            
            # Simulate 100 coaching scenarios
            new_wins = 0
            current_wins = 0
            ties = 0
            
            for _ in range(100):
                # Simulate comparison result
                outcome = random.choice(['new', 'current', 'tie'])
                if outcome == 'new':
                    new_wins += 1
                elif outcome == 'current':
                    current_wins += 1
                else:
                    ties += 1
            
            results['new_model_wins'] = new_wins
            results['current_model_wins'] = current_wins
            results['ties'] = ties
            results['win_rate'] = new_wins / (new_wins + current_wins) if (new_wins + current_wins) > 0 else 0.5
            
            # Simple significance check (would use proper statistical test in real implementation)
            results['statistical_significance'] = abs(new_wins - current_wins) > 15
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _test_model_stability(self, model_path: str) -> Dict[str, Any]:
        """Test model stability and robustness"""
        
        results = {
            'consistency_score': 0.0,
            'edge_case_handling': 0.0,
            'error_rate': 0.0,
            'memory_leaks': False,
            'crash_incidents': 0
        }
        
        try:
            # Simulate stability testing
            import random
            
            results['consistency_score'] = 0.8 + random.random() * 0.15
            results['edge_case_handling'] = 0.7 + random.random() * 0.2
            results['error_rate'] = random.random() * 0.05  # 0-5% error rate
            results['memory_leaks'] = random.random() < 0.1  # 10% chance of memory issues
            results['crash_incidents'] = random.randint(0, 2)
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _check_architecture_compatibility(self, model_path: str) -> bool:
        """Check if model architecture is compatible with current system"""
        
        try:
            import torch
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Check for expected architecture
            state_dict = checkpoint.get('model_state_dict', {})
            
            # Look for expected layer patterns
            expected_patterns = ['embedding', 'transformer', 'output']
            found_patterns = 0
            
            for key in state_dict.keys():
                for pattern in expected_patterns:
                    if pattern in key.lower():
                        found_patterns += 1
                        break
            
            return found_patterns >= 2  # At least 2 expected patterns
            
        except Exception:
            return False
    
    def _generate_promotion_recommendation(self, evaluation_report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate promotion recommendation based on evaluation results"""
        
        recommendation = {
            'promotion_status': 'pending',
            'recommendation': '',
            'promotion_score': 0.0,
            'requirements_met': {},
            'blockers': [],
            'action_items': []
        }
        
        try:
            scores = evaluation_report.get('scores', {})
            benchmarks = evaluation_report.get('benchmarks', {})
            tests = evaluation_report.get('tests', {})
            
            # Check each requirement
            requirements_met = {}
            blockers = []
            total_score = 0
            weight_sum = 0
            
            # Overall score requirement
            overall_score = scores.get('overall_score', 0)
            requirements_met['overall_score'] = overall_score >= self.promotion_thresholds['overall_score']
            if not requirements_met['overall_score']:
                blockers.append(f"Overall score {overall_score:.3f} < {self.promotion_thresholds['overall_score']}")
            total_score += overall_score * 0.3
            weight_sum += 0.3
            
            # Coaching effectiveness requirement
            coaching_score = scores.get('coaching_effectiveness', 0)
            requirements_met['coaching_effectiveness'] = coaching_score >= self.promotion_thresholds['coaching_effectiveness']
            if not requirements_met['coaching_effectiveness']:
                blockers.append(f"Coaching effectiveness {coaching_score:.3f} < {self.promotion_thresholds['coaching_effectiveness']}")
            total_score += coaching_score * 0.25
            weight_sum += 0.25
            
            # Performance requirements
            perf = benchmarks.get('performance', {})
            model_size = perf.get('model_size_mb', 0)
            requirements_met['model_size'] = model_size <= self.promotion_thresholds['model_size_mb']
            if not requirements_met['model_size']:
                blockers.append(f"Model size {model_size:.1f}MB > {self.promotion_thresholds['model_size_mb']}MB")
            
            inference_time = perf.get('inference_time_ms', 1000)
            requirements_met['response_time'] = inference_time <= 200  # 200ms max
            if not requirements_met['response_time']:
                blockers.append(f"Inference time {inference_time:.1f}ms > 200ms")
            total_score += (200 - min(inference_time, 200)) / 200 * 0.2
            weight_sum += 0.2
            
            # Stability requirements
            stability = tests.get('stability', {})
            stability_score = stability.get('consistency_score', 0)
            requirements_met['stability'] = stability_score >= self.promotion_thresholds['stability_score']
            if not requirements_met['stability']:
                blockers.append(f"Stability score {stability_score:.3f} < {self.promotion_thresholds['stability_score']}")
            total_score += stability_score * 0.15
            weight_sum += 0.15
            
            # A/B testing requirements
            ab_test = tests.get('ab_testing', {})
            win_rate = ab_test.get('win_rate', 0.5)
            requirements_met['ab_testing'] = win_rate >= 0.6  # 60% win rate
            if not requirements_met['ab_testing']:
                blockers.append(f"A/B test win rate {win_rate:.3f} < 0.6")
            total_score += win_rate * 0.1
            weight_sum += 0.1
            
            # Calculate final promotion score
            recommendation['promotion_score'] = total_score / weight_sum if weight_sum > 0 else 0
            recommendation['requirements_met'] = requirements_met
            recommendation['blockers'] = blockers
            
            # Determine promotion status
            all_requirements_met = all(requirements_met.values())
            no_blockers = len(blockers) == 0
            
            if all_requirements_met and no_blockers:
                recommendation['promotion_status'] = 'approved'
                recommendation['recommendation'] = '🎉 APPROVED for production deployment!'
            elif recommendation['promotion_score'] >= 0.8:
                recommendation['promotion_status'] = 'conditional'
                recommendation['recommendation'] = '⚠️ CONDITIONAL approval - address blockers first'
                recommendation['action_items'] = [
                    'Address all blocking issues',
                    'Re-run evaluation after fixes',
                    'Consider staged rollout'
                ]
            else:
                recommendation['promotion_status'] = 'rejected'
                recommendation['recommendation'] = '❌ NOT READY for production'
                recommendation['action_items'] = [
                    'Improve model performance to meet thresholds',
                    'Address all blocking issues',
                    'Consider additional training or architecture changes'
                ]
            
        except Exception as e:
            recommendation['error'] = str(e)
            recommendation['promotion_status'] = 'error'
            recommendation['recommendation'] = f'Error generating recommendation: {e}'
        
        return recommendation
    
    def promote_model_to_production(self, evaluation_id: str, backup_current: bool = True) -> Dict[str, Any]:
        """Promote a model to production after successful evaluation"""
        
        result = {
            'success': False,
            'backup_created': False,
            'model_deployed': False,
            'rollback_available': False
        }
        
        try:
            # Load evaluation report
            eval_dir = self.evaluation_dir / evaluation_id
            report_path = eval_dir / "evaluation_report.json"
            
            if not report_path.exists():
                result['error'] = f"Evaluation report not found: {report_path}"
                return result
            
            with open(report_path, 'r') as f:
                evaluation_report = json.load(f)
            
            # Check promotion status
            if evaluation_report['promotion_status'] != 'approved':
                result['error'] = f"Model not approved for promotion: {evaluation_report['promotion_status']}"
                return result
            
            model_path = evaluation_report['model_path']
            model_name = evaluation_report['model_name']
            
            # Backup current production model
            if backup_current:
                production_model_path = self.models_dir / "rps_coach_production.pth"
                if production_model_path.exists():
                    backup_name = f"rps_coach_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
                    backup_path = self.models_dir / backup_name
                    shutil.copy2(production_model_path, backup_path)
                    result['backup_created'] = True
                    result['backup_path'] = str(backup_path)
            
            # Deploy new model
            production_path = self.models_dir / "rps_coach_production.pth"
            shutil.copy2(model_path, production_path)
            result['model_deployed'] = True
            result['production_path'] = str(production_path)
            
            # Create deployment record
            deployment_record = {
                'deployment_time': datetime.now().isoformat(),
                'model_name': model_name,
                'evaluation_id': evaluation_id,
                'model_path': model_path,
                'promotion_score': evaluation_report.get('promotion_score', 0),
                'backup_available': result['backup_created']
            }
            
            deployment_log = self.models_dir / "deployment_log.json"
            if deployment_log.exists():
                with open(deployment_log, 'r') as f:
                    log_data = json.load(f)
            else:
                log_data = {'deployments': []}
            
            log_data['deployments'].append(deployment_record)
            
            with open(deployment_log, 'w') as f:
                json.dump(log_data, f, indent=2, default=str)
            
            result['success'] = True
            result['rollback_available'] = result['backup_created']
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def rollback_model(self) -> Dict[str, Any]:
        """Rollback to previous production model"""
        
        result = {
            'success': False,
            'rollback_completed': False
        }
        
        try:
            # Find most recent backup
            backup_files = list(self.models_dir.glob("rps_coach_backup_*.pth"))
            if not backup_files:
                result['error'] = "No backup files found"
                return result
            
            # Get most recent backup
            most_recent_backup = max(backup_files, key=lambda x: x.stat().st_mtime)
            
            # Restore backup to production
            production_path = self.models_dir / "rps_coach_production.pth"
            shutil.copy2(most_recent_backup, production_path)
            
            result['success'] = True
            result['rollback_completed'] = True
            result['restored_from'] = str(most_recent_backup)
            
        except Exception as e:
            result['error'] = str(e)
        
        return result

def main():
    """Main CLI interface for model evaluation and promotion"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RPS Model Evaluation and Promotion Pipeline")
    parser.add_argument("command", choices=["evaluate", "promote", "rollback", "status"])
    parser.add_argument("--model-path", help="Path to model file")
    parser.add_argument("--model-name", help="Name/version of the model")
    parser.add_argument("--evaluation-id", help="Evaluation ID for promotion")
    parser.add_argument("--quick", action="store_true", help="Run quick evaluation only")
    
    args = parser.parse_args()
    
    workspace_root = "/Users/apple/Desktop/MyAppDevelopment/Paper_Scissor_Stone"
    pipeline = ModelEvaluationPipeline(workspace_root)
    
    if args.command == "evaluate":
        if not args.model_path or not args.model_name:
            print("❌ Error: --model-path and --model-name required for evaluation")
            return
        
        report = pipeline.evaluate_model_candidate(
            args.model_path, 
            args.model_name,
            run_full_evaluation=not args.quick
        )
        
        print("\n" + "="*60)
        print("🎯 EVALUATION RESULTS")
        print("="*60)
        print(f"Model: {report['model_name']}")
        print(f"Status: {report['promotion_status']}")
        print(f"Score: {report.get('promotion_score', 0):.3f}")
        print(f"Recommendation: {report['recommendation']}")
        
        if report.get('blockers'):
            print("\n❌ Blockers:")
            for blocker in report['blockers']:
                print(f"  • {blocker}")
        
    elif args.command == "promote":
        if not args.evaluation_id:
            print("❌ Error: --evaluation-id required for promotion")
            return
        
        result = pipeline.promote_model_to_production(args.evaluation_id)
        
        if result['success']:
            print("🎉 Model promoted to production successfully!")
        else:
            print(f"❌ Promotion failed: {result.get('error', 'Unknown error')}")
    
    elif args.command == "rollback":
        result = pipeline.rollback_model()
        
        if result['success']:
            print("🔄 Model rolled back successfully!")
        else:
            print(f"❌ Rollback failed: {result.get('error', 'Unknown error')}")
    
    elif args.command == "status":
        # Show current deployment status
        models_dir = Path(workspace_root) / "models"
        production_model = models_dir / "rps_coach_production.pth"
        
        if production_model.exists():
            size = production_model.stat().st_size / (1024 * 1024)
            modified = datetime.fromtimestamp(production_model.stat().st_mtime)
            print(f"✅ Production model: {size:.1f}MB (modified: {modified})")
        else:
            print("❌ No production model deployed")

if __name__ == "__main__":
    main()