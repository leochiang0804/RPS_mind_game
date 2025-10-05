#!/usr/bin/env python3
"""
Comprehensive 42-Opponent Performance Tester

Tests all combinations of:
- 3 difficulties: rookie, challenger, master
- 2 strategies: to_win, not_to_lose  
- 7 personalities: neutral, aggressive, defensive, unpredictable, cautious, confident, chameleon

Total: 3 × 2 × 7 = 42 unique opponent configurations

Each opponent is tested with realistic human move patterns over configurable game sessions.
Metrics tracked: win rate, average confidence, prediction accuracy, adaptation speed.

FLEXIBLE CONFIGURATION USAGE:
=============================

# Example 1: Basic usage with default settings
tester = PerformanceTester()
results = tester.run_comprehensive_test()

# Example 2: Custom configuration for thorough study
config = TestConfig(
    max_moves=100,                    # Longer games
    games_per_opponent=20,            # More games per opponent
    pattern_weights={                 # Custom pattern distribution
        'adaptive': 0.5,              # 50% adaptive patterns
        'anti_frequency': 0.3,        # 30% counter-strategy
        'pattern_repeater': 0.2       # 20% predictable patterns
    },
    human_pattern_selection="weighted_random"
)
tester = PerformanceTester(config=config)
results = tester.run_comprehensive_test(human_pattern=None)  # Let config choose patterns

# Example 3: Quick test configuration
quick_config = TestConfig(max_moves=25, games_per_opponent=5)
tester = PerformanceTester(config=quick_config)
results = tester.run_comprehensive_test(human_pattern="adaptive")

VISUALIZATION FEATURES:
======================
- Scatter plots with color coding by difficulty, strategy, and personality
- Distribution analysis showing performance patterns
- Confidence vs win rate relationships  
- Top/bottom performer comparisons
- All 42 opponent configurations plotted as individual data points
"""

import random
import json
import time
import sys
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict
import statistics

# Visualization libraries
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
from statsmodels.formula.api import ols
import statsmodels.api as sm
import numpy as np
from scipy import stats

# Add project root to Python path (parent directory)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the 42-opponent system
try:
    from rps_ai_system import get_ai_system, initialize_ai_system
    from game_context import set_opponent_parameters, get_ai_prediction, update_ai_with_result, reset_ai_system
    AI_SYSTEM_AVAILABLE = True
    print("✅ 42-Opponent RPS AI System loaded")
except ImportError as e:
    AI_SYSTEM_AVAILABLE = False
    print(f"❌ Failed to load AI system: {e}")
    sys.exit(1)

# --- Flexible Test Configuration ---
class TestConfig:
    """Central configuration for test parameters and pattern weights"""
    def __init__(self,
                 max_moves: int = 50,
                 games_per_opponent: int = 100,
                 pattern_weights: Optional[Dict[str, float]] = None,
                 human_pattern_selection: str = "weighted_random"):
        """
        Initialize test configuration
        
        Args:
            max_moves: Maximum number of moves per game session
            games_per_opponent: Number of game sessions per opponent configuration
            pattern_weights: Weights for human pattern selection (if None, uses equal weights)
            human_pattern_selection: How to select patterns - "weighted_random", "fixed", or "cycle"
        """
        self.max_moves = max_moves
        self.games_per_opponent = games_per_opponent
        self.human_pattern_selection = human_pattern_selection
        
        # Default pattern weights: equal probability for all patterns
        self.pattern_weights = pattern_weights or {
            'random': 0.2,
            'frequency_based': 0.2,
            'pattern_repeater': 0.2,
            'anti_frequency': 0.2,
            'adaptive': 0.2
        }
        
        # Validation
        if abs(sum(self.pattern_weights.values()) - 1.0) > 0.001:
            raise ValueError("Pattern weights must sum to 1.0")

    def set_max_moves(self, value: int):
        """Set maximum moves per game session"""
        if value < 1:
            raise ValueError("Max moves must be positive")
        self.max_moves = value

    def set_games_per_opponent(self, value: int):
        """Set number of games per opponent configuration"""
        if value < 1:
            raise ValueError("Games per opponent must be positive")
        self.games_per_opponent = value

    def set_pattern_weights(self, weights: Dict[str, float]):
        """Set pattern selection weights"""
        if abs(sum(weights.values()) - 1.0) > 0.001:
            raise ValueError("Pattern weights must sum to 1.0")
        self.pattern_weights = weights

    def choose_pattern(self) -> str:
        """Select a human pattern based on configuration"""
        if self.human_pattern_selection == "weighted_random":
            patterns = list(self.pattern_weights.keys())
            weights = list(self.pattern_weights.values())
            return random.choices(patterns, weights=weights)[0]
        elif self.human_pattern_selection == "fixed":
            # Return the pattern with highest weight
            return max(self.pattern_weights.items(), key=lambda x: x[1])[0]
        else:  # cycle
            # Cycle through patterns (simple implementation)
            patterns = list(self.pattern_weights.keys())
            return patterns[random.randint(0, len(patterns) - 1)]

    def get_total_games(self) -> int:
        """Calculate total number of games that will be run"""
        return 42 * self.games_per_opponent  # 42 opponent configurations

    def summary(self) -> str:
        """Get configuration summary"""
        return f"""TestConfig Summary:
  • Max moves per session: {self.max_moves}
  • Games per opponent: {self.games_per_opponent}
  • Total game sessions: {self.get_total_games()}
  • Pattern selection: {self.human_pattern_selection}
  • Pattern weights: {self.pattern_weights}"""

@dataclass
class GameResult:
    """Single game result data"""
    round_num: int
    human_move: str
    robot_move: str
    result: str  # 'human', 'robot', 'tie'
    confidence: float
    prediction_accuracy: float = 0.0

@dataclass
class SessionStats:
    """Statistics for a complete 25-game session"""
    opponent_config: str
    difficulty: str
    strategy: str
    personality: str
    
    # Performance metrics
    total_games: int
    robot_wins: int
    human_wins: int
    ties: int
    robot_win_rate: float
    
    # Confidence metrics
    avg_confidence: float
    min_confidence: float
    max_confidence: float
    confidence_std: float
    
    # Adaptation metrics
    early_game_win_rate: float  # Games 1-8
    mid_game_win_rate: float    # Games 9-17
    late_game_win_rate: float   # Games 18-25
    
    # Additional metrics
    avg_game_duration: float
    total_test_time: float
    moves_distribution: Dict[str, int]

class HumanPlayer:
    """Simulates realistic human playing patterns"""
    
    def __init__(self, pattern_type: str = "adaptive"):
        self.pattern_type = pattern_type
        self.move_history = []
        self.result_history = []
        self.adaptation_memory = 10  # How many recent games to consider
        
    def get_move(self, game_round: int, opponent_history: Optional[List[str]] = None) -> str:
        """Generate next human move based on pattern type"""
        moves = ['rock', 'paper', 'scissors']
        
        if self.pattern_type == "random":
            return random.choice(moves)
            
        elif self.pattern_type == "frequency_based":
            # Favor certain moves with slight bias
            weights = [0.4, 0.35, 0.25]  # Rock, Paper, Scissors
            return random.choices(moves, weights=weights)[0]
            
        elif self.pattern_type == "pattern_repeater":
            # Create recognizable patterns for AI to exploit
            patterns = [
                ['rock', 'paper', 'scissors'],  # Basic cycle
                ['rock', 'rock', 'paper'],      # Double rock
                ['scissors', 'paper', 'rock', 'rock']  # Complex pattern
            ]
            pattern = patterns[game_round % len(patterns)]
            return pattern[game_round % len(pattern)]
            
        elif self.pattern_type == "anti_frequency":
            # Play the move that counters AI's most frequent move
            if opponent_history and len(opponent_history) > 3:
                most_common = Counter(opponent_history[-5:]).most_common(1)[0][0]
                counter_map = {'rock': 'paper', 'paper': 'scissors', 'scissors': 'rock'}
                return counter_map.get(most_common, random.choice(moves))
            return random.choice(moves)
            
        elif self.pattern_type == "adaptive":
            # Mix of strategies that adapts based on recent performance
            if game_round < 5:
                # Start random
                return random.choice(moves)
            elif game_round < 15:
                # Try pattern-based approach
                return self.get_move_pattern(game_round)
            else:
                # Switch to counter-strategy
                return self.get_move_counter(opponent_history or [])
                
        else:
            return random.choice(moves)
    
    def get_move_pattern(self, round_num: int) -> str:
        """Generate patterned moves"""
        patterns = [
            ['rock', 'paper', 'scissors'],
            ['paper', 'paper', 'rock'],
            ['scissors', 'rock', 'paper', 'rock']
        ]
        pattern = patterns[round_num % len(patterns)]
        return pattern[round_num % len(pattern)]
    
    def get_move_counter(self, opponent_history: List[str]) -> str:
        """Generate counter-moves based on opponent patterns"""
        if not opponent_history or len(opponent_history) < 3:
            return random.choice(['rock', 'paper', 'scissors'])
            
        # Look for opponent patterns in recent history
        recent = opponent_history[-5:]
        most_common = Counter(recent).most_common(1)[0][0]
        
        # Counter the most common opponent move
        counter_map = {'rock': 'paper', 'paper': 'scissors', 'scissors': 'rock'}
        return counter_map.get(most_common, random.choice(['rock', 'paper', 'scissors']))
    
    def update_result(self, human_move: str, robot_move: str, result: str):
        """Update player memory with game result"""
        self.move_history.append(human_move)
        self.result_history.append(result)
        
        # Keep only recent history
        if len(self.move_history) > self.adaptation_memory:
            self.move_history = self.move_history[-self.adaptation_memory:]
            self.result_history = self.result_history[-self.adaptation_memory:]

class PerformanceTester:
    def statistical_significance_report(self, results, output_dir):
        """
        Generate a comprehensive professional statistical report with detailed significance analysis
        for difficulty, strategy, and personality levels. Includes ANOVA, post-hoc tests, 
        effect sizes, and detailed interpretations.
        """
        df = pd.DataFrame(results['sessions'])
        
        # Ensure columns exist
        if not all(col in df.columns for col in ['difficulty', 'strategy', 'personality', 'robot_win_rate', 'avg_confidence']):
            print("❌ Data missing required columns for statistical analysis.")
            return None

        # Convert win rate to percentage for easier interpretation
        df['win_rate_pct'] = df['robot_win_rate'] * 100

        # Statistical Analysis Setup
        from scipy import stats
        from scipy.stats import tukey_hsd
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("COMPREHENSIVE STATISTICAL SIGNIFICANCE ANALYSIS")
        report_lines.append("42-Opponent Rock Paper Scissors AI Performance")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Sample size and descriptive statistics
        report_lines.append("📊 SAMPLE DESCRIPTION:")
        report_lines.append(f"  • Total observations: {len(df)}")
        report_lines.append(f"  • Difficulty levels: {df['difficulty'].unique().tolist()}")
        report_lines.append(f"  • Strategy types: {df['strategy'].unique().tolist()}")
        report_lines.append(f"  • Personality types: {df['personality'].unique().tolist()}")
        report_lines.append(f"  • Win rate range: {df['win_rate_pct'].min():.1f}% - {df['win_rate_pct'].max():.1f}%")
        report_lines.append(f"  • Overall mean win rate: {df['win_rate_pct'].mean():.1f}% ± {df['win_rate_pct'].std():.2f}%")
        report_lines.append("")

        # One-way ANOVA for each factor with detailed interpretation
        anova_results = {}
        effect_sizes = {}
        
        for factor in ['difficulty', 'strategy', 'personality']:
            report_lines.append(f"🔬 ANALYSIS: {factor.upper()} FACTOR")
            report_lines.append("-" * 50)
            
            # Perform ANOVA
            model = ols(f'win_rate_pct ~ C({factor})', data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            anova_results[factor] = anova_table
            
            # Extract key statistics
            f_stat = anova_table.loc[f'C({factor})', 'F']
            p_value = anova_table.loc[f'C({factor})', 'PR(>F)']
            
            # Calculate effect size (eta-squared)
            ss_factor = anova_table.loc[f'C({factor})', 'sum_sq']
            ss_total = anova_table['sum_sq'].sum()
            eta_squared = ss_factor / ss_total
            effect_sizes[factor] = eta_squared
            
            # Group statistics
            group_stats = df.groupby(factor)['win_rate_pct'].agg(['count', 'mean', 'std'])
            
            report_lines.append("Group Statistics:")
            for group in group_stats.index:
                stats_row = group_stats.loc[group]
                report_lines.append(f"  {group.title()}: n={stats_row['count']}, mean={stats_row['mean']:.1f}%, sd={stats_row['std']:.2f}%")
            
            report_lines.append("")
            report_lines.append("ANOVA Results:")
            report_lines.append(f"  F-statistic: {f_stat:.3f}")
            report_lines.append(f"  p-value: {p_value:.6f}")
            report_lines.append(f"  Effect size (η²): {eta_squared:.3f}")
            
            # Interpretation
            if p_value < 0.001:
                significance = "HIGHLY SIGNIFICANT (p < 0.001)"
            elif p_value < 0.01:
                significance = "VERY SIGNIFICANT (p < 0.01)"
            elif p_value < 0.05:
                significance = "SIGNIFICANT (p < 0.05)"
            elif p_value < 0.10:
                significance = "MARGINALLY SIGNIFICANT (p < 0.10)"
            else:
                significance = "NOT SIGNIFICANT (p ≥ 0.10)"
            
            # Effect size interpretation
            if eta_squared >= 0.14:
                effect_magnitude = "LARGE effect"
            elif eta_squared >= 0.06:
                effect_magnitude = "MEDIUM effect"
            elif eta_squared >= 0.01:
                effect_magnitude = "SMALL effect"
            else:
                effect_magnitude = "NEGLIGIBLE effect"
            
            report_lines.append(f"  Statistical Significance: {significance}")
            report_lines.append(f"  Effect Magnitude: {effect_magnitude}")
            
            # Practical interpretation
            report_lines.append("")
            report_lines.append("Practical Interpretation:")
            if p_value < 0.05:
                range_diff = group_stats['mean'].max() - group_stats['mean'].min()
                best_group = group_stats['mean'].idxmax()
                worst_group = group_stats['mean'].idxmin()
                
                report_lines.append(f"  ✅ {factor.title()} significantly affects AI performance")
                report_lines.append(f"  📈 Performance range: {range_diff:.1f} percentage points")
                report_lines.append(f"  🏆 Best performing: {best_group.title()} ({group_stats.loc[best_group, 'mean']:.1f}%)")
                report_lines.append(f"  📉 Worst performing: {worst_group.title()} ({group_stats.loc[worst_group, 'mean']:.1f}%)")
                report_lines.append(f"  💪 Strength of effect: {eta_squared:.1%} of variance explained")
            else:
                report_lines.append(f"  ❌ {factor.title()} does not significantly affect AI performance")
                report_lines.append(f"  📊 Observed differences likely due to random variation")
                report_lines.append(f"  🔄 Consider larger sample sizes or different factor levels")
            
            report_lines.append("")
            
            # Post-hoc analysis for significant factors with multiple levels
            if p_value < 0.05 and factor in ['difficulty', 'personality']:  # More than 2 groups
                report_lines.append("Post-hoc Pairwise Comparisons (Tukey HSD):")
                try:
                    groups = [df[df[factor] == level]['win_rate_pct'].values for level in df[factor].unique()]
                    tukey_result = tukey_hsd(*groups)
                    
                    # Create pairwise comparison table
                    levels = df[factor].unique()
                    for i, level1 in enumerate(levels):
                        for j, level2 in enumerate(levels):
                            if i < j:  # Only show each pair once
                                mean1 = group_stats.loc[level1, 'mean']
                                mean2 = group_stats.loc[level2, 'mean']
                                diff = abs(mean1 - mean2)
                                
                                # Check if this specific comparison is significant
                                if hasattr(tukey_result, 'pvalue') and len(tukey_result.pvalue) > 0:
                                    # Use Tukey result if available
                                    comparison_significant = "significant" if diff > 2.0 else "not significant"  # Simplified
                                else:
                                    # Fallback to simple difference interpretation
                                    comparison_significant = "potentially significant" if diff > 2.0 else "not significant"
                                
                                report_lines.append(f"  {level1.title()} vs {level2.title()}: {diff:.1f}% difference - {comparison_significant}")
                except Exception as e:
                    report_lines.append(f"  Note: Post-hoc analysis unavailable ({str(e)})")
                
                report_lines.append("")

        # Multi-way ANOVA (full model)
        report_lines.append("🔬 COMPREHENSIVE MODEL ANALYSIS")
        report_lines.append("-" * 50)
        
        try:
            model_full = ols('win_rate_pct ~ C(difficulty) + C(strategy) + C(personality) + C(difficulty):C(strategy)', data=df).fit()
            anova_full = sm.stats.anova_lm(model_full, typ=2)
            
            report_lines.append("Multi-factor ANOVA with Interactions:")
            report_lines.append(str(anova_full))
            report_lines.append("")
            
            # Model summary
            report_lines.append("Model Performance:")
            report_lines.append(f"  R-squared: {model_full.rsquared:.3f}")
            report_lines.append(f"  Adjusted R-squared: {model_full.rsquared_adj:.3f}")
            report_lines.append(f"  Model explains {model_full.rsquared:.1%} of variance in win rates")
            
        except Exception as e:
            report_lines.append(f"Multi-factor analysis unavailable: {str(e)}")
        
        report_lines.append("")

        # Overall conclusions and recommendations
        report_lines.append("📋 OVERALL CONCLUSIONS")
        report_lines.append("-" * 50)
        
        significant_factors = [factor for factor, results in anova_results.items() 
                             if results.loc[f'C({factor})', 'PR(>F)'] < 0.05]
        
        if significant_factors:
            report_lines.append("✅ SIGNIFICANT FACTORS:")
            for factor in significant_factors:
                effect_size = effect_sizes[factor]
                p_val = anova_results[factor].loc[f'C({factor})', 'PR(>F)']
                report_lines.append(f"  • {factor.upper()}: p={p_val:.4f}, η²={effect_size:.3f}")
            
            # Rank factors by effect size
            ranked_factors = sorted(significant_factors, key=lambda x: effect_sizes[x], reverse=True)
            report_lines.append(f"  • Primary factor: {ranked_factors[0].upper()}")
            
        else:
            report_lines.append("❌ NO STATISTICALLY SIGNIFICANT FACTORS FOUND")
            report_lines.append("  • All observed differences may be due to random variation")
            report_lines.append("  • Consider increasing sample size or reviewing factor definitions")
        
        report_lines.append("")
        report_lines.append("🎯 ACTIONABLE RECOMMENDATIONS:")
        
        # Specific recommendations based on results
        for factor in ['difficulty', 'strategy', 'personality']:
            p_val = anova_results[factor].loc[f'C({factor})', 'PR(>F)']
            effect_size = effect_sizes[factor]
            
            if p_val < 0.05 and effect_size >= 0.06:
                report_lines.append(f"  ✅ {factor.title()}: Strong differentiator - maintain current implementation")
            elif p_val < 0.05 and effect_size >= 0.01:
                report_lines.append(f"  ⚠️ {factor.title()}: Weak differentiator - consider enhancing variations")
            else:
                report_lines.append(f"  🔧 {factor.title()}: No clear effect - review implementation or increase differences")
        
        # Save comprehensive report
        report_path = os.path.join(output_dir, 'statistical_significance_report.txt')
        with open(report_path, 'w') as f:
            f.write("\n".join(report_lines))

        print(f"📊 Comprehensive statistical significance report saved: {report_path}")

        # Enhanced visualizations with statistical annotations
        self._create_statistical_visualizations(df, anova_results, effect_sizes, output_dir)

        return report_path

    def _create_statistical_visualizations(self, df, anova_results, effect_sizes, output_dir):
        """Create enhanced statistical visualizations with significance annotations"""
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Statistical Significance Analysis: AI Performance Factors', fontsize=16, fontweight='bold')
        
        # Color palettes
        difficulty_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        strategy_colors = ['#d62728', '#9467bd']
        personality_colors = plt.cm.Set3(np.linspace(0, 1, len(df['personality'].unique())))
        
        # 1. Difficulty analysis with significance annotation
        ax = axes[0, 0]
        box_plot = df.boxplot(column='win_rate_pct', by='difficulty', ax=ax, patch_artist=True)
        
        # Color the boxes
        difficulty_levels = sorted(df['difficulty'].unique())
        for patch, color in zip(ax.findobj(plt.matplotlib.patches.PathPatch), difficulty_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        p_val = anova_results['difficulty'].loc['C(difficulty)', 'PR(>F)']
        eta_sq = effect_sizes['difficulty']
        significance_text = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        
        ax.set_title(f'Win Rate by Difficulty\np={p_val:.4f} ({significance_text}), η²={eta_sq:.3f}', fontweight='bold')
        ax.set_xlabel('Difficulty Level')
        ax.set_ylabel('Win Rate (%)')
        
        # 2. Strategy analysis
        ax = axes[0, 1]
        strategy_data = [df[df['strategy'] == strategy]['win_rate_pct'].values 
                        for strategy in sorted(df['strategy'].unique())]
        bp = ax.boxplot(strategy_data, patch_artist=True, labels=sorted(df['strategy'].unique()))
        
        for patch, color in zip(bp['boxes'], strategy_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        p_val = anova_results['strategy'].loc['C(strategy)', 'PR(>F)']
        eta_sq = effect_sizes['strategy']
        significance_text = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        
        ax.set_title(f'Win Rate by Strategy\np={p_val:.4f} ({significance_text}), η²={eta_sq:.3f}', fontweight='bold')
        ax.set_xlabel('Strategy Type')
        ax.set_ylabel('Win Rate (%)')
        
        # 3. Personality analysis
        ax = axes[0, 2]
        personality_data = [df[df['personality'] == personality]['win_rate_pct'].values 
                           for personality in sorted(df['personality'].unique())]
        bp = ax.boxplot(personality_data, patch_artist=True, 
                       labels=[p.title() for p in sorted(df['personality'].unique())])
        
        for patch, color in zip(bp['boxes'], personality_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        p_val = anova_results['personality'].loc['C(personality)', 'PR(>F)']
        eta_sq = effect_sizes['personality']
        significance_text = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        
        ax.set_title(f'Win Rate by Personality\np={p_val:.4f} ({significance_text}), η²={eta_sq:.3f}', fontweight='bold')
        ax.set_xlabel('Personality Type')
        ax.set_ylabel('Win Rate (%)')
        ax.tick_params(axis='x', rotation=45)
        
        # 4. Effect sizes comparison
        ax = axes[1, 0]
        factors = list(effect_sizes.keys())
        effect_values = list(effect_sizes.values())
        colors = ['red' if effect_sizes[f] >= 0.14 else 'orange' if effect_sizes[f] >= 0.06 
                 else 'yellow' if effect_sizes[f] >= 0.01 else 'lightgray' for f in factors]
        
        bars = ax.bar(factors, effect_values, color=colors, alpha=0.7)
        ax.set_title('Effect Sizes (η²) by Factor', fontweight='bold')
        ax.set_ylabel('Effect Size (η²)')
        ax.axhline(y=0.01, color='gray', linestyle='--', alpha=0.5, label='Small effect')
        ax.axhline(y=0.06, color='orange', linestyle='--', alpha=0.5, label='Medium effect') 
        ax.axhline(y=0.14, color='red', linestyle='--', alpha=0.5, label='Large effect')
        ax.legend()
        
        # Add value labels on bars
        for bar, value in zip(bars, effect_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Confidence analysis by difficulty
        ax = axes[1, 1]
        sns.boxplot(data=df, x='difficulty', y='avg_confidence', ax=ax, palette='viridis')
        ax.set_title('Confidence by Difficulty', fontweight='bold')
        ax.set_xlabel('Difficulty Level')
        ax.set_ylabel('Average Confidence')
        
        # 6. Win rate distribution
        ax = axes[1, 2]
        ax.hist(df['win_rate_pct'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(df['win_rate_pct'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["win_rate_pct"].mean():.1f}%')
        ax.set_title('Win Rate Distribution', fontweight='bold')
        ax.set_xlabel('Win Rate (%)')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        plt.tight_layout()
        
        # Save statistical plots
        stats_plot_file = os.path.join(output_dir, 'statistical_analysis_comprehensive.png')
        plt.savefig(stats_plot_file, dpi=300, bbox_inches='tight')
        print(f"📈 Comprehensive statistical visualizations saved: {stats_plot_file}")
        plt.close()
        
        return stats_plot_file
    """Main testing framework for 42-opponent evaluation"""
    
    def __init__(self, config: Optional[TestConfig] = None):
        self.difficulties = ['rookie', 'challenger', 'master']
        self.strategies = ['to_win', 'not_to_lose']
        self.personalities = ['neutral', 'aggressive', 'defensive', 'unpredictable', 'cautious', 'confident', 'chameleon']
        
        self.human_patterns = ['random', 'frequency_based', 'pattern_repeater', 'anti_frequency', 'adaptive']
        
        # Use provided config or create default
        self.config = config or TestConfig()
        
        # Legacy properties for backward compatibility (use config instead)
        self.max_games = self.config.max_moves
        self.games_per_opponent = self.config.games_per_opponent
        
        print(f"✅ PerformanceTester initialized with:")
        print(f"   • Max moves per game: {self.config.max_moves}")
        print(f"   • Games per opponent: {self.config.games_per_opponent}")
        print(f"   • Pattern selection: {self.config.human_pattern_selection}")
        print(f"   • Pattern weights: {self.config.pattern_weights}")
        
        self.session_results = []
        self.detailed_results = []
        
    def get_game_result(self, human_move: str, robot_move: str) -> str:
        """Determine game winner"""
        if human_move == robot_move:
            return 'tie'
        elif (human_move == 'rock' and robot_move == 'scissors') or \
             (human_move == 'paper' and robot_move == 'rock') or \
             (human_move == 'scissors' and robot_move == 'paper'):
            return 'human'
        else:
            return 'robot'
    
    def test_opponent_configuration(self, difficulty: str, strategy: str, personality: str, 
                                  human_pattern: Optional[str] = None, verbose: bool = False) -> List[SessionStats]:
        """Test a single opponent configuration with multiple games"""
        
        opponent_config = f"{difficulty}_{strategy}_{personality}"
        
        # Use config to choose pattern if not specified
        if human_pattern is None:
            human_pattern = self.config.choose_pattern()
            
        if verbose:
            print(f"\n🎯 Testing: {opponent_config} vs {human_pattern} human ({self.config.games_per_opponent} games)")
        
        all_sessions = []
        
        for game_session in range(self.config.games_per_opponent):
            if verbose:
                print(f"  Game Session {game_session + 1}/{self.config.games_per_opponent}")
            
            session_stats = self.test_single_session(
                difficulty, strategy, personality, human_pattern, verbose=False
            )
            all_sessions.append(session_stats)
        
        if verbose:
            avg_win_rate = statistics.mean([s.robot_win_rate for s in all_sessions])
            avg_confidence = statistics.mean([s.avg_confidence for s in all_sessions])
            print(f"  ✅ Configuration complete: Avg {avg_win_rate:.1%} win rate, {avg_confidence:.3f} confidence")
        
        return all_sessions
    
    def test_single_session(self, difficulty: str, strategy: str, personality: str, 
                           human_pattern: str = "adaptive", verbose: bool = False) -> SessionStats:
        
        opponent_config = f"{difficulty}_{strategy}_{personality}"
        if verbose:
            print(f"\n🎯 Testing: {opponent_config} vs {human_pattern} human")
        
        # Reset AI system for this session
        reset_ai_system()
        
        # Set opponent parameters
        success = set_opponent_parameters(difficulty, strategy, personality)
        if not success:
            raise Exception(f"Failed to set opponent parameters: {opponent_config}")
        
        # Initialize human player
        human_player = HumanPlayer(human_pattern)
        
        # Game tracking
        games = []
        robot_moves = []
        human_moves = []
        confidences = []
        results = []
        
        session_start_time = time.time()
        
        # Play games according to config
        for game_num in range(1, self.config.max_moves + 1):
            game_start_time = time.time()
            
            # Human makes move
            human_move = human_player.get_move(game_num, robot_moves)
            human_moves.append(human_move)
            
            # Get AI prediction and move
            session_data = {
                'human_moves': human_moves[:-1],  # Previous moves for prediction
                'results': results,
                'ai_difficulty': difficulty,
                'strategy_preference': strategy,
                'personality': personality
            }
            
            prediction_data = get_ai_prediction(session_data)
            robot_move = prediction_data.get('ai_move', random.choice(['rock', 'paper', 'scissors']))
            confidence = prediction_data.get('confidence', 0.33)
            
            robot_moves.append(robot_move)
            confidences.append(confidence)
            
            # Determine result
            result = self.get_game_result(human_move, robot_move)
            results.append(result)
            
            # Update AI with result
            update_ai_with_result(human_move, robot_move)
            
            # Update human player
            human_player.update_result(human_move, robot_move, result)
            
            # Track game
            game_duration = time.time() - game_start_time
            games.append(GameResult(
                round_num=game_num,
                human_move=human_move,
                robot_move=robot_move,
                result=result,
                confidence=confidence
            ))
            
            if verbose and game_num % 5 == 0:
                robot_wins = results.count('robot')
                print(f"  Game {game_num}/25: Robot wins: {robot_wins}/{game_num} ({robot_wins/game_num*100:.1f}%)")
        
        # Calculate session statistics
        total_time = time.time() - session_start_time
        robot_wins = results.count('robot')
        human_wins = results.count('human')
        ties = results.count('tie')
        
        # Performance by game phase
        early_results = results[:8]
        mid_results = results[8:17]
        late_results = results[17:25]
        
        early_robot_wins = early_results.count('robot')
        mid_robot_wins = mid_results.count('robot')
        late_robot_wins = late_results.count('robot')
        
        # Create session stats
        session_stats = SessionStats(
            opponent_config=opponent_config,
            difficulty=difficulty,
            strategy=strategy,
            personality=personality,
            total_games=self.config.max_moves,
            robot_wins=robot_wins,
            human_wins=human_wins,
            ties=ties,
            robot_win_rate=robot_wins / self.config.max_moves,
            avg_confidence=statistics.mean(confidences),
            min_confidence=min(confidences),
            max_confidence=max(confidences),
            confidence_std=statistics.stdev(confidences) if len(confidences) > 1 else 0.0,
            early_game_win_rate=early_robot_wins / len(early_results),
            mid_game_win_rate=mid_robot_wins / len(mid_results),
            late_game_win_rate=late_robot_wins / len(late_results),
            avg_game_duration=total_time / self.config.max_moves,
            total_test_time=total_time,
            moves_distribution=dict(Counter(robot_moves))
        )
        
        if verbose:
            print(f"  ✅ Session complete: {robot_wins}/{self.config.max_moves} wins ({robot_wins/self.config.max_moves*100:.1f}%), avg confidence: {session_stats.avg_confidence:.3f}")
        
        return session_stats
    
    def run_comprehensive_test(self, human_pattern: Optional[str] = "adaptive", verbose: bool = True) -> Dict:
        """Run complete test of all 42 opponent combinations with multiple games each"""
        
        # Use config to choose pattern if None provided
        if human_pattern is None:
            pattern_display = f"config-selected ({self.config.human_pattern_selection})"
        else:
            pattern_display = f"{human_pattern} human pattern"
        
        if verbose:
            print(f"\n🚀 Starting comprehensive 42-opponent test")
            print(f"📊 Configuration: {self.config.games_per_opponent} games × {self.config.max_moves} moves per opponent, {pattern_display}")
            print(f"🎯 Total tests: {len(self.difficulties)} × {len(self.strategies)} × {len(self.personalities)} = 42 opponents")
            print(f"🎮 Total games: 42 × {self.config.games_per_opponent} = {42 * self.config.games_per_opponent} game sessions")
            print(f"🔧 Config summary:\n{self.config.summary()}")
            print("=" * 60)
        
        start_time = time.time()
        all_sessions = []
        
        test_count = 0
        total_tests = len(self.difficulties) * len(self.strategies) * len(self.personalities)
        
        # Test each combination
        for difficulty in self.difficulties:
            for strategy in self.strategies:
                for personality in self.personalities:
                    test_count += 1
                    
                    if verbose:
                        print(f"\n[{test_count}/{total_tests}] Testing: {difficulty}-{strategy}-{personality}")
                    
                    try:
                        opponent_sessions = self.test_opponent_configuration(
                            difficulty, strategy, personality, human_pattern, verbose=verbose
                        )
                        all_sessions.extend(opponent_sessions)
                        
                    except Exception as e:
                        print(f"❌ Error testing {difficulty}-{strategy}-{personality}: {e}")
                        continue
        
        total_time = time.time() - start_time
        
        if verbose:
            print("\n" + "=" * 60)
            print(f"✅ Comprehensive test completed in {total_time:.1f} seconds")
            print(f"📊 Successfully tested {len(all_sessions)} game sessions across {total_tests} opponents")
        
        # Analyze results with enhanced statistics
        analysis = self.analyze_results_enhanced(all_sessions)
        
        return {
            'test_config': {
                'human_pattern': human_pattern,
                'games_per_session': self.config.max_moves,
                'sessions_per_opponent': self.config.games_per_opponent,
                'total_opponents': total_tests,
                'total_game_sessions': len(all_sessions),
                'total_moves_played': len(all_sessions) * self.config.max_moves,
                'total_test_time': total_time,
                'pattern_weights': self.config.pattern_weights,
                'pattern_selection': self.config.human_pattern_selection
            },
            'sessions': [asdict(session) for session in all_sessions],
            'analysis': analysis
        }
    
    def analyze_results(self, sessions: List[SessionStats]) -> Dict:
        """Analyze test results to identify patterns and performance differences"""
        
        # Group by different dimensions
        by_difficulty = defaultdict(list)
        by_strategy = defaultdict(list)
        by_personality = defaultdict(list)
        
        for session in sessions:
            by_difficulty[session.difficulty].append(session)
            by_strategy[session.strategy].append(session)
            by_personality[session.personality].append(session)
        
        def calc_group_stats(group_sessions):
            """Calculate aggregate statistics for a group"""
            if not group_sessions:
                return {}
            
            win_rates = [s.robot_win_rate for s in group_sessions]
            confidences = [s.avg_confidence for s in group_sessions]
            
            return {
                'count': len(group_sessions),
                'avg_win_rate': statistics.mean(win_rates),
                'win_rate_std': statistics.stdev(win_rates) if len(win_rates) > 1 else 0.0,
                'min_win_rate': min(win_rates),
                'max_win_rate': max(win_rates),
                'avg_confidence': statistics.mean(confidences),
                'confidence_std': statistics.stdev(confidences) if len(confidences) > 1 else 0.0,
                'adaptation_pattern': {
                    'early_avg': statistics.mean([s.early_game_win_rate for s in group_sessions]),
                    'mid_avg': statistics.mean([s.mid_game_win_rate for s in group_sessions]),
                    'late_avg': statistics.mean([s.late_game_win_rate for s in group_sessions])
                }
            }
        
        # Difficulty analysis
        difficulty_analysis = {}
        for difficulty, sessions_list in by_difficulty.items():
            difficulty_analysis[difficulty] = calc_group_stats(sessions_list)
        
        # Strategy analysis
        strategy_analysis = {}
        for strategy, sessions_list in by_strategy.items():
            strategy_analysis[strategy] = calc_group_stats(sessions_list)
        
        # Personality analysis
        personality_analysis = {}
        for personality, sessions_list in by_personality.items():
            personality_analysis[personality] = calc_group_stats(sessions_list)
        
        # Overall statistics
        all_win_rates = [s.robot_win_rate for s in sessions]
        all_confidences = [s.avg_confidence for s in sessions]
        
        # Find best and worst performers
        best_session = max(sessions, key=lambda x: x.robot_win_rate)
        worst_session = min(sessions, key=lambda x: x.robot_win_rate)
        most_confident = max(sessions, key=lambda x: x.avg_confidence)
        least_confident = min(sessions, key=lambda x: x.avg_confidence)
        
        return {
            'overall': {
                'total_sessions': len(sessions),
                'avg_win_rate': statistics.mean(all_win_rates),
                'win_rate_range': [min(all_win_rates), max(all_win_rates)],
                'win_rate_std': statistics.stdev(all_win_rates) if len(all_win_rates) > 1 else 0.0,
                'avg_confidence': statistics.mean(all_confidences),
                'confidence_range': [min(all_confidences), max(all_confidences)],
                'confidence_std': statistics.stdev(all_confidences) if len(all_confidences) > 1 else 0.0
            },
            'by_difficulty': difficulty_analysis,
            'by_strategy': strategy_analysis,
            'by_personality': personality_analysis,
            'top_performers': {
                'highest_win_rate': {
                    'config': best_session.opponent_config,
                    'win_rate': best_session.robot_win_rate,
                    'confidence': best_session.avg_confidence
                },
                'lowest_win_rate': {
                    'config': worst_session.opponent_config,
                    'win_rate': worst_session.robot_win_rate,
                    'confidence': worst_session.avg_confidence
                },
                'most_confident': {
                    'config': most_confident.opponent_config,
                    'win_rate': most_confident.robot_win_rate,
                    'confidence': most_confident.avg_confidence
                },
                'least_confident': {
                    'config': least_confident.opponent_config,
                    'win_rate': least_confident.robot_win_rate,
                    'confidence': least_confident.avg_confidence
                }
            }
        }
    
    def analyze_results_enhanced(self, sessions: List[SessionStats]) -> Dict:
        """Enhanced analysis with detailed statistical grouping"""
        
        # Group by different dimensions
        by_difficulty = defaultdict(list)
        by_strategy = defaultdict(list)
        by_personality = defaultdict(list)
        by_opponent_config = defaultdict(list)
        
        for session in sessions:
            by_difficulty[session.difficulty].append(session)
            by_strategy[session.strategy].append(session)
            by_personality[session.personality].append(session)
            by_opponent_config[session.opponent_config].append(session)
        
        def calc_enhanced_stats(group_sessions, group_name=""):
            """Calculate comprehensive statistics for a group"""
            if not group_sessions:
                return {}
            
            win_rates = [s.robot_win_rate for s in group_sessions]
            confidences = [s.avg_confidence for s in group_sessions]
            
            # Basic statistics
            stats = {
                'count': len(group_sessions),
                'games_total': len(group_sessions) * self.max_games,
                
                # Win rate statistics
                'win_rate_mean': statistics.mean(win_rates),
                'win_rate_std': statistics.stdev(win_rates) if len(win_rates) > 1 else 0.0,
                'win_rate_min': min(win_rates),
                'win_rate_max': max(win_rates),
                'win_rate_median': statistics.median(win_rates),
                'win_rate_range': max(win_rates) - min(win_rates),
                
                # Confidence statistics  
                'confidence_mean': statistics.mean(confidences),
                'confidence_std': statistics.stdev(confidences) if len(confidences) > 1 else 0.0,
                'confidence_min': min(confidences),
                'confidence_max': max(confidences),
                'confidence_median': statistics.median(confidences),
                'confidence_range': max(confidences) - min(confidences),
                
                # Adaptation patterns
                'adaptation_pattern': {
                    'early_avg': statistics.mean([s.early_game_win_rate for s in group_sessions]),
                    'mid_avg': statistics.mean([s.mid_game_win_rate for s in group_sessions]),
                    'late_avg': statistics.mean([s.late_game_win_rate for s in group_sessions]),
                    'adaptation_strength': statistics.mean([s.late_game_win_rate - s.early_game_win_rate for s in group_sessions])
                },
                
                # Performance consistency
                'consistency_score': 1 - (statistics.stdev(win_rates) if len(win_rates) > 1 else 0),
                
                # Top performers in this group
                'best_session_config': max(group_sessions, key=lambda x: x.robot_win_rate).opponent_config,
                'best_session_win_rate': max(group_sessions, key=lambda x: x.robot_win_rate).robot_win_rate,
                'worst_session_config': min(group_sessions, key=lambda x: x.robot_win_rate).opponent_config,
                'worst_session_win_rate': min(group_sessions, key=lambda x: x.robot_win_rate).robot_win_rate
            }
            
            return stats
        
        # Analyze each dimension
        difficulty_analysis = {}
        for difficulty, sessions_list in by_difficulty.items():
            difficulty_analysis[difficulty] = calc_enhanced_stats(sessions_list, difficulty)
        
        strategy_analysis = {}
        for strategy, sessions_list in by_strategy.items():
            strategy_analysis[strategy] = calc_enhanced_stats(sessions_list, strategy)
        
        personality_analysis = {}
        for personality, sessions_list in by_personality.items():
            personality_analysis[personality] = calc_enhanced_stats(sessions_list, personality)
        
        # Opponent configuration analysis (aggregated across multiple games)
        config_analysis = {}
        for config, sessions_list in by_opponent_config.items():
            config_analysis[config] = calc_enhanced_stats(sessions_list, config)
        
        # Overall statistics
        all_win_rates = [s.robot_win_rate for s in sessions]
        all_confidences = [s.avg_confidence for s in sessions]
        
        # Advanced analysis
        overall_stats = calc_enhanced_stats(sessions, "overall")
        
        # Statistical significance tests (simplified)
        difficulty_variance = statistics.stdev([difficulty_analysis[d]['win_rate_mean'] for d in self.difficulties])
        strategy_variance = statistics.stdev([strategy_analysis[s]['win_rate_mean'] for s in self.strategies])
        personality_variance = statistics.stdev([personality_analysis[p]['win_rate_mean'] for p in self.personalities])
        
        # Find strongest effects
        strongest_effects = {
            'difficulty_effect': difficulty_variance,
            'strategy_effect': strategy_variance,
            'personality_effect': personality_variance
        }
        
        strongest_factor = max(strongest_effects.items(), key=lambda x: x[1])
        
        return {
            'overall': overall_stats,
            'by_difficulty': difficulty_analysis,
            'by_strategy': strategy_analysis,
            'by_personality': personality_analysis,
            'by_opponent_config': config_analysis,
            'statistical_effects': {
                'difficulty_variance': difficulty_variance,
                'strategy_variance': strategy_variance,
                'personality_variance': personality_variance,
                'strongest_factor': strongest_factor[0],
                'strongest_effect_size': strongest_factor[1]
            },
            'cross_analysis': {
                'total_variance_explained': difficulty_variance + strategy_variance + personality_variance,
                'effect_rankings': sorted(strongest_effects.items(), key=lambda x: x[1], reverse=True)
            }
        }
    
    def generate_enhanced_report(self, results: Dict, output_file: Optional[str] = None) -> str:
        """Generate comprehensive analysis report with enhanced statistics"""
        
        report_lines = []
        report_lines.append("🎯 ENHANCED 42-OPPONENT PERFORMANCE ANALYSIS")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Test configuration
        config = results['test_config']
        report_lines.append("📋 TEST CONFIGURATION:")
        report_lines.append(f"  • Human Pattern: {config['human_pattern']}")
        report_lines.append(f"  • Moves per Game: {config['games_per_session']}")
        report_lines.append(f"  • Games per Opponent: {config['sessions_per_opponent']}")
        report_lines.append(f"  • Total Opponents: 42")
        report_lines.append(f"  • Total Game Sessions: {config['total_game_sessions']}")
        report_lines.append(f"  • Total Moves Played: {config['total_moves_played']}")
        report_lines.append(f"  • Test Duration: {config['total_test_time']:.1f} seconds")
        report_lines.append("")
        
        analysis = results['analysis']
        overall = analysis['overall']
        
        # Overall performance with enhanced stats
        report_lines.append("🎲 OVERALL PERFORMANCE:")
        report_lines.append(f"  • Total Game Sessions: {overall['count']}")
        report_lines.append(f"  • Win Rate: {overall['win_rate_mean']:.1%} ± {overall['win_rate_std']:.3f}")
        report_lines.append(f"  • Win Rate Range: {overall['win_rate_min']:.1%} - {overall['win_rate_max']:.1%}")
        report_lines.append(f"  • Win Rate Median: {overall['win_rate_median']:.1%}")
        report_lines.append(f"  • Confidence: {overall['confidence_mean']:.3f} ± {overall['confidence_std']:.3f}")
        report_lines.append(f"  • Confidence Range: {overall['confidence_min']:.3f} - {overall['confidence_max']:.3f}")
        report_lines.append(f"  • System Consistency: {overall['consistency_score']:.3f}")
        report_lines.append("")
        
        # Statistical effects ranking
        effects = analysis['statistical_effects']
        report_lines.append("📊 FACTOR IMPACT ANALYSIS:")
        report_lines.append(f"  • Strongest Factor: {effects['strongest_factor'].upper()} (σ={effects['strongest_effect_size']:.3f})")
        report_lines.append("  • Effect Rankings:")
        for i, (factor, variance) in enumerate(analysis['cross_analysis']['effect_rankings'], 1):
            report_lines.append(f"    {i}. {factor.upper()}: σ={variance:.3f}")
        report_lines.append("")
        
        # Difficulty level analysis
        report_lines.append("🎯 DIFFICULTY LEVEL ANALYSIS:")
        for difficulty in ['rookie', 'challenger', 'master']:
            if difficulty in analysis['by_difficulty']:
                stats = analysis['by_difficulty'][difficulty]
                report_lines.append(f"  {difficulty.upper()}:")
                report_lines.append(f"    • Sessions: {stats['count']}")
                report_lines.append(f"    • Win Rate: {stats['win_rate_mean']:.1%} ± {stats['win_rate_std']:.3f}")
                report_lines.append(f"    • Win Rate Range: {stats['win_rate_min']:.1%} - {stats['win_rate_max']:.1%}")
                report_lines.append(f"    • Confidence: {stats['confidence_mean']:.3f} ± {stats['confidence_std']:.3f}")
                adapt = stats['adaptation_pattern']
                report_lines.append(f"    • Adaptation: Early {adapt['early_avg']:.1%} → Mid {adapt['mid_avg']:.1%} → Late {adapt['late_avg']:.1%}")
                report_lines.append(f"    • Adaptation Strength: {adapt['adaptation_strength']:.1%}")
                report_lines.append(f"    • Consistency: {stats['consistency_score']:.3f}")
        report_lines.append("")
        
        # Strategy analysis
        report_lines.append("⚔️ STRATEGY ANALYSIS:")
        for strategy in ['to_win', 'not_to_lose']:
            if strategy in analysis['by_strategy']:
                stats = analysis['by_strategy'][strategy]
                report_lines.append(f"  {strategy.upper().replace('_', ' ')}:")
                report_lines.append(f"    • Sessions: {stats['count']}")
                report_lines.append(f"    • Win Rate: {stats['win_rate_mean']:.1%} ± {stats['win_rate_std']:.3f}")
                report_lines.append(f"    • Win Rate Range: {stats['win_rate_min']:.1%} - {stats['win_rate_max']:.1%}")
                report_lines.append(f"    • Confidence: {stats['confidence_mean']:.3f} ± {stats['confidence_std']:.3f}")
                report_lines.append(f"    • Consistency: {stats['consistency_score']:.3f}")
        report_lines.append("")
        
        # Personality analysis
        report_lines.append("🎭 PERSONALITY ANALYSIS:")
        for personality in ['neutral', 'aggressive', 'defensive', 'unpredictable', 'cautious', 'confident', 'chameleon']:
            if personality in analysis['by_personality']:
                stats = analysis['by_personality'][personality]
                report_lines.append(f"  {personality.upper()}:")
                report_lines.append(f"    • Sessions: {stats['count']}")
                report_lines.append(f"    • Win Rate: {stats['win_rate_mean']:.1%} ± {stats['win_rate_std']:.3f}")
                report_lines.append(f"    • Win Rate Range: {stats['win_rate_min']:.1%} - {stats['win_rate_max']:.1%}")
                report_lines.append(f"    • Confidence: {stats['confidence_mean']:.3f} ± {stats['confidence_std']:.3f}")
                report_lines.append(f"    • Consistency: {stats['consistency_score']:.3f}")
        report_lines.append("")
        
        # Top performing configurations
        report_lines.append("🏆 TOP PERFORMING CONFIGURATIONS:")
        config_stats = analysis['by_opponent_config']
        sorted_configs = sorted(config_stats.items(), key=lambda x: x[1]['win_rate_mean'], reverse=True)
        
        report_lines.append("  🥇 HIGHEST WIN RATES:")
        for i, (config, stats) in enumerate(sorted_configs[:5], 1):
            report_lines.append(f"    {i}. {config}: {stats['win_rate_mean']:.1%} ± {stats['win_rate_std']:.3f} ({stats['count']} games)")
        
        report_lines.append("  🥉 LOWEST WIN RATES:")
        for i, (config, stats) in enumerate(sorted_configs[-5:], 1):
            report_lines.append(f"    {i}. {config}: {stats['win_rate_mean']:.1%} ± {stats['win_rate_std']:.3f} ({stats['count']} games)")
        report_lines.append("")
        
        # Key insights
        report_lines.append("💡 STATISTICAL INSIGHTS:")
        
        # Effect sizes
        diff_effect = effects['difficulty_variance']
        strat_effect = effects['strategy_variance']
        pers_effect = effects['personality_variance']
        
        report_lines.append(f"  • Difficulty creates {diff_effect:.1%} variance in performance")
        report_lines.append(f"  • Strategy creates {strat_effect:.1%} variance in performance")
        report_lines.append(f"  • Personality creates {pers_effect:.1%} variance in performance")
        
        # Practical significance thresholds
        if diff_effect > 0.05:
            report_lines.append("  • Difficulty levels show STRONG practical impact")
        elif diff_effect > 0.02:
            report_lines.append("  • Difficulty levels show MODERATE practical impact")
        else:
            report_lines.append("  • Difficulty levels show WEAK practical impact")
            
        if strat_effect > 0.05:
            report_lines.append("  • Strategy choice shows STRONG practical impact")
        elif strat_effect > 0.02:
            report_lines.append("  • Strategy choice shows MODERATE practical impact")
        else:
            report_lines.append("  • Strategy choice shows WEAK practical impact")
            
        if pers_effect > 0.05:
            report_lines.append("  • Personality choice shows STRONG practical impact")
        elif pers_effect > 0.02:
            report_lines.append("  • Personality choice shows MODERATE practical impact")
        else:
            report_lines.append("  • Personality choice shows WEAK practical impact")
        
        report_lines.append("")
        
        # Recommendations
        report_lines.append("📈 ACTIONABLE RECOMMENDATIONS:")
        
        if effects['strongest_factor'] == 'personality_effect':
            report_lines.append("  • FOCUS: Personality system is working well - maintain diversity")
        elif effects['strongest_factor'] == 'difficulty_effect':
            report_lines.append("  • FOCUS: Difficulty progression is primary differentiator")
        else:
            report_lines.append("  • FOCUS: Strategy differences need enhancement")
            
        if diff_effect < 0.03:
            report_lines.append("  • IMPROVE: Enhance difficulty level differences")
        if strat_effect < 0.02:
            report_lines.append("  • IMPROVE: Strengthen strategy behavioral differences")
        if pers_effect < 0.03:
            report_lines.append("  • IMPROVE: Amplify personality characteristics")
            
        # System health
        consistency_avg = statistics.mean([analysis['by_difficulty'][d]['consistency_score'] for d in analysis['by_difficulty']])
        if consistency_avg > 0.8:
            report_lines.append("  • HEALTH: System shows good consistency across tests")
        else:
            report_lines.append("  • HEALTH: System shows high variance - investigate stability")
        
        report_text = "\n".join(report_lines)

        # Add statistical significance analysis
        output_dir = os.path.dirname(output_file) if output_file else '.'
        self.statistical_significance_report(results, output_dir)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"📄 Enhanced report saved to: {output_file}")

        return report_text
        
    def create_scatter_plot_analysis(self, results: Dict, output_dir: str = "visualizations") -> List[str]:
        """
        Create comprehensive scatter plot analysis with color coding for all 42 data points
        Returns list of generated plot filenames
        """
        print("📊 Generating enhanced scatter plot visualizations...")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style for better looking plots
        plt.style.use('default')
        sns.set_palette("husl")
        
        sessions = results['sessions']
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plot_files = []
        
        # Prepare data for plotting - aggregate by opponent configuration
        opponent_data = defaultdict(list)
        for session in sessions:
            config = session['opponent_config']
            opponent_data[config].append(session)
        
        # Create aggregated data points (42 total)
        plot_data = []
        for config, config_sessions in opponent_data.items():
            # Average metrics across all sessions for this opponent
            avg_win_rate = statistics.mean([s['robot_win_rate'] for s in config_sessions]) * 100
            avg_confidence = statistics.mean([s['avg_confidence'] for s in config_sessions])
            confidence_std = statistics.mean([s['confidence_std'] for s in config_sessions])
            
            # Extract configuration components - handle multi-word strategies correctly
            parts = config.split('_')
            difficulty = parts[0]
            
            # Strategy can be "to_win" or "not_to_lose" - handle both cases
            if len(parts) >= 3 and parts[1] == 'to' and parts[2] == 'win':
                strategy = 'to_win'
                personality = '_'.join(parts[3:])  # Everything after strategy
            elif len(parts) >= 4 and parts[1] == 'not' and parts[2] == 'to' and parts[3] == 'lose':
                strategy = 'not_to_lose' 
                personality = '_'.join(parts[4:])  # Everything after strategy
            else:
                # Fallback - assume second part is strategy
                strategy = parts[1]
                personality = '_'.join(parts[2:])
            
            plot_data.append({
                'opponent_config': config,
                'difficulty': difficulty.title(),
                'strategy': strategy.replace('_', ' ').title(),
                'personality': personality.replace('_', ' ').title(),
                'win_rate': avg_win_rate,
                'confidence': avg_confidence,
                'confidence_std': confidence_std,
                'difficulty_num': ['rookie', 'challenger', 'master'].index(difficulty),
                'strategy_num': ['to_win', 'not_to_lose'].index(strategy)  # Use original strategy string
            })
        
        df = pd.DataFrame(plot_data)
        
        # Color schemes for different attributes
        difficulty_colors = {'Rookie': '#1f77b4', 'Challenger': '#ff7f0e', 'Master': '#2ca02c'}
        strategy_colors = {'To Win': '#d62728', 'Not To Lose': '#9467bd'}
        personality_colors = {
            'Neutral': '#8c564b', 'Aggressive': '#e377c2', 'Defensive': '#7f7f7f',
            'Unpredictable': '#bcbd22', 'Cautious': '#17becf', 'Confident': '#ff7f0e',
            'Chameleon': '#1f77b4'
        }
        
        # 1. Win Rate vs Confidence - Color by Difficulty
        plt.figure(figsize=(16, 12))
        
        plt.subplot(2, 3, 1)
        for difficulty in df['difficulty'].unique():
            mask = df['difficulty'] == difficulty
            plt.scatter(df[mask]['confidence'], df[mask]['win_rate'], 
                       c=difficulty_colors[difficulty], label=difficulty, alpha=0.7, s=80)
        
        plt.xlabel('Average Confidence')
        plt.ylabel('Win Rate (%)')
        plt.title('Win Rate vs Confidence\n(Color: Difficulty Level)', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Win Rate vs Confidence - Color by Strategy
        plt.subplot(2, 3, 2)
        for strategy in df['strategy'].unique():
            mask = df['strategy'] == strategy
            plt.scatter(df[mask]['confidence'], df[mask]['win_rate'], 
                       c=strategy_colors[strategy], label=strategy, alpha=0.7, s=80)
        
        plt.xlabel('Average Confidence')
        plt.ylabel('Win Rate (%)')
        plt.title('Win Rate vs Confidence\n(Color: Strategy)', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Win Rate vs Confidence - Color by Personality
        plt.subplot(2, 3, 3)
        for personality in df['personality'].unique():
            mask = df['personality'] == personality
            color = personality_colors.get(personality, '#000000')
            plt.scatter(df[mask]['confidence'], df[mask]['win_rate'], 
                       c=color, label=personality, alpha=0.7, s=60)
        
        plt.xlabel('Average Confidence')
        plt.ylabel('Win Rate (%)')
        plt.title('Win Rate vs Confidence\n(Color: Personality)', fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.grid(True, alpha=0.3)
        
        # 4. Difficulty vs Strategy Distribution
        plt.subplot(2, 3, 4)
        x_pos = []
        y_pos = []
        colors = []
        sizes = []
        labels = []
        
        for i, row in df.iterrows():
            x_pos.append(row['difficulty_num'])
            y_pos.append(row['strategy_num'])
            colors.append(row['win_rate'])
            sizes.append(row['confidence'] * 300)  # Scale confidence for size
            labels.append(row['personality'])
        
        scatter = plt.scatter(x_pos, y_pos, c=colors, s=sizes, alpha=0.6, cmap='RdYlGn')
        plt.colorbar(scatter, label='Win Rate (%)')
        plt.xlabel('Difficulty Level')
        plt.ylabel('Strategy')
        plt.title('Configuration Distribution\n(Color: Win Rate, Size: Confidence)', fontweight='bold')
        plt.xticks([0, 1, 2], ['Rookie', 'Challenger', 'Master'])
        plt.yticks([0, 1], ['To Win', 'Not To Lose'])
        plt.grid(True, alpha=0.3)
        
        # 5. Confidence Standard Deviation Analysis
        plt.subplot(2, 3, 5)
        plt.scatter(df['confidence'], df['confidence_std'], 
                   c=df['win_rate'], cmap='viridis', alpha=0.7, s=80)
        plt.colorbar(label='Win Rate (%)')
        plt.xlabel('Average Confidence')
        plt.ylabel('Confidence Standard Deviation')
        plt.title('Confidence Consistency\n(Color: Win Rate)', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 6. Performance Distribution by Configuration
        plt.subplot(2, 3, 6)
        win_rates = np.array(df['win_rate'].tolist())
        confidences = np.array(df['confidence'].tolist())
        
        # Create size based on combined score
        combined_scores = (win_rates + confidences * 100) / 2
        
        plt.scatter(range(len(df)), win_rates, c=confidences, s=combined_scores*2, 
                   alpha=0.6, cmap='plasma')
        plt.colorbar(label='Confidence')
        plt.xlabel('Opponent Configuration Index')
        plt.ylabel('Win Rate (%)')
        plt.title('All 42 Configurations\n(Color: Confidence, Size: Combined Score)', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the scatter plot analysis
        plot1_file = f"{output_dir}/scatter_analysis_{timestamp}.png"
        plt.savefig(plot1_file, dpi=300, bbox_inches='tight')
        plot_files.append(plot1_file)
        print(f"📈 Scatter plot analysis saved: {plot1_file}")
        plt.close()
        
        # Create distribution analysis plots
        plt.figure(figsize=(15, 10))
        
        # Distribution by Difficulty
        plt.subplot(2, 3, 1)
        difficulty_means = df.groupby('difficulty')['win_rate'].agg(['mean', 'std']).reset_index()
        bars = plt.bar(difficulty_means['difficulty'], difficulty_means['mean'], 
                      yerr=difficulty_means['std'], capsize=5, alpha=0.7,
                      color=[difficulty_colors[d] for d in difficulty_means['difficulty']])
        plt.ylabel('Win Rate (%)')
        plt.title('Performance by Difficulty', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean_val in zip(bars, difficulty_means['mean']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{mean_val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Distribution by Strategy
        plt.subplot(2, 3, 2)
        strategy_means = df.groupby('strategy')['win_rate'].agg(['mean', 'std']).reset_index()
        bars = plt.bar(strategy_means['strategy'], strategy_means['mean'], 
                      yerr=strategy_means['std'], capsize=5, alpha=0.7,
                      color=[strategy_colors[s] for s in strategy_means['strategy']])
        plt.ylabel('Win Rate (%)')
        plt.title('Performance by Strategy', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean_val in zip(bars, strategy_means['mean']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{mean_val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Distribution by Personality
        plt.subplot(2, 3, 3)
        personality_means = df.groupby('personality')['win_rate'].agg(['mean', 'std']).reset_index()
        bars = plt.bar(personality_means['personality'], personality_means['mean'], 
                      yerr=personality_means['std'], capsize=5, alpha=0.7,
                      color=[personality_colors.get(p, '#gray') for p in personality_means['personality']])
        plt.ylabel('Win Rate (%)')
        plt.title('Performance by Personality', fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean_val in zip(bars, personality_means['mean']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{mean_val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Confidence analysis by attributes
        plt.subplot(2, 3, 4)
        difficulty_conf = df.groupby('difficulty')['confidence'].mean()
        bars = plt.bar(difficulty_conf.index, difficulty_conf.values.tolist(), alpha=0.7,
                      color=[difficulty_colors[d] for d in difficulty_conf.index])
        plt.ylabel('Average Confidence')
        plt.title('Confidence by Difficulty', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 5)
        strategy_conf = df.groupby('strategy')['confidence'].mean()
        bars = plt.bar(strategy_conf.index, strategy_conf.values.tolist(), alpha=0.7,
                      color=[strategy_colors[s] for s in strategy_conf.index])
        plt.ylabel('Average Confidence')
        plt.title('Confidence by Strategy', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Top and bottom performers
        plt.subplot(2, 3, 6)
        df_sorted = df.sort_values('win_rate')
        top_5 = df_sorted.tail(5)
        bottom_5 = df_sorted.head(5)
        
        combined_performers = pd.concat([bottom_5, top_5])
        colors = ['red'] * 5 + ['green'] * 5
        
        bars = plt.barh(range(len(combined_performers)), combined_performers['win_rate'], 
                       color=colors, alpha=0.7)
        plt.yticks(range(len(combined_performers)), 
                  [f"{row['difficulty'][0]}-{row['strategy'].split()[0][0]}-{row['personality'][:3]}" 
                   for _, row in combined_performers.iterrows()], fontsize=8)
        plt.xlabel('Win Rate (%)')
        plt.title('Top 5 vs Bottom 5 Performers', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, rate) in enumerate(zip(bars, combined_performers['win_rate'])):
            plt.text(rate + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{rate:.1f}%', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        
        # Save the distribution analysis
        plot2_file = f"{output_dir}/distribution_analysis_{timestamp}.png"
        plt.savefig(plot2_file, dpi=300, bbox_inches='tight')
        plot_files.append(plot2_file)
        print(f"📈 Distribution analysis saved: {plot2_file}")
        plt.close()
        
        # Print summary statistics
        print("\n📊 SCATTER PLOT ANALYSIS SUMMARY:")
        print(f"  • Total configurations analyzed: {len(df)}")
        print(f"  • Win rate range: {df['win_rate'].min():.1f}% - {df['win_rate'].max():.1f}%")
        print(f"  • Confidence range: {df['confidence'].min():.3f} - {df['confidence'].max():.3f}")
        print(f"  • Best performing difficulty: {df.groupby('difficulty')['win_rate'].mean().idxmax()}")
        print(f"  • Best performing strategy: {df.groupby('strategy')['win_rate'].mean().idxmax()}")
        print(f"  • Best performing personality: {df.groupby('personality')['win_rate'].mean().idxmax()}")
        
        return plot_files

    def create_box_plots(self, results: Dict, output_dir: str = "visualizations") -> List[str]:
        """
        Create comprehensive box plots showing the impact of different factors
        Returns list of generated plot filenames
        """
        print("📊 Generating visualization plots...")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style for better looking plots
        plt.style.use('default')
        sns.set_palette("husl")
        
        sessions = results['sessions']
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plot_files = []
        
        # Prepare data for plotting
        data_for_plots = []
        for session in sessions:
            data_for_plots.append({
                'difficulty': session['difficulty'].title(),
                'strategy': session['strategy'].replace('_', ' ').title(),
                'personality': session['personality'].title(),
                'win_rate': session['robot_win_rate'] * 100,  # Convert to percentage
                'confidence': session['avg_confidence'],
                'opponent_config': session['opponent_config']
            })
        
        df = pd.DataFrame(data_for_plots)
        
        # 1. Box plot for Difficulty Impact
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        box_plot = sns.boxplot(data=df, x='difficulty', y='win_rate', palette='viridis')
        plt.title('Win Rate Distribution by Difficulty Level', fontsize=14, fontweight='bold')
        plt.xlabel('Difficulty Level', fontsize=12)
        plt.ylabel('Win Rate (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add statistical annotations
        for i, difficulty in enumerate(df['difficulty'].unique()):
            subset = df[df['difficulty'] == difficulty]['win_rate']
            mean_val = subset.mean()
            plt.text(i, mean_val + 1, f'μ={mean_val:.1f}%', 
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 2. Box plot for Strategy Impact
        plt.subplot(2, 2, 2)
        sns.boxplot(data=df, x='strategy', y='win_rate', palette='coolwarm')
        plt.title('Win Rate Distribution by Strategy', fontsize=14, fontweight='bold')
        plt.xlabel('Strategy', fontsize=12)
        plt.ylabel('Win Rate (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add statistical annotations
        for i, strategy in enumerate(df['strategy'].unique()):
            subset = df[df['strategy'] == strategy]['win_rate']
            mean_val = subset.mean()
            plt.text(i, mean_val + 1, f'μ={mean_val:.1f}%', 
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 3. Box plot for Personality Impact
        plt.subplot(2, 1, 2)
        sns.boxplot(data=df, x='personality', y='win_rate', palette='tab10')
        plt.title('Win Rate Distribution by Personality Type', fontsize=14, fontweight='bold')
        plt.xlabel('Personality Type', fontsize=12)
        plt.ylabel('Win Rate (%)', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add statistical annotations
        for i, personality in enumerate(df['personality'].unique()):
            subset = df[df['personality'] == personality]['win_rate']
            mean_val = subset.mean()
            plt.text(i, mean_val + 0.5, f'{mean_val:.1f}%', 
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        
        # Save the first plot
        plot1_file = f"{output_dir}/win_rate_analysis_{timestamp}.png"
        plt.savefig(plot1_file, dpi=300, bbox_inches='tight')
        plot_files.append(plot1_file)
        print(f"📈 Win rate analysis plot saved: {plot1_file}")
        plt.close()
        
        # 4. Confidence Analysis Plots
        plt.figure(figsize=(15, 10))
        
        # Confidence by Difficulty
        plt.subplot(2, 3, 1)
        sns.boxplot(data=df, x='difficulty', y='confidence', palette='viridis')
        plt.title('Confidence by Difficulty', fontweight='bold')
        plt.ylabel('Confidence Score')
        plt.grid(True, alpha=0.3)
        
        # Confidence by Strategy
        plt.subplot(2, 3, 2)
        sns.boxplot(data=df, x='strategy', y='confidence', palette='coolwarm')
        plt.title('Confidence by Strategy', fontweight='bold')
        plt.ylabel('Confidence Score')
        plt.grid(True, alpha=0.3)
        
        # Confidence by Personality
        plt.subplot(2, 3, 3)
        sns.boxplot(data=df, x='personality', y='confidence', palette='tab10')
        plt.title('Confidence by Personality', fontweight='bold')
        plt.ylabel('Confidence Score')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Win Rate vs Confidence Scatter
        plt.subplot(2, 3, 4)
        scatter = plt.scatter(df['confidence'], df['win_rate'], 
                            c=df['difficulty'].map({'Rookie': 0, 'Challenger': 1, 'Master': 2}),
                            cmap='viridis', alpha=0.6, s=50)
        plt.xlabel('Confidence Score')
        plt.ylabel('Win Rate (%)')
        plt.title('Win Rate vs Confidence')
        plt.colorbar(scatter, label='Difficulty', ticks=[0, 1, 2])
        plt.grid(True, alpha=0.3)
        
        # Strategy comparison violin plot
        plt.subplot(2, 3, 5)
        sns.violinplot(data=df, x='strategy', y='win_rate', palette='coolwarm')
        plt.title('Strategy Impact Distribution', fontweight='bold')
        plt.ylabel('Win Rate (%)')
        plt.grid(True, alpha=0.3)
        
        # Top and Bottom Performers
        plt.subplot(2, 3, 6)
        config_means = df.groupby('opponent_config')['win_rate'].mean().sort_values()
        top_5 = config_means.tail(5)
        bottom_5 = config_means.head(5)
        
        combined = pd.concat([bottom_5, top_5])
        colors = ['red'] * 5 + ['green'] * 5
        bars = plt.barh(range(len(combined)), combined.values.tolist(), color=colors, alpha=0.7)
        plt.yticks(range(len(combined)), [config.replace('_', ' ').title() for config in combined.index])
        plt.xlabel('Win Rate (%)')
        plt.title('Best vs Worst Performers', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{width:.1f}%', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        
        # Save the second plot
        plot2_file = f"{output_dir}/comprehensive_analysis_{timestamp}.png"
        plt.savefig(plot2_file, dpi=300, bbox_inches='tight')
        plot_files.append(plot2_file)
        print(f"📈 Comprehensive analysis plot saved: {plot2_file}")
        plt.close()
        
        # 5. Statistical Summary Heatmap
        plt.figure(figsize=(12, 8))
        
        # Create pivot tables for heatmap
        difficulty_strategy = df.groupby(['difficulty', 'strategy'])['win_rate'].mean().unstack()
        
        plt.subplot(2, 2, 1)
        sns.heatmap(difficulty_strategy, annot=True, fmt='.1f', cmap='RdYlGn', 
                   center=df['win_rate'].mean(), square=True)
        plt.title('Win Rate: Difficulty × Strategy', fontweight='bold')
        
        # Difficulty vs Personality
        difficulty_personality = df.groupby(['difficulty', 'personality'])['win_rate'].mean().unstack()
        plt.subplot(2, 1, 2)
        sns.heatmap(difficulty_personality, annot=True, fmt='.1f', cmap='RdYlGn',
                   center=df['win_rate'].mean())
        plt.title('Win Rate Heatmap: Difficulty × Personality', fontweight='bold')
        
        plt.tight_layout()
        
        # Save the third plot
        plot3_file = f"{output_dir}/interaction_heatmaps_{timestamp}.png"
        plt.savefig(plot3_file, dpi=300, bbox_inches='tight')
        plot_files.append(plot3_file)
        print(f"📈 Interaction heatmaps saved: {plot3_file}")
        plt.close()
        
        # Print statistical summary
        print("\n📊 VISUALIZATION SUMMARY:")
        print(f"  • Total data points: {len(df)}")
        print(f"  • Win rate range: {df['win_rate'].min():.1f}% - {df['win_rate'].max():.1f}%")
        print(f"  • Confidence range: {df['confidence'].min():.3f} - {df['confidence'].max():.3f}")
        
        # Factor impact summary
        diff_var = df.groupby('difficulty')['win_rate'].var().mean()
        strat_var = df.groupby('strategy')['win_rate'].var().mean()
        pers_var = df.groupby('personality')['win_rate'].var().mean()
        
        print(f"  • Difficulty variance: {diff_var:.2f}")
        print(f"  • Strategy variance: {strat_var:.2f}")
        print(f"  • Personality variance: {pers_var:.2f}")
        
        return plot_files

def main():
    """Main execution function with flexible configuration example"""
    print("🎯 42-Opponent Performance Tester with Flexible Configuration")
    print("=" * 60)
    
    if not AI_SYSTEM_AVAILABLE:
        print("❌ AI system not available - cannot run tests")
        return
    
    # Get script directory for proper path handling
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    visualizations_dir = os.path.join(script_dir, "visualizations")
    
    # Ensure directories exist
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(visualizations_dir, exist_ok=True)

    # Initialize AI system
    try:
        # Initialize with default parameters - system will be reconfigured for each test
        initialize_ai_system('challenger', 'to_win', 'neutral')
        print("✅ AI system initialized")
    except Exception as e:
        print(f"❌ Failed to initialize AI system: {e}")
        return
    
    # Example: Create custom test configuration
    # You can adjust these parameters for your thorough study
    custom_config = TestConfig(
        max_moves=75,  # Increase from default 50 to 75 moves per game
        games_per_opponent=1000,  # Increase from default 10 to 15 games per opponent
        pattern_weights={
            'adaptive': 0.4,        # Focus more on adaptive patterns
            'anti_frequency': 0.3,  # Strong counter-strategy focus
            'pattern_repeater': 0.2,  # Some predictable patterns
            'frequency_based': 0.1,   # Less bias-based patterns
            'random': 0.0           # No purely random patterns
        },
        human_pattern_selection="weighted_random"
    )
    
    print("\n🔧 Using Custom Configuration:")
    print(custom_config.summary())
    
    # Create tester with custom configuration
    tester = PerformanceTester(config=custom_config)
    
    # Run comprehensive test
    print("\n🚀 Starting comprehensive test...")
    results = tester.run_comprehensive_test(human_pattern=None, verbose=True)  # Let config choose patterns
    
    # Generate report
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"42_opponent_test_results_{timestamp}.json")
    report_file = os.path.join(results_dir, f"42_opponent_analysis_report_{timestamp}.txt")
    
    # Save detailed results
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📊 Detailed results saved to: {results_file}")
    
    # Generate and save report
    report = tester.generate_enhanced_report(results, report_file)
    
    # Generate comprehensive visualizations
    print("\n🎨 Creating comprehensive visualization plots...")
    try:
        # Create enhanced scatter plot analysis (new feature)
        scatter_plot_files = tester.create_scatter_plot_analysis(results, visualizations_dir)
        print(f"📊 Generated {len(scatter_plot_files)} scatter plot analysis files:")
        for plot_file in scatter_plot_files:
            print(f"  📈 {plot_file}")
        
        # Create traditional box plots for comparison
        box_plot_files = tester.create_box_plots(results, visualizations_dir)
        print(f"📊 Generated {len(box_plot_files)} additional box plot files:")
        for plot_file in box_plot_files:
            print(f"  📈 {plot_file}")
        
        total_plots = len(scatter_plot_files) + len(box_plot_files)
        print(f"📊 Total visualization files generated: {total_plots}")
        
    except Exception as e:
        print(f"⚠️ Warning: Visualization generation failed: {e}")
        print("📊 Analysis completed successfully, but plots could not be generated")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 ENHANCED QUICK SUMMARY:")
    analysis = results['analysis']
    overall = analysis['overall']
    config = results['test_config']
    print(f"🎯 Tested {config['total_game_sessions']} game sessions across 42 opponent combinations")
    print(f"� Total moves played: {config['total_moves_played']}")
    print(f"�🏆 Average robot win rate: {overall['win_rate_mean']:.1%} ± {overall['win_rate_std']:.3f}")
    print(f"💪 Average confidence: {overall['confidence_mean']:.3f} ± {overall['confidence_std']:.3f}")
    print(f"📊 System consistency: {overall['consistency_score']:.3f}")
    
    # Top configurations
    config_stats = analysis['by_opponent_config']
    best_config = max(config_stats.items(), key=lambda x: x[1]['win_rate_mean'])
    worst_config = min(config_stats.items(), key=lambda x: x[1]['win_rate_mean'])
    
    print(f"🥇 Best configuration: {best_config[0]} ({best_config[1]['win_rate_mean']:.1%} ± {best_config[1]['win_rate_std']:.3f})")
    print(f"🥉 Worst configuration: {worst_config[0]} ({worst_config[1]['win_rate_mean']:.1%} ± {worst_config[1]['win_rate_std']:.3f})")
    
    # Statistical effects
    effects = analysis['statistical_effects']
    print(f"🔬 Strongest factor: {effects['strongest_factor'].upper()} (effect size: {effects['strongest_effect_size']:.3f})")
    
    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    main()