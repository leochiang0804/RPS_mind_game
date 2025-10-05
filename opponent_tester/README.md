# 🎯 Opponent Testing Framework

This folder contains all testing scripts, results, and visualizations for the 42-opponent RPS AI system.

## 📁 Structure

```
opponent_tester/
├── README.md                          # This documentation
├── opponent_performance_tester.py     # Main testing script
├── verify_dynamic_params.py           # Parameter verification utility
├── view_plot.py                       # Plot viewing utility
├── COMPREHENSIVE_42_OPPONENT_ANALYSIS.md  # Detailed analysis documentation
├── results/                           # Test outputs
│   ├── 42_opponent_test_results_*.json    # Raw test data
│   └── 42_opponent_analysis_report_*.txt  # Analysis reports
└── visualizations/                    # Generated plots
    ├── win_rate_analysis_*.png        # Box plots by factor
    ├── comprehensive_analysis_*.png   # Multi-panel analysis
    └── interaction_heatmaps_*.png     # Factor interaction heatmaps
```

## 🚀 Usage

### Run Complete Testing Suite
```bash
cd opponent_tester
python3 opponent_performance_tester.py
```

This will:
- Test all 42 opponent combinations (3 difficulties × 2 strategies × 7 personalities)
- Run 10 games per opponent (50 moves each) = 420 total game sessions
- Generate statistical analysis with box plots and heatmaps
- Save results to `results/` and visualizations to `visualizations/`

### Verify Parameter Accuracy
```bash
python3 verify_dynamic_params.py
```

Confirms that tests use dynamic parameters from the real AI system, not static values.

### View Generated Plots
```bash
python3 view_plot.py
```

Display the latest visualization files.

## 📊 Test Configuration

- **Moves per Game**: 50
- **Games per Opponent**: 10
- **Total Game Sessions**: 420
- **Total Moves Tested**: 21,000
- **Human Pattern**: Adaptive (evolves during gameplay)

## 📈 Output Files

### Results (`results/`)
- **JSON files**: Raw test data with per-game statistics
- **TXT files**: Human-readable analysis reports with statistical insights

### Visualizations (`visualizations/`)
- **Win Rate Analysis**: Box plots showing factor impacts
- **Comprehensive Analysis**: Multi-panel analysis with confidence metrics
- **Interaction Heatmaps**: Factor interaction effects

## 🔧 Customization

To modify test parameters, edit `opponent_performance_tester.py`:

```python
class PerformanceTester:
    def __init__(self):
        self.max_games = 50          # Moves per game
        self.games_per_opponent = 10 # Games per opponent
```

## 📋 Recent Test Results

The testing framework has revealed:

1. **Strategy Impact**: Strongest factor (6.7% variance)
   - TO_WIN strategy: 47.2% win rate
   - NOT_TO_LOSE strategy: 37.7% win rate

2. **Difficulty Impact**: Moderate (1.0% variance)
   - Rookie: 41.3% win rate
   - Challenger: 43.1% win rate  
   - Master: 42.9% win rate

3. **Personality Impact**: Weak (0.9% variance)
   - Best: Unpredictable (43.6%)
   - Worst: Chameleon (41.1%)

## 🎯 Recommendations

1. **Enhance difficulty differentiation** - Master should be significantly harder
2. **Strengthen personality traits** - Amplify characteristic behaviors
3. **Balance strategies** - Make NOT_TO_LOSE more competitive

## 📞 Integration

The testing framework uses the real AI system components:
- `rps_ai_system.py` - Core AI logic
- `game_context.py` - Parameter management
- `parameter_synthesis_engine.py` - Dynamic parameter generation

This ensures test results accurately reflect actual application behavior.