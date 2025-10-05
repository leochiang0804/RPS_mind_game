# 🎮 Paper-Scissor-Stone: Advanced AI Mind Game

An intelligent Rock-Paper-Scissors game featuring advanced machine learning strategies, real-time analytics, visual character system, personality-based AI dialogue, and comprehensive performance monitoring. This project demonstrates sophisticated pattern recognition, character visualization, conversational AI, and immersive gaming experiences.

## 🚀 Current Features

### ✅ **Core ML Foundation**
- **Enhanced ML Strategies**: Multi-layered AI with Random, Frequency, Markov, and LSTM models
- **Change-Point Detection**: Real-time strategy change detection using chi-squared testing
- **Pattern Recognition**: Advanced sequence analysis and prediction confidence scoring
- **Adaptive Learning**: Models update and evolve based on player behavior patterns

### ✅ **Intelligence & Analytics**
- **Real-Time Analytics**: Live charts for move distribution, win rates, and strategy usage
- **Performance Monitoring**: Comprehensive system performance tracking and optimization
- **Developer Console**: Advanced debugging and analytics tools for development
- **Unified Metrics**: Single source of truth for all game statistics and insights

### ✅ **Visual & Interactive Features**
- **Visual Battle Arena**: Dynamic move display with character avatars and winner highlighting
- **Personality System**: Multiple robot personalities with unique behaviors and dialogue
- **Real-Time Visualizations**: 
  - Move Distribution: Pie chart showing player move preferences
  - Win Rate Trends: Line chart tracking performance over time
  - Strategy Timeline: Bar chart showing AI strategy usage
- **Conversation Interface**: Direct text banter with mood indicators

### ✅ **Advanced AI Features**
- **LSTM Neural Network**: Deep learning for complex pattern recognition
- **Confidence Scoring**: Prediction confidence metrics for all models
- **Strategy Optimization**: ToWin and NotToLose advanced strategy implementations
- **Context-Aware Analysis**: Unified game context building for consistent data across all features

## 🧠 AI Architecture

### Machine Learning Models
- **Frequency Strategy**: Analyzes move frequency patterns
- **Markov Strategy**: Uses state-based prediction modeling
- **Enhanced ML Model**: Advanced pattern recognition with multiple algorithms
- **LSTM Integration**: Neural network for deep sequence learning

### Intelligence Systems
- **Change Point Detector**: Identifies when players switch strategies
- **Performance Optimizer**: Real-time system optimization and monitoring
- **Game Context Builder**: Centralized data management and metrics calculation

## 🎨 Visual Features

### Character System
- **Character Avatars**: Visual feedback based on difficulty and personality
- **Battle Arena**: Professional move visualization and winner highlighting
- **Mood System**: Visual indicators showing robot emotional state

### Real-Time Interface
- **Live Analytics Dashboard**: Dynamic charts updating with each move
- **Strategy Insights**: Real-time display of AI decision-making process
- **Performance Metrics**: Live system performance and accuracy tracking

## 🕹️ How to Play

### Installation & Setup

```bash
# Clone the repository
git clone https://github.com/leochiang0804/Paper_Scissor_Stone_a_Mind_game.git
cd Paper_Scissor_Stone

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the web application
python webapp/app.py
```

### Game Controls
1. Open http://127.0.0.1:5050 in your browser
2. Select AI difficulty and personality
3. Play using keyboard shortcuts or click buttons:
   - **P**: Play Paper
   - **R**: Play Rock
   - **S**: Play Scissors
4. View real-time analytics and robot dialogue
5. Monitor your performance against AI predictions

## 📁 Project Structure

```
Paper_Scissor_Stone/
├── webapp/
│   ├── app.py              # Main Flask application with unified metrics
│   ├── templates/
│   │   ├── index.html      # Main web interface
│   │   ├── developer_console.html # Developer analytics interface
│   │   └── performance_dashboard.html # Performance monitoring
│   └── static/             # Game assets and styling
├── opponent_tester/        # 42-Opponent Testing Framework
│   ├── opponent_performance_tester.py # Comprehensive testing suite
│   ├── verify_dynamic_params.py # Parameter validation utility
│   ├── view_plot.py        # Visualization viewer
│   ├── results/            # Test results and reports
│   ├── visualizations/     # Generated plots and charts
│   └── README.md           # Testing framework documentation
├── strategy.py             # Core ML strategies implementation
├── ml_model_enhanced.py    # Enhanced ML models and algorithms
├── game_context.py         # Unified game context and metrics builder
├── change_point_detector.py # Real-time strategy change detection
├── lstm_model.py           # LSTM neural network implementation
├── lstm_web_integration.py # LSTM web integration layer
├── personality_engine.py   # Robot personality system
├── optimized_strategies.py # Advanced strategy implementations
├── move_mapping.py         # Move normalization and mapping utilities
├── developer_console.py    # Development and debugging tools
├── performance_optimizer.py # System performance monitoring
└── requirements.txt        # Python dependencies
```

## 🔧 Technical Features

### Backend Architecture
- **Flask Web Application**: Robust web server with session management
- **Unified Metrics System**: Single source of truth for all game statistics
- **Modular AI Components**: Plug-and-play strategy system
- **Real-Time Data Processing**: Live analytics and performance monitoring

### Frontend Features
- **Responsive Design**: Modern web interface with real-time updates
- **Interactive Visualizations**: Dynamic charts and visual feedback
- **Live Analytics**: Real-time metrics display without page refresh
- **End-Game Optimization**: Proper timing for final statistics display

### Optional Components
- **LSTM Model**: Advanced neural network (optional, fails gracefully if unavailable)
- **Developer Console**: Advanced debugging tools (optional)
- **Performance Monitoring**: System optimization tracking (optional)

## 🧪 Testing

### Test Scripts
All test scripts are located in the `tests/` directory:

- **`strategy_comparison_test.py`**: Comprehensive analysis of AI strategy behavior and confidence scoring across different difficulties and personalities
- **`test_ai_simplified.py`**: Simplified AI confidence and strategy testing with direct simulation of webapp robot_strategy function
- **`api_baseline.py`**: API baseline testing for regression detection
- **`capture_baseline.py`**: Captures baseline performance metrics
- **`simple_baseline.py`**: Simple baseline performance testing

### Running Tests
```bash
# Run strategy comparison analysis
cd tests/
python strategy_comparison_test.py

# Run simplified AI testing
python test_ai_simplified.py

# Run baseline tests
python api_baseline.py
```

### Test Features
- **Confidence Score Analysis**: Tests AI confidence patterns across all personalities and difficulties
- **Strategy Validation**: Verifies ToWin vs NotToLose strategy differentiation
- **LSTM Integration Testing**: Validates neural network model integration
- **Regression Testing**: Monitors for performance regressions

## 🚀 Recent Improvements

### Performance & Reliability
- **Unified Backend Metrics**: Eliminated dual calculation systems for consistent data
- **Real-Time Updates**: Fixed metrics updating in real-time during gameplay
- **End-Game Timing**: Resolved final move statistics display before game freeze
- **Code Cleanup**: Removed unused standalone scripts for cleaner codebase

### Architecture Enhancements
- **Single Source of Truth**: All metrics calculated in backend via `game_context.py`
- **Graceful Degradation**: Optional components fail gracefully without breaking core functionality
- **Optimized Data Flow**: Streamlined data processing for better performance

## 🤝 Contributing

This is an educational project demonstrating advanced AI concepts in game development. The codebase showcases:
- Machine learning integration in web applications
- Real-time data visualization
- Modular AI architecture
- Performance optimization techniques

Feel free to explore, experiment, and contribute!

## 📄 License

Open source for educational purposes.