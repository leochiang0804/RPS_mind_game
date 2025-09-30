# 🎮 Paper-Scissor-Stone: Advanced AI Mind Game

An intelligent Rock-Paper-Scissors game featuring advanced machine learning strategies, real-time analytics, tournament systems, and comprehensive AI behavior analysis. This project demonstrates sophisticated pattern recognition, change-point detection, and competitive gaming features.

## 🚀 Current Features

### ✅ **Phase 1: Core ML Foundation (COMPLETED)**
- **1.1 Enhanced ML Strategies**: Multi-layered AI with Random, Frequency, Markov, and Enhanced ML models
- **1.2 Change-Point Detection**: Real-time strategy change detection using chi-squared testing
- **1.3 Pattern Recognition**: Advanced sequence analysis and prediction confidence scoring

### ✅ **Phase 2: Intelligence & Coaching (COMPLETED)**
- **2.1 Enhanced UI/UX**: Interactive web interface with strategy timeline visualization
- **2.2 Coach Tips Generator**: Intelligent coaching system analyzing player patterns and providing strategic advice
- **2.3 Advanced Analytics Dashboard**: Comprehensive metrics including entropy calculations, predictability scoring, and JSON export

### ✅ **Phase 3: Advanced Features (IN PROGRESS)**
- **3.1 Visual Charts Integration**: Real-time Chart.js visualizations with model prediction tracking ✅
- **3.2 ML Model Comparison Dashboard**: Performance metrics, accuracy graphs, confidence trends, and smart recommendations ✅
- **3.3 Multiplayer Tournament System**: ELO ratings, leaderboards, match history, and competitive brackets ✅
- **3.4 AI Personality Modes**: Aggressive, Defensive, Adaptive behavioral patterns (PENDING)
- **3.5 Game Replay & Analysis System**: Move-by-move playback with strategy analysis (PENDING)

## 🎯 Key Technologies

- **Backend**: Flask, Python, scikit-learn, NumPy
- **Frontend**: HTML5, CSS3, JavaScript, Chart.js
- **AI/ML**: Markov Chains, Frequency Analysis, Enhanced ML with confidence scoring
- **Analytics**: Change-point detection, entropy calculations, pattern recognition
- **Tournament**: ELO rating system, persistent data storage, match management

## 🏆 Game Modes

### 1. **Single Player vs AI**
- Choose from 4 AI difficulty levels: Random, Frequency, Markov, Enhanced
- Real-time coaching tips and pattern analysis
- Performance tracking and strategy recommendations

### 2. **Tournament Mode**
- Create players with ELO ratings
- Competitive match system (best-of-5)
- Live leaderboards and statistics
- Match history and player profiles

### 3. **Analytics Mode**
- Model prediction tracking charts
- Strategy performance comparison
- Confidence trend analysis
- Exportable data for further analysis

## 📊 Advanced Analytics Features

### Real-Time Visualizations
- **Move Distribution**: Pie chart showing player move preferences
- **Win Rate Trends**: Line chart tracking performance over time
- **Strategy Timeline**: Bar chart showing AI strategy usage
- **Model Prediction Tracking**: Multi-line chart comparing human vs robot vs all model predictions
- **Confidence Analysis**: Real-time confidence scoring for all AI models

### Intelligent Coaching
- Pattern recognition and analysis
- Strategic recommendations based on gameplay
- Change-point detection for strategy shifts
- Personalized tips for improvement

### Performance Metrics
- Accuracy tracking for all AI models
- Prediction confidence trends
- Model comparison dashboard
- Smart strategy recommendations

## 🎮 How to Play

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
- **P**: Play Paper
- **R**: Play Rock  
- **S**: Play Scissor
- **🏆 Tournament**: Access tournament system
- **📊 Export Data**: Download analytics data
- **🔄 Reset Game**: Start fresh session

### Web Interface
1. Open http://127.0.0.1:5000 in your browser
2. Select AI difficulty level
3. Use keyboard keys (P/R/S) or click buttons to play
4. View real-time analytics and coaching tips
5. Access tournament mode for competitive play

## 🧠 AI Strategy Details

### Enhanced ML Strategy
- **Sequence Analysis**: Analyzes move patterns and sequences
- **Confidence Scoring**: Provides prediction confidence levels
- **Adaptive Learning**: Continuously updates based on player behavior
- **Pattern Recognition**: Identifies recurring behavioral patterns

### Change-Point Detection
- **Chi-squared Testing**: Statistical analysis of strategy changes
- **Window-based Analysis**: Sliding window for pattern detection
- **Real-time Adaptation**: Dynamic strategy switching based on detected changes

### Tournament ELO System
- **Standard ELO Implementation**: Chess-style rating calculations
- **K-factor Adjustment**: Balanced rating changes
- **Persistent Rankings**: Saved player statistics and history

## 📁 Project Structure

```
Paper_Scissor_Stone/
├── webapp/
│   ├── app.py              # Main Flask application
│   ├── templates/
│   │   └── index_working.html  # Main web interface
│   └── static/             # Game assets (images)
├── strategy.py             # ML strategy implementations
├── change_point_detector.py # Strategy change detection
├── coach_tips.py          # Intelligent coaching system
├── tournament_system.py   # Tournament and ELO management
├── ml_model.py           # Core ML models
├── stats_manager.py      # Statistics and analytics
├── visualizer.py         # Data visualization tools
├── main.py              # CLI version
└── requirements.txt     # Python dependencies
```

## 🎯 Development Roadmap

### Current Status: **Phase 3 - Advanced Features**
- ✅ Visual Charts Integration (3.1)
- ✅ ML Model Comparison Dashboard (3.2) 
- ✅ Tournament System (3.3)
- 🔄 AI Personality Modes (3.4) - IN DEVELOPMENT
- ⏳ Game Replay System (3.5) - PLANNED

### Future Enhancements
- Advanced tournament brackets
- Multiplayer real-time matches
- AI personality customization
- Mobile-responsive design
- Cloud deployment options

## 📈 Performance Metrics

The system tracks comprehensive performance metrics:
- **Model Accuracy**: Real-time accuracy for all AI strategies
- **Prediction Confidence**: Confidence levels for each prediction
- **Pattern Analysis**: Entropy and predictability calculations
- **Player Statistics**: Win rates, move distributions, strategy effectiveness
- **Tournament Stats**: ELO ratings, match history, leaderboard rankings

## 🤝 Contributing

This is an educational project demonstrating advanced AI concepts in game development. Feel free to explore the code and experiment with different AI strategies!

## 📄 License

This project is open source and available for educational purposes.

---

**🎮 Ready to challenge the AI? Your move!**
