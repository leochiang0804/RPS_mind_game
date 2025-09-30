# 🎮 Paper-Scissor-Stone: Advanced AI Mind Game

An intelligent Rock-Paper-Scissors game featuring advanced machine learning strategies, real-time analytics, visual character system, personality-based AI dialogue, and comprehensive battle arena interface. This project demonstrates sophisticated pattern recognition, character visualization, conversational AI, and immersive gaming experiences.

## 🚀 Current Features

### ✅ **Phase 1: Core ML Foundation (COMPLETED)**
- **1.1 Enhanced ML Strategies**: Multi-layered AI with Random, Frequency, Markov, and LSTM models
- **1.2 Change-Point Detection**: Real-time strategy change detection using chi-squared testing
- **1.3 Pattern Recognition**: Advanced sequence analysis and prediction confidence scoring

### ✅ **Phase 2: Intelligence & Coaching (COMPLETED)**
- **2.1 Enhanced UI/UX**: Interactive web interface with strategy timeline visualization
- **2.2 Coach Tips Generator**: Intelligent coaching system analyzing player patterns and providing strategic advice
- **2.3 Advanced Analytics Dashboard**: Comprehensive metrics including entropy calculations, predictability scoring, and JSON export

### ✅ **Phase 3: Advanced Features (COMPLETED)**
- **3.1 Visual Charts Integration**: Real-time Chart.js visualizations with model prediction tracking ✅
- **3.2 ML Model Comparison Dashboard**: Performance metrics, accuracy graphs, confidence trends, and smart recommendations ✅
- **3.3 Multiplayer Tournament System**: ELO ratings, leaderboards, match history, and competitive brackets ✅
- **3.4 AI Personality Modes**: 6 unique personalities with distinct behavioral patterns ✅
- **3.5 Visual Battle Arena**: Dynamic move display with winner highlighting and animations ✅

### ✅ **Phase 4: Visual & Character System (NEW - COMPLETED)**
- **4.1 Robot Character Visualization**: Dynamic character avatars combining difficulty, strategy, and personality
- **4.2 Personality-Based Dialogue System**: Real-time conversation with typing animations and mood indicators
- **4.3 Enhanced Battle Arena**: Rock/Paper/Scissors image display with winner highlighting
- **4.4 Dynamic Character Names**: Procedurally generated robot names based on personality and difficulty
- **4.5 Strategy Symbol System**: Visual indicators showing current AI strategy with symbolic representations

## 🎯 Key Technologies

- **Backend**: Flask, Python, scikit-learn, NumPy, TensorFlow/Keras (LSTM)
- **Frontend**: HTML5, CSS3, JavaScript, Chart.js, Dynamic Character System
- **AI/ML**: Markov Chains, Frequency Analysis, Enhanced ML, LSTM Neural Networks
- **Visual System**: Dynamic character avatars, personality-based dialogue, animated UI
- **Analytics**: Change-point detection, entropy calculations, pattern recognition
- **Tournament**: ELO rating system, persistent data storage, match management

## 🤖 AI Personality System

### 6 Unique Robot Personalities
Each personality features distinct dialogue, behavior patterns, and visual characteristics:

1. **💀 The Berserker** - Ultra aggressive, bloodthirsty combat style
2. **🛡️ The Guardian** - Defensive expert with honor-focused dialogue
3. **🦎 The Chameleon** - Adaptive strategy with evolving responses
4. **🎓 The Professor** - Analytical approach with scientific dialogue
5. **🃏 The Wildcard** - Unpredictable chaos with random elements
6. **🪞 The Mirror** - Mimicking strategy with reflection-based responses

### Dynamic Character Features
- **Visual Avatars**: Personality-specific emojis and character designs
- **Difficulty Shading**: Background colors indicating AI strength (Green→Yellow→Red→Dark)
- **Strategy Symbols**: Icons showing current AI strategy (🎲🧠⚔️📊🔄🤖)
- **Procedural Names**: Generated names like "Master Destroyer" or "Junior Scholar"
- **Mood System**: Real-time mood indicators reflecting game state and personality

## 🎮 Battle Arena System

### Enhanced Visual Interface
- **Move Display Window**: Shows human vs robot moves with rock/paper/scissors images
- **Winner Highlighting**: Dynamic border colors and glowing effects for winners
- **Animated Transitions**: Smooth timing with dramatic reveal sequences
- **Result Persistence**: Combat outcomes remain visible until next move
- **Fast Loading**: Optimized timing for responsive gameplay

### Conversation System
- **Real-time Dialogue**: Personality-based responses to game events
- **Typing Animation**: Realistic character-by-character text reveal
- **Speech Bubbles**: Professional chat interface with visual styling
- **Context Awareness**: Different responses for wins, losses, ties, and game start
- **Mood Indicators**: Visual emotes showing current robot emotional state

## 🏆 Game Modes

### 1. **Enhanced Single Player vs AI**
- Choose from 6 AI difficulty levels including LSTM neural network
- 6 unique AI personalities with distinct dialogue and behavior
- Visual character system showing opponent details
- Real-time coaching tips and pattern analysis
- Performance tracking and strategy recommendations
- Immersive battle arena with move visualizations

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
- **LSTM Performance Metrics**: Neural network accuracy and prediction tracking

### Intelligent Coaching
- Pattern recognition and analysis
- Strategic recommendations based on gameplay
- Change-point detection for strategy shifts
- Personalized tips for improvement

### Performance Metrics
- Accuracy tracking for all AI models including LSTM
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
- **🤖 Personality**: Select robot personality for different gameplay experience

### Web Interface
1. Open http://127.0.0.1:5000 in your browser
2. Select AI difficulty level and personality
3. Watch your robot opponent appear with visual character details
4. Use keyboard keys (P/R/S) or click buttons to play
5. Enjoy the battle arena with move visualizations and robot dialogue
6. View real-time analytics and coaching tips
7. Access tournament mode for competitive play

## 🧠 AI Strategy Details

### Enhanced ML Strategy with LSTM
- **Sequence Analysis**: Analyzes move patterns and sequences using advanced algorithms
- **LSTM Neural Network**: Deep learning model for complex pattern recognition
- **Confidence Scoring**: Provides prediction confidence levels for all models
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
│   ├── app.py              # Main Flask application with enhanced features
│   ├── templates/
│   │   └── index.html      # Main web interface with character system
│   └── static/             # Game assets (rock/paper/scissors images)
├── strategy.py             # ML strategy implementations with LSTM
├── ml_model_enhanced.py    # Consolidated ML models (Enhanced + LSTM)
├── change_point_detector.py # Strategy change detection
├── coach_tips.py          # Intelligent coaching system
├── tournament_system.py   # Tournament and ELO management  
├── stats_manager.py      # Statistics and analytics
├── visualizer.py         # Data visualization tools
├── lstm_web_integration.py # LSTM neural network integration
├── data_store.py         # Game data persistence
├── main.py              # CLI version
└── requirements.txt     # Python dependencies
```

## 🎯 Development Roadmap

### Current Status: **Phase 4 - Visual & Character System (COMPLETED)**
- ✅ Visual Charts Integration (3.1)
- ✅ ML Model Comparison Dashboard (3.2) 
- ✅ Tournament System (3.3)
- ✅ AI Personality Modes (3.4) - 6 unique personalities with dialogue system
- ✅ Visual Battle Arena (3.5) - Enhanced move display and character visualization
- ✅ Robot Character System (4.1) - Dynamic avatars with difficulty/strategy/personality
- ✅ Conversation System (4.2) - Real-time personality-based dialogue
- ✅ Enhanced UI/UX (4.3) - Professional battle interface with animations
- ✅ LSTM Integration (4.4) - Neural network strategy with performance tracking

### Future Enhancements (Phase 5)
- Advanced tournament brackets with elimination rounds
- Multiplayer real-time matches with character customization
- AI personality customization and training
- Mobile-responsive design optimization
- Cloud deployment with user accounts
- Voice synthesis for robot dialogue
- Advanced character animations and 3D models

## 📈 Performance Metrics

The system tracks comprehensive performance metrics:
- **Model Accuracy**: Real-time accuracy for all AI strategies including LSTM
- **Prediction Confidence**: Confidence levels for each prediction
- **Pattern Analysis**: Entropy and predictability calculations
- **Player Statistics**: Win rates, move distributions, strategy effectiveness
- **Tournament Stats**: ELO ratings, match history, leaderboard rankings
- **Character Interaction**: Dialogue engagement and personality preference tracking
- **Visual Analytics**: Battle arena usage patterns and user interface metrics

## 🎨 Visual Features

### Character System
- **Dynamic Avatars**: Robot characters that change based on difficulty, strategy, and personality
- **Visual Feedback**: Color-coded difficulty indicators and strategy symbols
- **Procedural Generation**: Unique robot names combining personality traits and difficulty levels

### Battle Arena
- **Move Visualization**: Clear display of human vs robot moves with game assets
- **Winner Highlighting**: Visual effects emphasizing round winners
- **Smooth Animations**: Timed sequences for dramatic gameplay experience

### Conversation Interface
- **Speech Bubbles**: Professional chat styling with personality-appropriate dialogue
- **Typing Effects**: Character-by-character text animation for realistic conversation
- **Mood System**: Visual indicators showing robot emotional state and reactions

## 🚀 Latest Updates (Current Release)

### New in this Version:
- **🤖 Robot Character Visualization**: Complete character system with visual avatars
- **💬 Personality Dialogue System**: Real-time conversation with 6 unique personalities
- **⚔️ Enhanced Battle Arena**: Professional move display with winner highlighting
- **🧠 LSTM Neural Network**: Advanced AI strategy using deep learning
- **🎨 Visual Polish**: Improved UI/UX with animations and professional styling
- **🔧 Code Consolidation**: Streamlined codebase with enhanced maintainability

## 🤝 Contributing

This is an educational project demonstrating advanced AI concepts in game development. Feel free to explore the code and experiment with different AI strategies!

## 📄 License

This project is open source and available for educational purposes.

---

**🎮 Ready to challenge the AI? Your move!**
