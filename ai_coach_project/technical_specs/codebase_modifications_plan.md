# Codebase Modification Plan for LLM Coaching Integration

## 🗂️ Files Requiring Modifications

### **Core System Files**

#### 1. `coach_tips.py` - MAJOR ENHANCEMENT
**Current**: Basic rule-based coaching  
**New**: Enhanced system with basic/AI mode toggle

```python
# Key additions needed:
class EnhancedCoachSystem:
    def __init__(self):
        self.basic_coach = CoachTipsGenerator()  # Keep existing
        self.ai_coach = None  # Will be loaded when AI mode enabled
        self.mode = 'basic'
        self.metrics_aggregator = GameMetricsAggregator()
    
    def set_coaching_mode(self, mode: str):
        if mode == 'ai' and self.ai_coach is None:
            self.ai_coach = LLMCoachingEngine()
        self.mode = mode
    
    def generate_coaching_advice(self, game_state, coaching_type='real_time'):
        # Route to appropriate coaching system
        pass

# Enhanced metrics extraction for LLM
class GameMetricsAggregator:
    def extract_comprehensive_metrics(self, game_state):
        # Combine ALL existing metrics into LLM-ready format
        pass
```

#### 2. `webapp/app.py` - MODERATE ENHANCEMENT
**Current**: Basic coaching endpoint `/coaching`  
**New**: Multiple LLM coaching endpoints

```python
# New routes to add:
@app.route('/coaching/mode', methods=['POST'])
def set_coaching_mode():
    # Toggle between basic/AI coaching
    pass

@app.route('/coaching/ai', methods=['GET']) 
def get_ai_coaching():
    # Real-time AI coaching advice
    pass

@app.route('/analysis/comprehensive', methods=['POST'])
def generate_comprehensive_analysis():
    # Post-game comprehensive analysis
    pass

@app.route('/coaching/context', methods=['GET'])
def get_coaching_context():
    # Provide all available metrics for LLM context
    pass

# Enhanced context building
def build_comprehensive_game_context():
    # Aggregate all metrics from existing systems:
    # - change_point_detector.py metrics
    # - stats_manager.py data  
    # - performance_optimizer.py insights
    # - replay_system.py analysis
    # - ml_model_enhanced.py predictions
    pass
```

#### 3. `webapp/templates/index.html` - MODERATE ENHANCEMENT
**Current**: Basic coach panel with tips and experiments  
**New**: Mode toggle + enhanced AI coaching display

```html
<!-- Modifications needed in existing coach panel around line 475-505 -->
<div class="panel coach-panel">
    <div class="panel-heading">
        <h2 class="panel-title">🎓 AI Coach</h2>
        <!-- NEW: Mode toggle -->
        <div class="coaching-mode-toggle">
            <label class="toggle-switch">
                <input type="checkbox" id="ai-coaching-toggle" onchange="toggleCoachingMode()">
                <span class="toggle-slider"></span>
            </label>
            <span id="coaching-mode-label">Basic Coaching</span>
        </div>
    </div>
    
    <!-- Existing content stays, enhanced with AI mode -->
    <div id="coaching-content-area">
        <!-- Keep existing coach-card for basic tips -->
        
        <!-- NEW: AI coaching panel (hidden by default) -->
        <div id="ai-coaching-panel" class="panel coach-card" style="display: none;">
            <div id="ai-insights-content" class="coach-content">
                <div id="behavioral-insights"></div>
                <div id="strategic-analysis"></div>
                <div id="adaptation-recommendations"></div>
            </div>
        </div>
    </div>
</div>

<!-- NEW: Post-game analysis section (show only after game ends) -->
<div class="endgame-analysis-section" id="endgame-analysis" style="display: none;">
    <button class="btn primary large" onclick="generateComprehensiveAnalysis()">
        🎯 Get Comprehensive Game Analysis
    </button>
    <div id="comprehensive-analysis-content"></div>
</div>
```

```javascript
// New JavaScript functions to add around line 2983 (near existing refreshCoachingTips)
function toggleCoachingMode() {
    const toggle = document.getElementById('ai-coaching-toggle');
    const label = document.getElementById('coaching-mode-label');
    const aiPanel = document.getElementById('ai-coaching-panel');
    
    const mode = toggle.checked ? 'ai' : 'basic';
    
    // Update backend
    fetch('/coaching/mode', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({mode: mode})
    });
    
    // Update UI
    label.textContent = mode === 'ai' ? 'AI Coaching' : 'Basic Coaching';
    aiPanel.style.display = toggle.checked ? 'block' : 'none';
    
    // Refresh coaching content
    refreshCoachingTips();
}

function generateComprehensiveAnalysis() {
    // Fetch comprehensive post-game analysis
    fetch('/analysis/comprehensive', {method: 'POST'})
        .then(response => response.json())
        .then(data => displayComprehensiveAnalysis(data));
}

function displayComprehensiveAnalysis(analysis) {
    // Render comprehensive analysis in modal or dedicated section
    pass
}
```

---

### **New Files to Create**

#### 4. `llm_coaching_engine.py` - NEW CORE FILE
**Purpose**: Main LLM coaching system integration

```python
"""
LLM-powered coaching engine for Rock-Paper-Scissors
Integrates distilled model with comprehensive game metrics
"""

import torch
import onnxruntime
from typing import Dict, List, Any, Optional
import json
import time
from collections import deque

class LLMCoachingEngine:
    """Main engine for LLM-powered coaching"""
    
    def __init__(self, model_path="models/rps_coaching_llm.onnx"):
        self.model = self.load_model(model_path)
        self.context_builder = GameContextBuilder()
        self.response_formatter = CoachingResponseFormatter()
        self.cache = IntelligentResponseCache()
        self.performance_monitor = InferencePerformanceMonitor()
    
    def generate_coaching_advice(self, game_state, coaching_type='real_time'):
        """Main entry point for generating coaching advice"""
        start_time = time.time()
        
        try:
            # Build comprehensive context
            context = self.context_builder.build_context(game_state, coaching_type)
            
            # Check cache first
            cached_response = self.cache.get_similar_context(context)
            if cached_response:
                return cached_response
            
            # Generate new advice using LLM
            raw_response = self.model.generate(context)
            
            # Format response appropriately
            formatted_response = self.response_formatter.format(
                raw_response, context, coaching_type
            )
            
            # Cache for future use
            self.cache.store(context, formatted_response)
            
            # Monitor performance
            inference_time = time.time() - start_time
            self.performance_monitor.log_inference(inference_time, len(context))
            
            return formatted_response
            
        except Exception as e:
            # Fallback to basic coaching on any error
            return self.generate_fallback_advice(game_state)
    
    def load_model(self, model_path):
        """Load optimized ONNX model"""
        session = onnxruntime.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']  # CPU optimized
        )
        return OptimizedModelWrapper(session)

class GameContextBuilder:
    """Builds comprehensive context from all game metrics"""
    
    def build_context(self, game_state, coaching_type):
        """Extract and format all available game metrics"""
        
        # Import existing analysis modules
        from change_point_detector import ChangePointDetector
        from stats_manager import StatsManager  # If exists
        from coach_tips import CoachTipsGenerator
        
        base_context = {
            # Core game state
            'current_round': game_state['round'],
            'human_moves': game_state['human_history'],
            'robot_moves': game_state['robot_history'],
            'results': game_state['result_history'],
            
            # Performance metrics
            'win_rates': self.calculate_win_rates(game_state['result_history']),
            'recent_performance': self.analyze_recent_performance(game_state),
            
            # Pattern analysis (leverage existing coach_tips.py)
            'pattern_analysis': self.extract_pattern_analysis(game_state),
            
            # Strategy adaptation (leverage change_point_detector.py)
            'adaptation_analysis': self.extract_adaptation_analysis(game_state),
            
            # AI behavior analysis
            'ai_analysis': self.analyze_ai_behavior(game_state),
            
            # Meta context
            'game_phase': self.determine_game_phase(game_state['round']),
            'coaching_context': coaching_type
        }
        
        # Add coaching-type specific context
        if coaching_type == 'real_time':
            base_context.update(self.build_realtime_context(game_state))
        elif coaching_type == 'endgame_analysis':
            base_context.update(self.build_endgame_context(game_state))
        
        return base_context

class CoachingResponseFormatter:
    """Formats LLM responses for different coaching modes"""
    
    def format(self, raw_response, context, coaching_type):
        """Format response based on coaching type"""
        
        if coaching_type == 'real_time':
            return self.format_realtime_advice(raw_response, context)
        elif coaching_type == 'endgame_analysis':
            return self.format_comprehensive_analysis(raw_response, context)
    
    def format_realtime_advice(self, response, context):
        """Format for real-time coaching display"""
        return {
            'immediate_tips': response.get('tactical_advice', []),
            'strategic_insights': response.get('strategic_analysis', ''),
            'behavioral_observations': response.get('behavioral_insights', ''),
            'adaptation_suggestions': response.get('adaptation_advice', ''),
            'confidence_level': response.get('confidence', 0.8),
            'educational_note': response.get('educational_insight', '')
        }
    
    def format_comprehensive_analysis(self, response, context):
        """Format for post-game comprehensive analysis"""
        return {
            'psychological_patterns': response.get('psychological_analysis', ''),
            'strategic_evolution': response.get('strategy_evolution', ''),
            'decision_making_insights': response.get('decision_analysis', ''),
            'learning_recommendations': response.get('improvement_suggestions', ''),
            'fascinating_discoveries': response.get('behavioral_discoveries', ''),
            'educational_summary': response.get('educational_summary', ''),
            'performance_assessment': response.get('performance_review', '')
        }

class IntelligentResponseCache:
    """Cache coaching responses for similar game contexts"""
    
    def __init__(self, max_size=1000):
        self.cache = deque(maxlen=max_size)
        self.similarity_threshold = 0.85
    
    def get_similar_context(self, context):
        """Find cached response for similar context"""
        for cached_item in self.cache:
            if self.calculate_similarity(context, cached_item['context']) > self.similarity_threshold:
                return cached_item['response']
        return None
    
    def store(self, context, response):
        """Store context-response pair"""
        self.cache.append({
            'context': context,
            'response': response,
            'timestamp': time.time()
        })
    
    def calculate_similarity(self, context1, context2):
        """Calculate similarity between contexts"""
        # Implement context similarity calculation
        # Consider: move patterns, win rates, game phase, etc.
        pass
```

#### 5. `model_training/training_pipeline.py` - NEW FILE
**Purpose**: Complete training pipeline for the coaching LLM

```python
"""
Training pipeline for RPS coaching LLM
Includes data generation, model training, and distillation
"""

class TrainingPipeline:
    """Complete pipeline for training coaching LLM"""
    
    def __init__(self):
        self.data_generator = TrainingDataGenerator()
        self.teacher_models = TeacherModelEnsemble()
        self.student_model = CompactCoachingTransformer()
        self.distiller = KnowledgeDistiller()
    
    def run_complete_training(self):
        """Execute full training pipeline"""
        
        # Phase 1: Generate training data
        print("🔄 Generating training data using teacher models...")
        training_data = self.data_generator.generate_comprehensive_dataset()
        
        # Phase 2: Knowledge distillation
        print("🧠 Training student model via knowledge distillation...")
        trained_model = self.distiller.distill_knowledge(
            self.teacher_models, 
            self.student_model, 
            training_data
        )
        
        # Phase 3: Optimization
        print("⚡ Optimizing model for inference...")
        optimized_model = self.optimize_for_production(trained_model)
        
        # Phase 4: Validation
        print("✅ Validating coaching quality...")
        validation_results = self.validate_coaching_quality(optimized_model)
        
        return optimized_model, validation_results
```

#### 6. `models/model_utils.py` - NEW FILE
**Purpose**: Model loading, optimization, and inference utilities

```python
"""
Utilities for model management and optimization
"""

class OptimizedModelWrapper:
    """Wrapper for optimized ONNX model with performance monitoring"""
    
    def __init__(self, onnx_session):
        self.session = onnx_session
        self.tokenizer = self.load_tokenizer()
        self.max_length = 512
    
    def generate(self, context):
        """Generate coaching advice from context"""
        # Tokenize input
        inputs = self.tokenizer.encode(context, max_length=self.max_length)
        
        # Run inference
        outputs = self.session.run(None, {'input_ids': inputs})
        
        # Decode output
        response = self.tokenizer.decode(outputs[0])
        
        return self.parse_coaching_response(response)
    
    def parse_coaching_response(self, raw_response):
        """Parse structured coaching response"""
        # Parse the model output into structured coaching advice
        pass
```

---

### **Files Requiring Minor Modifications**

#### 7. `requirements.txt` - ADD DEPENDENCIES
```
# Add LLM-related dependencies
torch>=1.9.0
onnxruntime>=1.12.0
transformers>=4.20.0
datasets>=2.0.0
tokenizers>=0.12.0
```

#### 8. `main.py` - MINOR ADDITION
```python
# Add import for enhanced coaching system
from coach_tips import EnhancedCoachSystem

# Initialize enhanced coach instead of basic coach
enhanced_coach = EnhancedCoachSystem()
```

#### 9. `game.py` - MINOR ENHANCEMENT
```python
# Add methods to track additional metrics for LLM coaching
class Game:
    def __init__(self):
        # ... existing code ...
        self.response_times = []  # Track timing for behavioral analysis
        self.move_timestamps = []  # For pressure analysis
    
    def track_move_timing(self, response_time):
        """Track response time for behavioral analysis"""
        self.response_times.append(response_time)
        self.move_timestamps.append(time.time())
```

---

## 🔄 Integration Dependencies

### **Module Interaction Map**

```
┌─────────────────────┐    ┌──────────────────────┐
│   webapp/app.py     │────│  llm_coaching_engine │
│   (API endpoints)   │    │  (LLM integration)   │
└─────────────────────┘    └──────────────────────┘
           │                          │
           ▼                          ▼
┌─────────────────────┐    ┌──────────────────────┐
│   coach_tips.py     │────│  Game Context        │
│   (Enhanced system) │    │  Builder             │
└─────────────────────┘    └──────────────────────┘
           │                          │
           ▼                          ▼
┌─────────────────────┐    ┌──────────────────────┐
│  Existing Analytics │    │  Model Inference     │
│  (All modules)      │    │  (ONNX Runtime)      │
└─────────────────────┘    └──────────────────────┘
```

### **Data Flow**

1. **Frontend** sends coaching request to `/coaching/ai`
2. **webapp/app.py** calls `enhanced_coach.generate_coaching_advice()`
3. **Enhanced coach** determines mode and routes appropriately
4. **LLM engine** builds context using existing analytics modules
5. **Model inference** generates coaching advice
6. **Response formatter** structures output for frontend
7. **Frontend** displays enhanced coaching advice

---

## 📋 Implementation Checklist

### **Phase 1: Core System**
- [ ] Create `llm_coaching_engine.py` with main LLM integration
- [ ] Enhance `coach_tips.py` with mode toggle functionality  
- [ ] Create `GameContextBuilder` to aggregate all metrics
- [ ] Implement response caching system

### **Phase 2: Model Integration**
- [ ] Create training pipeline in `model_training/`
- [ ] Implement knowledge distillation process
- [ ] Create model optimization utilities
- [ ] Generate and validate training data

### **Phase 3: Backend API**
- [ ] Add new coaching endpoints to `webapp/app.py`
- [ ] Implement comprehensive context building
- [ ] Add error handling and fallback systems
- [ ] Create performance monitoring

### **Phase 4: Frontend Enhancement**
- [ ] Add coaching mode toggle to `index.html`
- [ ] Implement AI coaching panel display
- [ ] Add post-game analysis section
- [ ] Create JavaScript for mode switching

### **Phase 5: Testing & Optimization**
- [ ] Performance testing (<50ms target)
- [ ] Model size optimization (<100MB target)
- [ ] User experience testing
- [ ] Integration testing with all existing features

---

## ⚠️ Risk Mitigation

### **Fallback Systems**
- Always fallback to basic coaching if LLM fails
- Cache common responses to ensure availability  
- Progressive loading - basic features work without LLM

### **Performance Safeguards**
- Timeout limits on LLM inference (max 100ms)
- Model size monitoring
- Memory usage tracking
- Graceful degradation if resources limited

### **Integration Safety**
- All existing functionality remains unchanged
- LLM features are additive only
- Basic mode remains fully functional
- No breaking changes to existing APIs

*All modifications designed to be backward-compatible and non-disruptive to existing functionality*