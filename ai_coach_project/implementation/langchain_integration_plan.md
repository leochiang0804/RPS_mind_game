# Streamlined LangChain Integration Plan

## 🔗 LangChain Integration Benefits

### **Why LangChain for RPS Coaching:**
1. **Prompt Engineering**: Built-in template management for coaching prompts
2. **Chain Orchestration**: Streamlined multi-step analysis pipelines
3. **Memory Management**: Conversation memory for session context
4. **Tool Integration**: Easy integration with existing analytics modules
5. **Caching**: Built-in caching mechanisms for responses

## 🏗️ Simplified Architecture with LangChain

### **Core Components:**

```python
from langchain.llms import LlamaCpp
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

class LangChainCoachingEngine:
    """Simplified coaching engine using LangChain"""
    
    def __init__(self):
        # Use local oLLM with LangChain
        self.llm = LlamaCpp(
            model_path="models/coaching_llm.gguf",  # Quantized model
            temperature=0.7,
            max_tokens=512,
            n_ctx=2048
        )
        
        # Coaching prompt templates
        self.real_time_template = PromptTemplate(
            input_variables=["game_context", "current_metrics"],
            template="""
            You are an expert Rock-Paper-Scissors coach. Analyze the current game state and provide immediate coaching advice.
            
            Game Context: {game_context}
            Current Metrics: {current_metrics}
            
            Provide:
            1. Immediate tactical advice (1-2 moves)
            2. Pattern observation
            3. Strategic insight
            
            Response:
            """
        )
        
        # Memory for session context
        self.memory = ConversationBufferWindowMemory(k=10)
        
        # Coaching chain
        self.coaching_chain = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            prompt=self.real_time_template
        )
        
        # Analytics tools
        self.analytics_tools = [
            Tool(
                name="PatternAnalyzer",
                description="Analyze move patterns and predictability",
                func=self.analyze_patterns
            ),
            Tool(
                name="PerformanceTracker", 
                description="Track performance trends",
                func=self.track_performance
            )
        ]
    
    def get_real_time_coaching(self, game_state):
        """Get immediate coaching advice"""
        context = self.build_game_context(game_state)
        metrics = self.extract_metrics(game_state)
        
        response = self.coaching_chain.predict(
            game_context=context,
            current_metrics=metrics
        )
        
        return self.parse_coaching_response(response)

class LangGraphAnalysisPipeline:
    """Use LangGraph for complex post-game analysis workflow"""
    
    def __init__(self):
        self.graph = self.create_analysis_graph()
    
    def create_analysis_graph(self):
        """Create analysis workflow graph"""
        from langgraph.graph import StateGraph, END
        
        workflow = StateGraph(dict)
        
        # Add analysis nodes
        workflow.add_node("session_analysis", self.analyze_session)
        workflow.add_node("behavioral_analysis", self.analyze_behavior) 
        workflow.add_node("educational_recommendations", self.generate_recommendations)
        workflow.add_node("insight_generation", self.generate_insights)
        
        # Define workflow
        workflow.set_entry_point("session_analysis")
        workflow.add_edge("session_analysis", "behavioral_analysis")
        workflow.add_edge("behavioral_analysis", "educational_recommendations")
        workflow.add_edge("educational_recommendations", "insight_generation")
        workflow.add_edge("insight_generation", END)
        
        return workflow.compile()
    
    def run_comprehensive_analysis(self, complete_session):
        """Run complete analysis pipeline"""
        initial_state = {"session_data": complete_session}
        result = self.graph.invoke(initial_state)
        return result
```

## 📦 Simplified Dependencies

### **Minimal Additional Dependencies:**
```python
# Add to requirements.txt - much simpler!
LANGCHAIN_DEPENDENCIES = [
    "langchain>=0.1.0",
    "langchain-community>=0.0.20", 
    "llama-cpp-python>=0.2.0",  # For local oLLM
    "faiss-cpu>=1.7.0",        # For vector storage (optional)
]

# Remove complex dependencies:
# - torch (not needed with llama-cpp-python)
# - onnxruntime (llama-cpp handles optimization)
# - transformers (langchain abstracts this)
```

## 🚀 Streamlined Implementation

### **Phase 1: Quick MVP (Week 1)**
```python
# Minimal viable implementation
class MinimalLLMCoach:
    def __init__(self):
        # Use existing coach as fallback
        self.basic_coach = CoachTipsGenerator()
        
        # Simple LangChain integration
        try:
            from langchain.llms import LlamaCpp
            self.llm = LlamaCpp(model_path="models/tiny_coaching_model.gguf")
            self.llm_available = True
        except:
            self.llm_available = False
    
    def get_coaching_advice(self, game_state, mode='basic'):
        if mode == 'ai' and self.llm_available:
            return self.get_llm_advice(game_state)
        else:
            return self.basic_coach.generate_tips(
                game_state['human_history'],
                game_state['robot_history'],
                game_state['result_history'], 
                game_state.get('change_points', [])
            )
    
    def get_llm_advice(self, game_state):
        # Simple prompt with existing metrics
        context = f"""
        Game Round: {game_state['round']}
        Recent Moves: {game_state['human_history'][-10:]}
        Win Rate: {game_state['stats']['human_win']/game_state['round']*100:.1f}%
        
        Provide brief coaching advice:
        """
        
        try:
            response = self.llm(context)
            return {'tips': [response], 'experiments': [], 'insights': {}}
        except:
            # Fallback to basic coaching
            return self.basic_coach.generate_tips(
                game_state['human_history'],
                game_state['robot_history'], 
                game_state['result_history'],
                game_state.get('change_points', [])
            )
```

*This approach dramatically reduces complexity while providing immediate LLM capabilities*