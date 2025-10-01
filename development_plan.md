# Development Plan

## High-Level Overview
The goals are:

1. Develop a **Coach Feature**:
   - Use a lightweight, task-specific LLM (~150M–300M parameters) optimized for real-time coaching during gameplay.
   - Ensure the model is downsized (knowledge distillation, quantization) to fit within the resource constraints of the application.

2. Develop an **Advanced Play Mode with Wild Function Cards**:
   - Add dynamic, disruptive mechanics to enhance unpredictability and player engagement.
   - Ensure seamless integration with the existing AI strategies and game logic.

---

## 2. Coach Feature Development

### Phase 1: Planning
- Define the coaching system’s scope:
  - Real-time actionable insights (e.g., tips on player moves, AI strategy awareness).
  - Seamless integration with **Advanced Analytics Dashboard** and existing ML pipelines.
- Finalize the LLM:
  - Select a lightweight, fine-tuned model (e.g., **DistilGPT-2**, **TinyBERT**, or GPT-Neo).
  - Ensure the model is downsized:
    - **Knowledge Distillation**: Use the large teacher model to train a smaller student model (150M–300M parameters).
    - **Quantization**: Convert the student model to INT8 to reduce runtime memory usage (~100MB–200MB RAM).

### Phase 2: Backend Development
1. **LLM Integration**:
   - Train the distilled student model for gameplay-specific tasks:
     - Use a curated dataset of game logs, strategy patterns, and coaching examples.
     - Fine-tune the model using **oLLM** or frameworks like Hugging Face Transformers.
   - Quantize the model to **INT8** using ONNX Runtime or TensorRT for optimal performance.
   - Example integration:
     ```python
     from ollm import OpenLLM
     model = OpenLLM.load("distil-gpt2", quantized=True)

     game_state = {"player_moves": ["rock", "scissors"], "ai_strategy": "Markov"}
     coaching_tip = model.generate(f"Suggest coaching advice for this game state: {game_state}")
     ```

2. **Real-Time Coaching Pipeline**:
   - Extend the **`coach_tips.py`** module:
     - Analyze game state (e.g., player patterns, AI predictions, strategy shifts).
     - Generate coaching insights using:
       - Predefined templates for common scenarios (e.g., repetitive moves).
       - LLM outputs for nuanced, context-aware advice.

3. **Caching and Optimization**:
   - Cache frequent coaching outputs (e.g., common scenarios) to minimize LLM calls.
   - Use batch inference for efficiency during high gameplay activity.

### Phase 3: Frontend Integration
1. **Coach Panel UI**:
   - Add a panel to the **Advanced Analytics Dashboard** to display dynamic coaching tips.
   - Visualize metrics like:
     - **Move Frequency Chart**: Highlight player tendencies.
     - **Entropy Analysis**: Indicate predictability of player moves.
   - Use visual alerts for high-priority tips (e.g., "AI strategy change detected!").

2. **Optional Enhancements**:
   - Add **text-to-speech (TTS)** for audible coaching tips.
   - Allow players to toggle coaching depth (basic vs. advanced tips).

### Phase 4: Testing and Optimization
- Validate the accuracy and relevance of coaching tips in real gameplay scenarios.
- Optimize the pipeline for low-latency performance:
  - Ensure inference time <100ms for smooth gameplay.

### Phase 5: Deployment
- Deploy the Coach Feature:
  - Host the LLM on lightweight cloud servers (e.g., AWS EC2 T2) or integrate locally for on-device inference.
  - Monitor feature performance and gather user feedback.

---

## 3. Advanced Play Mode with Wild Function Cards

### Phase 1: Design and Mechanics
- Define the Wild Function Cards:
  - **Card Types**:
    1. **Probability Disruptor**: Randomizes AI predictions for one round.
    2. **Strategy Shuffler**: Forces the AI to switch strategies mid-game.
    3. **Blind Spot Card**: Makes the AI ignore certain player moves temporarily.
  - **Impact Levels**:
    - **Low Impact**: Minor disruptions (e.g., randomizing predictions slightly).
    - **Medium Impact**: Moderate disruptions (e.g., switching to a suboptimal strategy).
    - **High Impact**: Major disruptions (e.g., forcing fully random predictions).

### Phase 2: Backend Implementation
1. **Card Logic**:
   - Extend the **`strategy.py`** module to implement Wild Card effects:
     - **Example**: Probability Disruptor
       ```python
       if active_card == "probability_disruptor":
           ai_prediction = random.choice(["rock", "paper", "scissors"])
       ```
   - Add cooldown mechanics to limit card usage per game.

2. **AI Adaptation**:
   - Allow the AI to adapt to Wild Cards over time:
     - Gradually reduce the impact of cards as the game progresses.
     - Introduce "resistance" mechanics where the AI learns from disruptions.

### Phase 3: Frontend Integration
1. **UI for Wild Function Cards**:
   - Add a card selection panel to the gameplay interface.
   - Display available cards with descriptions and activation buttons.
2. **Visual Effects**:
   - Add animations for card activations (e.g., screen shake, glowing AI predictions).

### Phase 4: Testing and Balancing
- Test the Wild Card mechanics with players:
  - Ensure cards add fun and unpredictability without breaking game balance.
  - Adjust cooldowns, usage limits, and impact levels based on feedback.

### Phase 5: Deployment
- Release the Advanced Play Mode as an optional feature:
  - Allow players to enable/disable Wild Cards in the game settings.
  - Monitor usage and feedback for future enhancements.

---

## 4. Integration with Existing Features
- **Advanced Analytics**:
  - Coach Feature will leverage existing analytics (e.g., entropy calculations, predictability scoring) for deeper insights.
- **AI Strategies**:
  - The Wild Function Cards will interact dynamically with the existing AI strategies (Random, Markov, LSTM, etc.).
- **UI/UX**:
  - Both features will integrate seamlessly into the web app, enhancing the overall experience.

---

## 5. Estimated Resource Requirements
| **Component**              | **Storage (Disk)** | **Runtime Memory (RAM)** |
|----------------------------|--------------------|--------------------------|
| Lightweight LLM            | 50MB–200MB        | 100MB–400MB             |
| Predefined Templates       | ~5MB              | ~10MB                   |
| Coach Panel Logic          | ~10MB             | ~50MB                   |
| Wild Card Logic            | ~20MB             | ~50MB–100MB             |
| Frontend Assets            | ~30MB             | Negligible              |
| Backend Overhead           | Already Included  | ~100MB                  |
| **Total**                  | **115MB–265MB**   | **310MB–660MB**         |

---

## 6. Timeline
| **Phase**                   | **Task**                                  | **Timeline** |
|-----------------------------|------------------------------------------|--------------|
| **Coach Feature**           |                                          |              |
| Phase 1                   | Research and Planning                    | 1 Week       |
| Phase 2                   | Backend Development                      | 2 Weeks      |
| Phase 3                   | Frontend Integration                     | 2 Weeks      |
| Phase 4                   | Testing and Optimization                 | 1 Week       |
| Phase 5                   | Deployment                               | 1 Week       |
| **Wild Function Cards**     |                                          |              |
| Phase 1                   | Game Design                              | 1 Week       |
| Phase 2                   | Backend Implementation                   | 2 Weeks      |
| Phase 3                   | Frontend Integration                     | 1 Week       |
| Phase 4                   | Testing and Balancing                    | 1 Week       |
| Phase 5                   | Deployment                               | 1 Week       |

**Total Time**: ~10 Weeks (~2.5 Months)

---

## 7. Conclusion
This plan ensures the Coach Feature and Wild Function Cards are implemented efficiently, leveraging downsized LLMs and optimized logic. The features remain lightweight, scalable, and fun, while aligning with the existing application architecture.