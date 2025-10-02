"""
Coaching Tips Glossary System
=============================

A comprehensive glossary of 100+ coaching tips that are dynamically selected
based on game progress, player patterns, and performance metrics.

All three coaching modes (basic, lightweight LLM, advanced LLM) use this
unified tip selection system to ensure consistent, contextual advice.
"""

import random
from typing import Dict, List, Tuple, Any

# CORE TIP GLOSSARY - Organized by categories
GLOSSARY_TIPS = {
    # === EARLY GAME TIPS (Rounds 1-10) ===
    'early_game': [
        "Start with Rock - it's the most common opening move",
        "Watch for their favorite opening move in the first 3 rounds",
        "Try a simple pattern like Rock-Paper-Scissors to test their reaction", 
        "Don't overthink early rounds - focus on learning their style",
        "Most players start with Rock or Paper - avoid Scissors early",
        "Pay attention to whether they mirror your moves initially",
        "Test if they prefer aggressive (Rock) or defensive (Paper) openings",
        "Early rounds set the tone - establish unpredictability",
        "Look for immediate patterns - some players repeat their first move",
        "Stay flexible - early game is about gathering intelligence"
    ],
    
    # === PATTERN RECOGNITION TIPS ===
    'pattern_detection': [
        "They're showing a 3-move cycle - predict the next in sequence",
        "Classic alternating pattern detected - break it with a counter-cycle",
        "They favor Rock after losing - prepare Paper when they're down",
        "Strong Paper bias when ahead - use Scissors to cut their confidence", 
        "They're mirroring your moves with a 1-round delay - stay unpredictable",
        "Frequency pattern: they're overusing one move - counter it heavily",
        "They react to your wins by switching - anticipate the change",
        "Meta-pattern detected: they're trying to break their own patterns",
        "They show revenge patterns - beating their winning move next round",
        "Conditional pattern: their move depends on the previous result"
    ],
    
    # === PSYCHOLOGICAL WARFARE ===
    'psychology': [
        "Rock = aggression, Paper = defense, Scissors = precision - read their mood",
        "Winning streaks make players overconfident - exploit with unexpected moves",
        "Losing streaks make players desperate - they'll try risky patterns",
        "Mirror their energy: aggressive players respond to counter-aggression",
        "Patient players are vulnerable to sudden strategy shifts",
        "Frustrated players abandon strategy - capitalize on their chaos",
        "Confident players become predictable - use their confidence against them",
        "When they're thinking too hard, they often default to their favorite move",
        "Stress patterns: under pressure, players revert to simple sequences",
        "They're trying to psychology you - don't fall for reverse psychology"
    ],
    
    # === MID-GAME STRATEGY (Rounds 11-25) ===
    'mid_game': [
        "Establish a false pattern, then break it dramatically",
        "They've adapted to your early strategy - time for phase 2",
        "Mix randomness with subtle patterns - keep them guessing",
        "Counter their strongest pattern while building your own trap",
        "They're getting comfortable - disrupt their rhythm now",
        "Mid-game is about controlled chaos - patterns within randomness",
        "Start your main strategy now that you know their tendencies",
        "They think they've figured you out - prove them wrong",
        "Layer deception: make them think they've found your pattern",
        "Mid-game plateau detected - escalate your strategy complexity"
    ],
    
    # === ENDGAME MASTERY (Rounds 26+) ===
    'endgame': [
        "Final rounds: they're either desperate or overconfident - exploit it",
        "Everything counts now - stick to your highest-probability plays",
        "They know your patterns - use their knowledge against them", 
        "Endgame psychology: they'll either crack under pressure or find zen",
        "Deploy your best counter-strategy for their dominant pattern",
        "If ahead: play conservatively. If behind: calculated risks only",
        "They're making their final stand - predict their desperation moves",
        "Endgame mastery: make every move count with surgical precision",
        "Close games create max pressure - stay calm, think one move ahead",
        "Victory is within reach - don't let them steal it with a comeback"
    ],
    
    # === ANTI-STRATEGY TIPS ===
    'counter_strategy': [
        "They're using the 'anti-Rock' strategy - flood them with Paper",
        "Frequency analysis shows their weakness - hammer their least-used counter",
        "They're playing too randomly - build a statistical edge",
        "They're over-strategizing - simple patterns will confuse them",
        "They've found your weakness - immediately switch tactics",
        "Mirror-match detected - whoever breaks first loses",
        "They're using psychological pressure - ignore it and play smart",
        "Complex pattern user - disrupt with strategic randomness",
        "They're baiting you into their trap - reverse the trap",
        "Advanced player detected - match their sophistication level"
    ],
    
    # === MOMENTUM AND STREAKS ===
    'momentum': [
        "You're on a hot streak - don't get cocky, maintain focus",
        "Cold streak detected - break the pattern with a bold move",
        "They're building momentum - stop it now before it's unstoppable",
        "Momentum shift incoming - be ready to capitalize",
        "You're in their head - press your psychological advantage",
        "They're in your head - clear your mind and trust your instincts",
        "Streak psychology: they expect you to break it - double down instead",
        "Close game = high tension - use it to force their mistakes",
        "They're trying to ride their hot streak - cool them down",
        "Momentum is mental - control the psychological battlefield"
    ],
    
    # === ADAPTIVE PLAY ===
    'adaptation': [
        "They've adapted to you - counter-adapt immediately",
        "Your strategy is stale - inject fresh unpredictability",
        "They're learning too fast - accelerate your strategy evolution",
        "Perfect time to pivot - they're locked into countering your old strategy",
        "They think they've solved you - prove them wrong with style",
        "Meta-game time: play the player, not just the game",
        "They're stuck in their pattern - exploit their rigidity",
        "Adaptation war detected - whoever adapts faster wins",
        "They're trying to out-think you - sometimes simple works best",
        "Evolution phase: your game must grow stronger"
    ],
    
    # === EMERGENCY TACTICS ===
    'emergency': [
        "Behind by 3+ points - time for high-risk, high-reward plays",
        "They're about to win - deploy your most desperate strategy",
        "Critical moment - one mistake costs everything",
        "Panic time - stay calm, trust your best pattern",
        "They smell victory - make them overconfident and careless",
        "Everything's on the line - play your absolute best move",
        "Miracle needed - find their one exploitable weakness",
        "Last stand - make every move count like it's the final one",
        "They're coasting to victory - disrupt their cruise control",
        "Down to the wire - surgical precision required"
    ],
    
    # === ADVANCED TECHNIQUES ===
    'advanced': [
        "Conditional probability favors your next counter-move - execute it",
        "Bayesian analysis suggests they'll choose their comfort pick",
        "Game theory optimal: mix your three most successful patterns",
        "Information asymmetry: you know their tells, they don't know yours",
        "Nash equilibrium disrupted - exploit their sub-optimal play",
        "Pattern complexity escalation - layer multiple micro-patterns",
        "Psychological game theory: make them think you think they think...",
        "Advanced pattern: 2nd-order thinking about their counter-strategy",
        "Meta-cognitive warfare: make them overthink your thinking",
        "Expert-level play: predictable unpredictability"
    ]
}

# INSIGHTS GLOSSARY - Rich contextual analysis
INSIGHTS_GLOSSARY = {
    'pattern_strength': [
        "Strong pattern detected - high confidence prediction",
        "Weak pattern signal - proceed with caution", 
        "No clear pattern - focus on psychological reads",
        "Pattern emerging - a few more rounds to confirm",
        "Pattern breaking down - they're adapting"
    ],
    
    'player_psychology': [
        "Aggressive player profile - favors Rock in pressure",
        "Defensive player profile - Paper is their safety net",
        "Analytical player - they're trying to solve you mathematically", 
        "Intuitive player - they go with gut feelings",
        "Adaptive player - constantly evolving their strategy"
    ],
    
    'momentum_analysis': [
        "Momentum strongly in your favor",
        "Momentum shifting toward opponent",
        "Momentum is neutral - next few moves are critical",
        "False momentum - they're setting a trap",
        "Psychological momentum vs actual advantage mismatch"
    ],
    
    'strategic_advice': [
        "Maintain current strategy - it's working",
        "Strategy adjustment needed - they've adapted",
        "Emergency pivot required - current approach failing",
        "Complex strategy vs simple approach decision point",
        "Trust your instincts vs analytical approach dilemma"
    ]
}

def select_tips_from_glossary(comprehensive_metrics: Dict[str, Any], llm_type: str) -> Tuple[List[str], Dict[str, str]]:
    """
    Select 3-5 contextual tips from the glossary based on comprehensive game metrics.
    
    Args:
        comprehensive_metrics: Complete game state and analysis
        llm_type: Type of coaching mode ('mock', 'trained', 'real')
        
    Returns:
        Tuple of (selected_tips, contextual_insights)
    """
    
    # Extract key metrics with safe defaults
    core_game = comprehensive_metrics.get('core_game', {})
    round_num = core_game.get('current_round', 0)
    player_score = core_game.get('human_score', 0)
    ai_score = core_game.get('ai_score', 0)
    
    pattern_analysis = comprehensive_metrics.get('pattern_analysis', {})
    
    # === TIP SELECTION LOGIC ===
    selected_tips = []
    category_priorities = []
    
    # Round-based category selection
    if round_num <= 10:
        category_priorities.append('early_game')
    elif round_num <= 25:
        category_priorities.append('mid_game')
    else:
        category_priorities.append('endgame')
    
    # Pattern-based categories
    if pattern_analysis.get('dominant_pattern'):
        category_priorities.append('pattern_detection')
        
    # Momentum-based categories  
    score_diff = player_score - ai_score
    if abs(score_diff) >= 3:
        category_priorities.append('emergency')
    elif abs(score_diff) >= 2:
        category_priorities.append('momentum')
        
    # Strategy complexity based on LLM type
    if llm_type == 'real':
        category_priorities.append('advanced')
    elif llm_type == 'trained':
        category_priorities.append('counter_strategy')
    
    # Always include psychology and adaptation  
    category_priorities.extend(['psychology', 'adaptation'])
    
    # Select 1-2 tips from each priority category
    for category in category_priorities[:3]:  # Top 3 categories
        if category in GLOSSARY_TIPS:
            available_tips = GLOSSARY_TIPS[category]
            num_tips = 2 if category == category_priorities[0] else 1  # More from top category
            selected_tips.extend(random.sample(available_tips, min(num_tips, len(available_tips))))
            
        if len(selected_tips) >= 5:
            break
    
    # Ensure we have 3-5 tips
    if len(selected_tips) < 3:
        # Fill with general tips
        general_categories = ['psychology', 'adaptation', 'pattern_detection']
        for category in general_categories:
            if len(selected_tips) >= 3:
                break
            available_tips = GLOSSARY_TIPS[category]
            selected_tips.extend(random.sample(available_tips, min(1, len(available_tips))))
    
    # Trim to 5 max
    selected_tips = selected_tips[:5]
    
    # === INSIGHT SELECTION ===
    insights = {}
    
    # Pattern strength insight
    pattern_confidence = pattern_analysis.get('confidence', 0.0)
    if pattern_confidence > 0.7:
        insights['pattern'] = random.choice(INSIGHTS_GLOSSARY['pattern_strength'][:2])  # Strong/emerging
    elif pattern_confidence > 0.3:
        insights['pattern'] = random.choice(INSIGHTS_GLOSSARY['pattern_strength'][2:4])  # Weak/emerging  
    else:
        insights['pattern'] = INSIGHTS_GLOSSARY['pattern_strength'][2]  # No clear pattern
    
    # Player psychology insight
    ai_behavior = comprehensive_metrics.get('ai_behavior', {})
    ai_strategy = ai_behavior.get('ai_strategy', 'unknown')
    
    if 'aggressive' in ai_strategy.lower():
        insights['psychology'] = INSIGHTS_GLOSSARY['player_psychology'][0]
    elif 'defensive' in ai_strategy.lower():
        insights['psychology'] = INSIGHTS_GLOSSARY['player_psychology'][1]
    else:
        insights['psychology'] = random.choice(INSIGHTS_GLOSSARY['player_psychology'][2:])
    
    # Momentum insight
    if score_diff > 2:
        insights['momentum'] = INSIGHTS_GLOSSARY['momentum_analysis'][0]  # Your favor
    elif score_diff < -2:
        insights['momentum'] = INSIGHTS_GLOSSARY['momentum_analysis'][1]  # Opponent favor
    else:
        insights['momentum'] = INSIGHTS_GLOSSARY['momentum_analysis'][2]  # Neutral
    
    # Strategic advice insight
    recent_performance = comprehensive_metrics.get('performance_metrics', {})
    win_rate = recent_performance.get('win_rate', 0.5)
    
    if win_rate > 0.6:
        insights['strategy'] = INSIGHTS_GLOSSARY['strategic_advice'][0]  # Maintain
    elif win_rate < 0.4:
        insights['strategy'] = INSIGHTS_GLOSSARY['strategic_advice'][2]  # Emergency pivot
    else:
        insights['strategy'] = INSIGHTS_GLOSSARY['strategic_advice'][1]  # Adjustment needed
    
    return selected_tips, insights

def generate_experiments_from_data(comprehensive_metrics: Dict[str, Any]) -> List[str]:
    """
    Generate strategic experiments based on comprehensive game metrics.
    
    Args:
        comprehensive_metrics: Complete game state and analysis
        
    Returns:
        List of experiment suggestions
    """
    
    experiments = []
    
    # Extract metrics safely
    core_game = comprehensive_metrics.get('core_game', {})
    pattern_analysis = comprehensive_metrics.get('pattern_analysis', {})
    round_num = core_game.get('current_round', 0)
    
    # Round-based experiments
    if round_num <= 10:
        experiments.extend([
            "Try the 'Echo Test' - repeat their last move to see their reaction",
            "Test their 'Rock Bias' - play Paper 3 times in a row",
            "Pattern Probe - play Rock-Paper-Scissors and see if they mirror"
        ])
    elif round_num <= 25:
        experiments.extend([
            "False Pattern Setup - establish Rock-Rock-Paper, then break it",
            "Counter-Frequency Attack - target their least-used move's counter",
            "Psychological Pressure Test - play their favorite move to confuse them"
        ])
    else:
        experiments.extend([
            "Endgame Blitz - rapid strategy change to throw them off",
            "Statistical Exploit - hammer their weakest defensive pattern",
            "Pressure Point Test - play opposite of their stress pattern"
        ])
    
    # Pattern-based experiments
    dominant_pattern = pattern_analysis.get('dominant_pattern')
    if dominant_pattern:
        experiments.append(f"Pattern Breaker - counter their {dominant_pattern} with precision timing")
    
    # Always include a psychological experiment
    psych_experiments = [
        "Mind Game - play randomly for 3 rounds to reset their pattern detection",
        "Confidence Shaker - target their winning move with its counter",
        "Prediction Test - play what you think they expect you NOT to play"
    ]
    experiments.append(random.choice(psych_experiments))
    
    # Return 3-4 experiments max
    return experiments[:4]


# UTILITY FUNCTIONS FOR ENHANCED TIP QUALITY

def format_tips_for_llm_type(tips: List[str], llm_type: str) -> List[str]:
    """Format tips appropriately for different LLM types"""
    
    if llm_type == 'mock':
        # Basic formatting - clear and direct
        return [f"💡 {tip}" for tip in tips]
    elif llm_type == 'trained':
        # Enhanced formatting with strategy emphasis
        return [f"🎯 Strategy: {tip}" for tip in tips]
    else:  # real LLM
        # Advanced formatting with deeper context
        return [f"🧠 Advanced: {tip}" for tip in tips]

def get_tip_confidence_score(comprehensive_metrics: Dict[str, Any]) -> float:
    """Calculate confidence score for tip quality"""
    
    # Base confidence on data completeness and pattern strength
    pattern_analysis = comprehensive_metrics.get('pattern_analysis', {})
    core_game = comprehensive_metrics.get('core_game', {})
    
    # More rounds = higher confidence
    rounds = core_game.get('current_round', 0)
    round_confidence = min(rounds / 20.0, 1.0)  # Max confidence at 20 rounds
    
    # Pattern strength adds confidence
    pattern_confidence = pattern_analysis.get('confidence', 0.0)
    
    # Data completeness check
    data_completeness = 0.0
    required_fields = ['core_game', 'pattern_analysis', 'ai_behavior']
    available_fields = sum(1 for field in required_fields if field in comprehensive_metrics)
    data_completeness = available_fields / len(required_fields)
    
    # Combined confidence score
    overall_confidence = (round_confidence * 0.4 + pattern_confidence * 0.4 + data_completeness * 0.2)
    
    return min(max(overall_confidence, 0.1), 1.0)  # Clamp between 0.1 and 1.0