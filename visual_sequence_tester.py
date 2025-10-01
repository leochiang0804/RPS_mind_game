#!/usr/bin/env python3
"""
Robot Distinctiveness Analysis Script

This script uses optimal winning sequences to test all AI combinations and analyze
whether different difficulty/strategy/personality combinations create truly distinctive
robot behaviors with measurably different performance characteristics.
"""

import json
import time
from typing import List, Dict, Any
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class RobotDistinctivenessAnalyzer:
    """Analyzes robot distinctiveness using optimal sequences."""
    
    def __init__(self):
        self.load_optimal_sequences()
        self.robot_combinations = self._generate_robot_combinations()
        self.analysis_results = {}
    
    def load_optimal_sequences(self):
        """Load the optimal sequences from the JSON file."""
        try:
            with open('best_sequences_quick_ref.json', 'r') as f:
                self.sequences = json.load(f)
            print("✅ Loaded optimal sequences successfully")
        except FileNotFoundError:
            print("❌ Optimal sequences not found. Please run optimal_move_sequences.py first.")
            sys.exit(1)
    
    def _generate_robot_combinations(self) -> List[Dict[str, str]]:
        """Generate all possible robot character combinations."""
        difficulties = ['random', 'frequency', 'markov', 'enhanced', 'lstm']
        strategies = ['balanced', 'to_win', 'not_to_lose']
        personalities = ['neutral', 'berserker', 'guardian', 'chameleon', 'professor', 'wildcard', 'mirror']
        
        combinations = []
        for difficulty in difficulties:
            for strategy in strategies:
                for personality in personalities:
                    combinations.append({
                        'difficulty': difficulty,
                        'strategy': strategy,
                        'personality': personality,
                        'name': f"{difficulty.title()} {strategy.replace('_', ' ').title()} {personality.title()}"
                    })
        
        return combinations
    
    def generate_javascript_test_function(self) -> str:
        """Generate JavaScript code for automated testing in the web interface."""
        
        js_code = '''
// Automated Visual Test for Optimal Move Sequences
// Copy and paste this into the browser console while on the game page

class OptimalSequenceTester {
    constructor() {
        this.sequences = ''' + json.dumps(self.sequences, indent=8) + ''';
        this.robotCombinations = ''' + json.dumps(self.robot_combinations, indent=8) + ''';
        this.currentTest = 0;
        this.results = [];
        this.testDelay = 500; // ms between moves
        this.comboDelay = 2000; // ms between robot combinations
    }
    
    async startTest(gameLength = 25) {
        console.log(`🎮 Starting automated test with ${gameLength}-move sequences`);
        
        const sequenceKey = `${gameLength}_moves`;
        if (!this.sequences[sequenceKey]) {
            console.error(`No sequence found for ${gameLength} moves`);
            return;
        }
        
        const optimalSequence = this.sequences[sequenceKey];
        console.log(`📊 Testing sequence: ${optimalSequence.name}`);
        console.log(`🎯 Expected average win rate: ${optimalSequence.avg_win_rate.toFixed(1)}%`);
        
        // Test against all robot combinations
        for (let i = 0; i < this.robotCombinations.length; i++) {
            const combo = this.robotCombinations[i];
            console.log(`\\n🤖 Testing against: ${combo.name} (${i + 1}/${this.robotCombinations.length})`);
            
            const result = await this.testSequenceAgainstCombo(optimalSequence.sequence, combo);
            this.results.push({
                combo: combo,
                result: result,
                sequence_name: optimalSequence.name
            });
            
            // Add delay between combinations
            if (i < this.robotCombinations.length - 1) {
                await this.delay(this.comboDelay);
            }
        }
        
        this.displayFinalResults(gameLength);
    }
    
    async testSequenceAgainstCombo(sequence, combo) {
        // Set robot configuration
        this.setRobotConfiguration(combo);
        
        // Reset game first
        await this.resetGame();
        
        // Track results
        let wins = {human: 0, robot: 0, tie: 0};
        
        // Play the sequence
        for (let i = 0; i < sequence.length; i++) {
            const move = sequence[i];
            console.log(`  Move ${i + 1}/${sequence.length}: Playing ${move}`);
            
            const result = await this.playMove(move);
            if (result) {
                wins[result]++;
            }
            
            // Small delay between moves for visual effect
            await this.delay(this.testDelay);
        }
        
        const totalMoves = wins.human + wins.robot + wins.tie;
        const winRate = totalMoves > 0 ? (wins.human / totalMoves) * 100 : 0;
        
        console.log(`  Results: ${wins.human}W-${wins.robot}L-${wins.tie}T (${winRate.toFixed(1)}% win rate)`);
        
        return {
            wins: wins,
            winRate: winRate,
            totalMoves: totalMoves
        };
    }
    
    setRobotConfiguration(combo) {
        // Set difficulty
        const difficultySelect = document.getElementById('difficulty');
        if (difficultySelect) {
            difficultySelect.value = combo.difficulty;
            setDifficulty();
        }
        
        // Set strategy
        const strategySelect = document.getElementById('strategy');
        if (strategySelect) {
            strategySelect.value = combo.strategy;
            setStrategy();
        }
        
        // Set personality
        const personalitySelect = document.getElementById('personality');
        if (personalitySelect) {
            personalitySelect.value = combo.personality;
            setPersonality();
        }
        
        console.log(`  🔧 Robot configured: ${combo.difficulty} + ${combo.strategy} + ${combo.personality}`);
    }
    
    async resetGame() {
        return new Promise((resolve) => {
            if (typeof resetGame === 'function') {
                resetGame();
                setTimeout(resolve, 1000); // Wait for reset to complete
            } else {
                // Fallback: reload page
                location.reload();
            }
        });
    }
    
    async playMove(move) {
        return new Promise((resolve) => {
            if (typeof submitMove === 'function') {
                // Store original updateUI to capture results
                const originalUpdateUI = window.updateUI;
                let moveResult = null;
                
                window.updateUI = function(data) {
                    // Call original function
                    originalUpdateUI.call(this, data);
                    
                    // Capture result
                    if (data && data.result) {
                        moveResult = Array.isArray(data.result) ? data.result[0] : data.result;
                    }
                    
                    // Restore original function
                    window.updateUI = originalUpdateUI;
                    
                    resolve(moveResult);
                };
                
                // Submit the move
                submitMove(move);
            } else {
                console.error('submitMove function not found');
                resolve(null);
            }
        });
    }
    
    displayFinalResults(gameLength) {
        console.log(`\\n` + `=`.repeat(80));
        console.log(`📈 FINAL TEST RESULTS FOR ${gameLength}-MOVE SEQUENCE`);
        console.log(`=`.repeat(80));
        
        // Calculate overall statistics
        const totalTests = this.results.length;
        const avgWinRate = this.results.reduce((sum, r) => sum + r.result.winRate, 0) / totalTests;
        const beatsCount = this.results.filter(r => r.result.winRate > 50).length;
        
        console.log(`🎯 Overall Performance:`);
        console.log(`   Average Win Rate: ${avgWinRate.toFixed(1)}%`);
        console.log(`   Beats: ${beatsCount}/${totalTests} combinations (${(beatsCount/totalTests*100).toFixed(1)}%)`);
        
        // Find best and worst performing combinations
        const sortedResults = [...this.results].sort((a, b) => b.result.winRate - a.result.winRate);
        
        console.log(`\\n🏆 BEST PERFORMING AGAINST:`);
        for (let i = 0; i < Math.min(5, sortedResults.length); i++) {
            const r = sortedResults[i];
            console.log(`   ${i + 1}. ${r.combo.name}: ${r.result.winRate.toFixed(1)}% win rate`);
        }
        
        console.log(`\\n🛡️ WORST PERFORMING AGAINST:`);
        for (let i = 0; i < Math.min(5, sortedResults.length); i++) {
            const r = sortedResults[sortedResults.length - 1 - i];
            console.log(`   ${i + 1}. ${r.combo.name}: ${r.result.winRate.toFixed(1)}% win rate`);
        }
        
        // Create downloadable results
        this.createDownloadableResults(gameLength);
    }
    
    createDownloadableResults(gameLength) {
        const resultsData = {
            gameLength: gameLength,
            sequenceName: this.results[0]?.sequence_name || 'unknown',
            timestamp: new Date().toISOString(),
            summary: {
                totalTests: this.results.length,
                avgWinRate: this.results.reduce((sum, r) => sum + r.result.winRate, 0) / this.results.length,
                beatsCount: this.results.filter(r => r.result.winRate > 50).length
            },
            detailedResults: this.results
        };
        
        const dataStr = JSON.stringify(resultsData, null, 2);
        const dataBlob = new Blob([dataStr], {type: 'application/json'});
        const url = URL.createObjectURL(dataBlob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `optimal_sequence_test_results_${gameLength}moves_${Date.now()}.json`;
        a.style.display = 'none';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        console.log(`\\n💾 Results saved to download file`);
    }
    
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Create and expose the tester
window.optimalTester = new OptimalSequenceTester();

// Usage instructions
console.log(`
🎮 OPTIMAL SEQUENCE TESTER LOADED!

To run the test:
1. Make sure you're on the game page
2. Run one of these commands:
   
   // Test 25-move sequence
   optimalTester.startTest(25);
   
   // Test 50-move sequence  
   optimalTester.startTest(50);

The test will automatically:
- Configure each robot combination
- Play the optimal sequence
- Track results with visual feedback
- Generate a comprehensive report
- Download detailed results as JSON

⚠️  Warning: This will take about 15-20 minutes to complete all 105 combinations!
`);
'''
        
        return js_code
    
    def create_html_test_page(self) -> str:
        """Create a standalone HTML test page."""
        
        html_content = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Optimal Sequence Test Results</title>
    <style>
        body {{
            font-family: 'Courier New', monospace;
            background: #1a1a1a;
            color: #00ff88;
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            border-bottom: 2px solid #00ff88;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .sequence-box {{
            background: #2a2a2a;
            border: 1px solid #00ff88;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }}
        .move-sequence {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(60px, 1fr));
            gap: 10px;
            margin: 15px 0;
        }}
        .move {{
            background: #333;
            border: 1px solid #555;
            border-radius: 4px;
            padding: 8px;
            text-align: center;
            font-weight: bold;
        }}
        .move.paper {{ background: #4CAF50; color: white; }}
        .move.stone {{ background: #FF9800; color: white; }}
        .move.scissor {{ background: #F44336; color: white; }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #333;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #00ff88;
        }}
        .instructions {{
            background: #2a2a4a;
            border: 1px solid #4CAF50;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }}
        .copy-button {{
            background: #00ff88;
            color: #1a1a1a;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
        }}
        .copy-button:hover {{
            background: #00cc66;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎮 Optimal Move Sequences for RPS AI Testing</h1>
            <p>Scientifically designed sequences to maximize win rate against AI opponents</p>
        </div>
        
        <div class="instructions">
            <h2>📋 How to Use</h2>
            <ol>
                <li>Open the main game in another tab</li>
                <li>Copy the JavaScript test code below</li>
                <li>Open browser developer console (F12)</li>
                <li>Paste and run the code</li>
                <li>Follow the console instructions to start automated testing</li>
            </ol>
            <p><strong>⚠️ Warning:</strong> Full testing takes 15-20 minutes as it tests all 105 robot combinations!</p>
        </div>
        
        <div class="sequence-box">
            <h2>🎯 25-Move Optimal Sequence</h2>
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-value">{self.sequences['25_moves']['avg_win_rate']:.1f}%</div>
                    <div>Average Win Rate</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{self.sequences['25_moves']['beats_count']}</div>
                    <div>Combinations Beaten</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{self.sequences['25_moves']['name'].title()}</div>
                    <div>Strategy Type</div>
                </div>
            </div>
            <h3>Sequence:</h3>
            <div class="move-sequence">
                {' '.join([f'<div class="move {move}">{move.title()}</div>' for move in self.sequences['25_moves']['sequence']])}
            </div>
        </div>
        
        <div class="sequence-box">
            <h2>🎯 50-Move Optimal Sequence</h2>
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-value">{self.sequences['50_moves']['avg_win_rate']:.1f}%</div>
                    <div>Average Win Rate</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{self.sequences['50_moves']['beats_count']}</div>
                    <div>Combinations Beaten</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{self.sequences['50_moves']['name'].title()}</div>
                    <div>Strategy Type</div>
                </div>
            </div>
            <h3>Sequence:</h3>
            <div class="move-sequence">
                {' '.join([f'<div class="move {move}">{move.title()}</div>' for move in self.sequences['50_moves']['sequence']])}
            </div>
        </div>
        
        <div class="sequence-box">
            <h2>🤖 JavaScript Test Code</h2>
            <p>Copy this code and paste it into the browser console on the game page:</p>
            <textarea id="jsCode" style="width: 100%; height: 200px; background: #1a1a1a; color: #00ff88; border: 1px solid #555; padding: 10px; font-family: monospace;">// Loading...</textarea>
            <br><br>
            <button class="copy-button" onclick="copyCode()">📋 Copy Code to Clipboard</button>
        </div>
    </div>
    
    <script>
        // Load the JavaScript code
        document.getElementById('jsCode').value = `{self.generate_javascript_test_function().replace('`', '\\`')}`;
        
        function copyCode() {{
            const textarea = document.getElementById('jsCode');
            textarea.select();
            document.execCommand('copy');
            alert('✅ Code copied to clipboard!');
        }}
    </script>
</body>
</html>
'''
        
        return html_content
    
    def generate_visual_test_files(self):
        """Generate all visual test files."""
        
        # Generate JavaScript test function
        js_code = self.generate_javascript_test_function()
        with open('optimal_sequence_test.js', 'w') as f:
            f.write(js_code)
        print("✅ Generated JavaScript test file: optimal_sequence_test.js")
        
        # Generate HTML test page
        html_content = self.create_html_test_page()
        with open('optimal_sequence_test.html', 'w') as f:
            f.write(html_content)
        print("✅ Generated HTML test page: optimal_sequence_test.html")
        
        # Generate quick manual test instructions
        manual_instructions = f'''
# 🎮 Manual Testing Instructions for Optimal Sequences

## Best 25-Move Sequence ({self.sequences['25_moves']['name'].title()})
**Expected Win Rate: {self.sequences['25_moves']['avg_win_rate']:.1f}%**

Sequence: {' → '.join(self.sequences['25_moves']['sequence'])}

## Best 50-Move Sequence ({self.sequences['50_moves']['name'].title()})
**Expected Win Rate: {self.sequences['50_moves']['avg_win_rate']:.1f}%**

Sequence: {' → '.join(self.sequences['50_moves']['sequence'])}

## How to Test Manually:
1. Set game length to 25 or 50 moves
2. Configure robot: Try different difficulty/strategy/personality combinations
3. Play the sequence exactly as shown above
4. Compare your win rate to the expected rate

## Recommended Test Combinations:
### Most Vulnerable (Easy to beat):
- Random + Not to Lose + Any personality
- Frequency + Balanced + Neutral

### Most Resilient (Hardest to beat):
- Markov + Balanced + Chameleon
- Enhanced + To Win + Berserker

## Tips:
- The sequences work best when played exactly as designed
- Some combinations may still be difficult due to randomness
- Results may vary ±10% due to random elements in AI
'''
        
        with open('manual_testing_instructions.md', 'w') as f:
            f.write(manual_instructions)
        print("✅ Generated manual testing instructions: manual_testing_instructions.md")


def main():
    """Main function to generate visual test files."""
    print("🎮 Visual Test Generator for Optimal Move Sequences")
    print("=" * 60)
    
    tester = VisualTester()
    tester.generate_visual_test_files()
    
    print(f"\n📊 Generated test files for sequences:")
    print(f"   • 25-move sequence: {tester.sequences['25_moves']['name']} ({tester.sequences['25_moves']['avg_win_rate']:.1f}% win rate)")
    print(f"   • 50-move sequence: {tester.sequences['50_moves']['name']} ({tester.sequences['50_moves']['avg_win_rate']:.1f}% win rate)")
    
    print(f"\n📁 Files created:")
    print(f"   • optimal_sequence_test.js - JavaScript code for browser console")
    print(f"   • optimal_sequence_test.html - Visual test page with instructions")
    print(f"   • manual_testing_instructions.md - Manual testing guide")
    
    print(f"\n🚀 Next steps:")
    print(f"   1. Open optimal_sequence_test.html in your browser")
    print(f"   2. Copy the JavaScript code from the page")
    print(f"   3. Open your game in another tab")
    print(f"   4. Paste the code in browser console (F12)")
    print(f"   5. Run: optimalTester.startTest(25) or optimalTester.startTest(50)")
    
    print(f"\n✅ Visual test files generated successfully!")


if __name__ == '__main__':
    main()