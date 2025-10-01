
// Automated Visual Test for Optimal Move Sequences
// Copy and paste this into the browser console while on the game page

class OptimalSequenceTester {
    constructor() {
        this.sequences = {
        "25_moves": {
                "name": "anti_lstm",
                "sequence": [
                        "paper",
                        "stone",
                        "scissor",
                        "paper",
                        "stone",
                        "stone",
                        "scissor",
                        "paper",
                        "paper",
                        "stone",
                        "scissor",
                        "scissor",
                        "paper",
                        "scissor",
                        "stone",
                        "paper",
                        "paper",
                        "paper",
                        "scissor",
                        "paper",
                        "stone",
                        "stone",
                        "stone",
                        "paper",
                        "scissor"
                ],
                "avg_win_rate": 35.46666666666667,
                "beats_count": 3
        },
        "50_moves": {
                "name": "entropy_maximizer",
                "sequence": [
                        "scissor",
                        "paper",
                        "scissor",
                        "paper",
                        "stone",
                        "stone",
                        "scissor",
                        "stone",
                        "paper",
                        "paper",
                        "stone",
                        "paper",
                        "scissor",
                        "stone",
                        "stone",
                        "paper",
                        "scissor",
                        "stone",
                        "paper",
                        "scissor",
                        "stone",
                        "paper",
                        "scissor",
                        "scissor",
                        "scissor",
                        "paper",
                        "paper",
                        "stone",
                        "stone",
                        "stone",
                        "scissor",
                        "scissor",
                        "paper",
                        "stone",
                        "paper",
                        "paper",
                        "scissor",
                        "stone",
                        "stone",
                        "scissor",
                        "scissor",
                        "stone",
                        "scissor",
                        "paper",
                        "paper",
                        "paper",
                        "scissor",
                        "paper",
                        "stone",
                        "stone"
                ],
                "avg_win_rate": 42.34285714285714,
                "beats_count": 20
        }
};
        this.robotCombinations = [
        {
                "difficulty": "random",
                "strategy": "balanced",
                "personality": "neutral",
                "name": "Random Balanced Neutral"
        },
        {
                "difficulty": "random",
                "strategy": "balanced",
                "personality": "berserker",
                "name": "Random Balanced Berserker"
        },
        {
                "difficulty": "random",
                "strategy": "balanced",
                "personality": "guardian",
                "name": "Random Balanced Guardian"
        },
        {
                "difficulty": "random",
                "strategy": "balanced",
                "personality": "chameleon",
                "name": "Random Balanced Chameleon"
        },
        {
                "difficulty": "random",
                "strategy": "balanced",
                "personality": "professor",
                "name": "Random Balanced Professor"
        },
        {
                "difficulty": "random",
                "strategy": "balanced",
                "personality": "wildcard",
                "name": "Random Balanced Wildcard"
        },
        {
                "difficulty": "random",
                "strategy": "balanced",
                "personality": "mirror",
                "name": "Random Balanced Mirror"
        },
        {
                "difficulty": "random",
                "strategy": "to_win",
                "personality": "neutral",
                "name": "Random To Win Neutral"
        },
        {
                "difficulty": "random",
                "strategy": "to_win",
                "personality": "berserker",
                "name": "Random To Win Berserker"
        },
        {
                "difficulty": "random",
                "strategy": "to_win",
                "personality": "guardian",
                "name": "Random To Win Guardian"
        },
        {
                "difficulty": "random",
                "strategy": "to_win",
                "personality": "chameleon",
                "name": "Random To Win Chameleon"
        },
        {
                "difficulty": "random",
                "strategy": "to_win",
                "personality": "professor",
                "name": "Random To Win Professor"
        },
        {
                "difficulty": "random",
                "strategy": "to_win",
                "personality": "wildcard",
                "name": "Random To Win Wildcard"
        },
        {
                "difficulty": "random",
                "strategy": "to_win",
                "personality": "mirror",
                "name": "Random To Win Mirror"
        },
        {
                "difficulty": "random",
                "strategy": "not_to_lose",
                "personality": "neutral",
                "name": "Random Not To Lose Neutral"
        },
        {
                "difficulty": "random",
                "strategy": "not_to_lose",
                "personality": "berserker",
                "name": "Random Not To Lose Berserker"
        },
        {
                "difficulty": "random",
                "strategy": "not_to_lose",
                "personality": "guardian",
                "name": "Random Not To Lose Guardian"
        },
        {
                "difficulty": "random",
                "strategy": "not_to_lose",
                "personality": "chameleon",
                "name": "Random Not To Lose Chameleon"
        },
        {
                "difficulty": "random",
                "strategy": "not_to_lose",
                "personality": "professor",
                "name": "Random Not To Lose Professor"
        },
        {
                "difficulty": "random",
                "strategy": "not_to_lose",
                "personality": "wildcard",
                "name": "Random Not To Lose Wildcard"
        },
        {
                "difficulty": "random",
                "strategy": "not_to_lose",
                "personality": "mirror",
                "name": "Random Not To Lose Mirror"
        },
        {
                "difficulty": "frequency",
                "strategy": "balanced",
                "personality": "neutral",
                "name": "Frequency Balanced Neutral"
        },
        {
                "difficulty": "frequency",
                "strategy": "balanced",
                "personality": "berserker",
                "name": "Frequency Balanced Berserker"
        },
        {
                "difficulty": "frequency",
                "strategy": "balanced",
                "personality": "guardian",
                "name": "Frequency Balanced Guardian"
        },
        {
                "difficulty": "frequency",
                "strategy": "balanced",
                "personality": "chameleon",
                "name": "Frequency Balanced Chameleon"
        },
        {
                "difficulty": "frequency",
                "strategy": "balanced",
                "personality": "professor",
                "name": "Frequency Balanced Professor"
        },
        {
                "difficulty": "frequency",
                "strategy": "balanced",
                "personality": "wildcard",
                "name": "Frequency Balanced Wildcard"
        },
        {
                "difficulty": "frequency",
                "strategy": "balanced",
                "personality": "mirror",
                "name": "Frequency Balanced Mirror"
        },
        {
                "difficulty": "frequency",
                "strategy": "to_win",
                "personality": "neutral",
                "name": "Frequency To Win Neutral"
        },
        {
                "difficulty": "frequency",
                "strategy": "to_win",
                "personality": "berserker",
                "name": "Frequency To Win Berserker"
        },
        {
                "difficulty": "frequency",
                "strategy": "to_win",
                "personality": "guardian",
                "name": "Frequency To Win Guardian"
        },
        {
                "difficulty": "frequency",
                "strategy": "to_win",
                "personality": "chameleon",
                "name": "Frequency To Win Chameleon"
        },
        {
                "difficulty": "frequency",
                "strategy": "to_win",
                "personality": "professor",
                "name": "Frequency To Win Professor"
        },
        {
                "difficulty": "frequency",
                "strategy": "to_win",
                "personality": "wildcard",
                "name": "Frequency To Win Wildcard"
        },
        {
                "difficulty": "frequency",
                "strategy": "to_win",
                "personality": "mirror",
                "name": "Frequency To Win Mirror"
        },
        {
                "difficulty": "frequency",
                "strategy": "not_to_lose",
                "personality": "neutral",
                "name": "Frequency Not To Lose Neutral"
        },
        {
                "difficulty": "frequency",
                "strategy": "not_to_lose",
                "personality": "berserker",
                "name": "Frequency Not To Lose Berserker"
        },
        {
                "difficulty": "frequency",
                "strategy": "not_to_lose",
                "personality": "guardian",
                "name": "Frequency Not To Lose Guardian"
        },
        {
                "difficulty": "frequency",
                "strategy": "not_to_lose",
                "personality": "chameleon",
                "name": "Frequency Not To Lose Chameleon"
        },
        {
                "difficulty": "frequency",
                "strategy": "not_to_lose",
                "personality": "professor",
                "name": "Frequency Not To Lose Professor"
        },
        {
                "difficulty": "frequency",
                "strategy": "not_to_lose",
                "personality": "wildcard",
                "name": "Frequency Not To Lose Wildcard"
        },
        {
                "difficulty": "frequency",
                "strategy": "not_to_lose",
                "personality": "mirror",
                "name": "Frequency Not To Lose Mirror"
        },
        {
                "difficulty": "markov",
                "strategy": "balanced",
                "personality": "neutral",
                "name": "Markov Balanced Neutral"
        },
        {
                "difficulty": "markov",
                "strategy": "balanced",
                "personality": "berserker",
                "name": "Markov Balanced Berserker"
        },
        {
                "difficulty": "markov",
                "strategy": "balanced",
                "personality": "guardian",
                "name": "Markov Balanced Guardian"
        },
        {
                "difficulty": "markov",
                "strategy": "balanced",
                "personality": "chameleon",
                "name": "Markov Balanced Chameleon"
        },
        {
                "difficulty": "markov",
                "strategy": "balanced",
                "personality": "professor",
                "name": "Markov Balanced Professor"
        },
        {
                "difficulty": "markov",
                "strategy": "balanced",
                "personality": "wildcard",
                "name": "Markov Balanced Wildcard"
        },
        {
                "difficulty": "markov",
                "strategy": "balanced",
                "personality": "mirror",
                "name": "Markov Balanced Mirror"
        },
        {
                "difficulty": "markov",
                "strategy": "to_win",
                "personality": "neutral",
                "name": "Markov To Win Neutral"
        },
        {
                "difficulty": "markov",
                "strategy": "to_win",
                "personality": "berserker",
                "name": "Markov To Win Berserker"
        },
        {
                "difficulty": "markov",
                "strategy": "to_win",
                "personality": "guardian",
                "name": "Markov To Win Guardian"
        },
        {
                "difficulty": "markov",
                "strategy": "to_win",
                "personality": "chameleon",
                "name": "Markov To Win Chameleon"
        },
        {
                "difficulty": "markov",
                "strategy": "to_win",
                "personality": "professor",
                "name": "Markov To Win Professor"
        },
        {
                "difficulty": "markov",
                "strategy": "to_win",
                "personality": "wildcard",
                "name": "Markov To Win Wildcard"
        },
        {
                "difficulty": "markov",
                "strategy": "to_win",
                "personality": "mirror",
                "name": "Markov To Win Mirror"
        },
        {
                "difficulty": "markov",
                "strategy": "not_to_lose",
                "personality": "neutral",
                "name": "Markov Not To Lose Neutral"
        },
        {
                "difficulty": "markov",
                "strategy": "not_to_lose",
                "personality": "berserker",
                "name": "Markov Not To Lose Berserker"
        },
        {
                "difficulty": "markov",
                "strategy": "not_to_lose",
                "personality": "guardian",
                "name": "Markov Not To Lose Guardian"
        },
        {
                "difficulty": "markov",
                "strategy": "not_to_lose",
                "personality": "chameleon",
                "name": "Markov Not To Lose Chameleon"
        },
        {
                "difficulty": "markov",
                "strategy": "not_to_lose",
                "personality": "professor",
                "name": "Markov Not To Lose Professor"
        },
        {
                "difficulty": "markov",
                "strategy": "not_to_lose",
                "personality": "wildcard",
                "name": "Markov Not To Lose Wildcard"
        },
        {
                "difficulty": "markov",
                "strategy": "not_to_lose",
                "personality": "mirror",
                "name": "Markov Not To Lose Mirror"
        },
        {
                "difficulty": "enhanced",
                "strategy": "balanced",
                "personality": "neutral",
                "name": "Enhanced Balanced Neutral"
        },
        {
                "difficulty": "enhanced",
                "strategy": "balanced",
                "personality": "berserker",
                "name": "Enhanced Balanced Berserker"
        },
        {
                "difficulty": "enhanced",
                "strategy": "balanced",
                "personality": "guardian",
                "name": "Enhanced Balanced Guardian"
        },
        {
                "difficulty": "enhanced",
                "strategy": "balanced",
                "personality": "chameleon",
                "name": "Enhanced Balanced Chameleon"
        },
        {
                "difficulty": "enhanced",
                "strategy": "balanced",
                "personality": "professor",
                "name": "Enhanced Balanced Professor"
        },
        {
                "difficulty": "enhanced",
                "strategy": "balanced",
                "personality": "wildcard",
                "name": "Enhanced Balanced Wildcard"
        },
        {
                "difficulty": "enhanced",
                "strategy": "balanced",
                "personality": "mirror",
                "name": "Enhanced Balanced Mirror"
        },
        {
                "difficulty": "enhanced",
                "strategy": "to_win",
                "personality": "neutral",
                "name": "Enhanced To Win Neutral"
        },
        {
                "difficulty": "enhanced",
                "strategy": "to_win",
                "personality": "berserker",
                "name": "Enhanced To Win Berserker"
        },
        {
                "difficulty": "enhanced",
                "strategy": "to_win",
                "personality": "guardian",
                "name": "Enhanced To Win Guardian"
        },
        {
                "difficulty": "enhanced",
                "strategy": "to_win",
                "personality": "chameleon",
                "name": "Enhanced To Win Chameleon"
        },
        {
                "difficulty": "enhanced",
                "strategy": "to_win",
                "personality": "professor",
                "name": "Enhanced To Win Professor"
        },
        {
                "difficulty": "enhanced",
                "strategy": "to_win",
                "personality": "wildcard",
                "name": "Enhanced To Win Wildcard"
        },
        {
                "difficulty": "enhanced",
                "strategy": "to_win",
                "personality": "mirror",
                "name": "Enhanced To Win Mirror"
        },
        {
                "difficulty": "enhanced",
                "strategy": "not_to_lose",
                "personality": "neutral",
                "name": "Enhanced Not To Lose Neutral"
        },
        {
                "difficulty": "enhanced",
                "strategy": "not_to_lose",
                "personality": "berserker",
                "name": "Enhanced Not To Lose Berserker"
        },
        {
                "difficulty": "enhanced",
                "strategy": "not_to_lose",
                "personality": "guardian",
                "name": "Enhanced Not To Lose Guardian"
        },
        {
                "difficulty": "enhanced",
                "strategy": "not_to_lose",
                "personality": "chameleon",
                "name": "Enhanced Not To Lose Chameleon"
        },
        {
                "difficulty": "enhanced",
                "strategy": "not_to_lose",
                "personality": "professor",
                "name": "Enhanced Not To Lose Professor"
        },
        {
                "difficulty": "enhanced",
                "strategy": "not_to_lose",
                "personality": "wildcard",
                "name": "Enhanced Not To Lose Wildcard"
        },
        {
                "difficulty": "enhanced",
                "strategy": "not_to_lose",
                "personality": "mirror",
                "name": "Enhanced Not To Lose Mirror"
        },
        {
                "difficulty": "lstm",
                "strategy": "balanced",
                "personality": "neutral",
                "name": "Lstm Balanced Neutral"
        },
        {
                "difficulty": "lstm",
                "strategy": "balanced",
                "personality": "berserker",
                "name": "Lstm Balanced Berserker"
        },
        {
                "difficulty": "lstm",
                "strategy": "balanced",
                "personality": "guardian",
                "name": "Lstm Balanced Guardian"
        },
        {
                "difficulty": "lstm",
                "strategy": "balanced",
                "personality": "chameleon",
                "name": "Lstm Balanced Chameleon"
        },
        {
                "difficulty": "lstm",
                "strategy": "balanced",
                "personality": "professor",
                "name": "Lstm Balanced Professor"
        },
        {
                "difficulty": "lstm",
                "strategy": "balanced",
                "personality": "wildcard",
                "name": "Lstm Balanced Wildcard"
        },
        {
                "difficulty": "lstm",
                "strategy": "balanced",
                "personality": "mirror",
                "name": "Lstm Balanced Mirror"
        },
        {
                "difficulty": "lstm",
                "strategy": "to_win",
                "personality": "neutral",
                "name": "Lstm To Win Neutral"
        },
        {
                "difficulty": "lstm",
                "strategy": "to_win",
                "personality": "berserker",
                "name": "Lstm To Win Berserker"
        },
        {
                "difficulty": "lstm",
                "strategy": "to_win",
                "personality": "guardian",
                "name": "Lstm To Win Guardian"
        },
        {
                "difficulty": "lstm",
                "strategy": "to_win",
                "personality": "chameleon",
                "name": "Lstm To Win Chameleon"
        },
        {
                "difficulty": "lstm",
                "strategy": "to_win",
                "personality": "professor",
                "name": "Lstm To Win Professor"
        },
        {
                "difficulty": "lstm",
                "strategy": "to_win",
                "personality": "wildcard",
                "name": "Lstm To Win Wildcard"
        },
        {
                "difficulty": "lstm",
                "strategy": "to_win",
                "personality": "mirror",
                "name": "Lstm To Win Mirror"
        },
        {
                "difficulty": "lstm",
                "strategy": "not_to_lose",
                "personality": "neutral",
                "name": "Lstm Not To Lose Neutral"
        },
        {
                "difficulty": "lstm",
                "strategy": "not_to_lose",
                "personality": "berserker",
                "name": "Lstm Not To Lose Berserker"
        },
        {
                "difficulty": "lstm",
                "strategy": "not_to_lose",
                "personality": "guardian",
                "name": "Lstm Not To Lose Guardian"
        },
        {
                "difficulty": "lstm",
                "strategy": "not_to_lose",
                "personality": "chameleon",
                "name": "Lstm Not To Lose Chameleon"
        },
        {
                "difficulty": "lstm",
                "strategy": "not_to_lose",
                "personality": "professor",
                "name": "Lstm Not To Lose Professor"
        },
        {
                "difficulty": "lstm",
                "strategy": "not_to_lose",
                "personality": "wildcard",
                "name": "Lstm Not To Lose Wildcard"
        },
        {
                "difficulty": "lstm",
                "strategy": "not_to_lose",
                "personality": "mirror",
                "name": "Lstm Not To Lose Mirror"
        }
];
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
            console.log(`\n🤖 Testing against: ${combo.name} (${i + 1}/${this.robotCombinations.length})`);
            
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
        console.log(`\n` + `=`.repeat(80));
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
        
        console.log(`\n🏆 BEST PERFORMING AGAINST:`);
        for (let i = 0; i < Math.min(5, sortedResults.length); i++) {
            const r = sortedResults[i];
            console.log(`   ${i + 1}. ${r.combo.name}: ${r.result.winRate.toFixed(1)}% win rate`);
        }
        
        console.log(`\n🛡️ WORST PERFORMING AGAINST:`);
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
        
        console.log(`\n💾 Results saved to download file`);
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
