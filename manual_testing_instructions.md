
# 🎮 Manual Testing Instructions for Optimal Sequences

## Best 25-Move Sequence (Anti_Lstm)
**Expected Win Rate: 35.5%**

Sequence: paper → stone → scissor → paper → stone → stone → scissor → paper → paper → stone → scissor → scissor → paper → scissor → stone → paper → paper → paper → scissor → paper → stone → stone → stone → paper → scissor

## Best 50-Move Sequence (Entropy_Maximizer)
**Expected Win Rate: 42.3%**

Sequence: scissor → paper → scissor → paper → stone → stone → scissor → stone → paper → paper → stone → paper → scissor → stone → stone → paper → scissor → stone → paper → scissor → stone → paper → scissor → scissor → scissor → paper → paper → stone → stone → stone → scissor → scissor → paper → stone → paper → paper → scissor → stone → stone → scissor → scissor → stone → scissor → paper → paper → paper → scissor → paper → stone → stone

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
