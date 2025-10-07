# Paper, Scissors & Strategy

Welcome to the adaptive Rock–Paper–Scissors arena!  
This project powers a web-based experience where you can battle a shape-shifting AI opponent that learns from your moves in real time. The focus of this public release is pure gameplay: pick a difficulty, choose a strategy flavour, add a personality twist, and try to outsmart the robot.

---

## Play The Game

| Item | Details |
|------|---------|
| **UI** | Single-page web app (Flask backend + rich frontend dashboard) |
| **Modes** | Human vs AI (single-player) |
| **Rounds** | 25 by default (configurable) |
| **Controls** | Click the buttons or press `P` (paper), `R` (rock), `S` (scissors) once the match starts |
| **Status Board** | Live win/tie counts, confidence gauges, personality banter, streak alerts |

### Opponent Settings

1. **Difficulty**  
   - *Rookie* – forgiving, plenty of randomness.  
   - *Challenger* – balanced learner with mid-tier pattern tracking.  
   - *Master* – reacts fast to streaks and short patterns.  
   - *Grandmaster* – uses a blended ML ensemble, punishes repetition, far less exploration.

2. **Strategy**  
   - *To Win* – picks the counter to the most likely human move and takes risks for victory.  
   - *Not To Lose* – prioritises avoiding losses, aiming for safe wins or ties.

3. **Personality**  
   Each personality re-shapes confidence and move choices. For example:  
   - *Aggressive* (Berserker) sharpens counterattacks after your streaks.  
   - *Defensive* (Guardian) prefers ties when you’re on a roll.  
   - *Unpredictable* (Wildcard) injects volatility to keep you guessing.  
   - *Cautious* (Professor) hunts deeper patterns methodically.  
   - *Confident* (Mirror) mirrors your playstyle to beat you at your own game.  
   - *Chameleon* adapts dynamically whenever performance dips.

Start a new match any time with the **Reset** button. The AI retains no memory between games unless you opt into developer features (disabled in this public release).

---

## Quick Start

1. **Setup**
   ```bash
   python -m venv venv
   source venv/bin/activate       # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Launch the web app**
   ```bash
   cd webapp
   flask --app app run
   ```
   Visit `http://127.0.0.1:5000` in your browser.

3. **Play**
   - Choose difficulty, strategy, and personality.
   - Hit **Start**, then click `Paper`, `Rock`, or `Scissors` or use the keyboard shortcuts.
   - Watch the dashboard for AI confidence, win rates, streaks, and banter.

---

## What Makes The AI Tick?

- **Adaptive human model** – Tracks your move frequencies, multi-order patterns, and how recently you’ve switched tactics. Grandmaster blends in an ML ensemble for extra bite.
- **Strategy layer** – Converts predictions into move scores. *To Win* goes for the jugular; *Not To Lose* hunts for safe results.  
- **Personality layer** – Adds character. Aggression, risk tolerance, and confidence sensitivity all alter the final move and the tone of the banter.
- **Anti-repeat guard** – Higher difficulties get stricter when you spam the same move.
- **Developer-friendly instrumentation** – Simulations, analytics, and deeper docs live in the developer readme (see below).

---

## Tips & Tricks

- Try mixing your moves early. Grandmaster penalises streaks quickly once it spots them.  
- Personalities can work for or against you. Playing against *Wildcard*? Expect chaos. Facing *Guardian*? Exploit its desire to tie.  
- The dashboard’s “Recent Momentum” card highlights your 10-round bias. If you see “Paper 70 %”, so does the AI.  
- Want a chill match? Stick with Rookie/Neutral/Not-To-Lose. Looking for pain? Grandmaster/To Win/Aggressive.

---

## Advanced & Developer Resources

- **Developer documentation** lives in `DEVELOPER_README.md` (ignored by default, intended for contributors).  
- **Simulation output** from the in-house test suite is stored under `simulation_results/`.  
- **Tests** – Run `python - <<'PY'\nimport pytest\npytest.main(['tests/test_adaptive_ai.py','-q'])\nPY` to validate AI behaviour.

---

## License & Contributions

This build is released for gameplay evaluation only. No license has been published yet; treat it as “look, don’t redeploy” until a formal license lands.

Bug reports or gameplay feedback are welcome via GitHub issues once the repository is public. For substantial code changes, please consult the developer readme before opening pull requests.

Have fun, and may your counters land on time! 🪨📄✂️
