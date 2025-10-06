# Paper-Scissor-Stone (Adaptive Opponent Edition)

Paper-Scissor-Stone is an experiment in opponent modelling for the classic
Rock–Paper–Scissors game. The project now centres on an **adaptive opponent
engine** that separates difficulty, strategy, and personality so we can generate
dozens of distinct AI behaviours, stress-test them against scripted human
simulators, and monitor performance with reproducible reports.

---

## Highlights

- **Adaptive AI core** – `rps_ai_system.py` blends an exponential-forgetting
  human model, strategy-layer payoffs, and personality-specific risk/variance
  shaping. Difficulties control pattern depth/speed, strategies weight win vs.
  tie objectives, and personalities modulate how the AI acts on its read.
- **Human move simulators** – `human_simulators.py` supplies four archetypes:
  `random`, `realistic`, `strategic`, and `explorer`. Use them to test how the
  AI responds to everyday, adaptive, or worst-case human play.
- **Performance tester** – `opponent_performance_tester.py` sweeps every
  difficulty/strategy/personality combination against each simulator, saves
  detailed CSVs, produces box plots (with quartile annotations), and emits a
  stats report (ANOVA + descriptive tables).
- **Regression guardrails** – `tests/test_adaptive_ai.py` locks in baseline
  behaviour using the embedded simulator so changes to the AI can be validated
  quickly.

---

## Project Layout

```
├── README.md
├── ARCHITECTURE.md          # Deep dive into the adaptive engine
├── human_simulators.py      # Human move generators (random/realistic/etc.)
├── opponent_performance_tester.py
├── rps_ai_system.py         # Adaptive opponent implementation
├── game_context.py          # Backend bridge for the Flask app
├── webapp/                  # Flask UI and assets
└── tests/                   # Pytest-based regression coverage
```

Legacy modules (`parameter_synthesis_engine.py`, `markov_predictor.py`, etc.)
remain in the tree for reference but are no longer used by the adaptive system.

---

## Getting Started

1. **Install dependencies**

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run the web app**

   ```bash
   cd webapp
   FLASK_APP=app.py flask run
   ```

   The UI exposes difficulty/strategy/personality controls, live stats, and
   developer tooling to inspect predictions.

3. **Run the regression suite**

   ```bash
   pytest tests/test_adaptive_ai.py -q
   ```

   These tests ensure random play stays balanced, fast-switching humans still
   beat aggressive opponents, and personality variance remains within expected
   bounds. Use them whenever tuning difficulty/strategy presets.

---

## Simulation & Reporting Pipeline

### Human simulators

All simulators share a common interface:

```python
from human_simulators import SIMULATOR_REGISTRY
sim_cls = SIMULATOR_REGISTRY['strategic']
simulator = sim_cls(np.random.default_rng(123))
```

- **random** – uniformly samples moves.
- **realistic** – emulates common human habits (rock bias, win-stay/lose-shift).
- **strategic** – uses the AI’s human prediction to choose the least expected
  move, with a mild win-stay tendency.
- **explorer** – optimises expected payoff from the AI’s announced move
  distribution (upper bound / worst case adversary).

### Opponent tester

Use `opponent_performance_tester.py` to sweep the opponent matrix:

```bash
python opponent_performance_tester.py \
  --games 30 \
  --rounds 90 \
  --seed 20241109
```

Outputs are written to `simulation_results/adaptive_report_<timestamp>/`:

- `game_results.csv` – per-game round counts and win/tie classification.
- `summary_results.csv` – aggregated rates by simulator/difficulty/strategy/
  personality.
- `*_boxplots.png` – grouped box plots (human/robot/tie game win rates) with
  Med/Q1/Q3 annotations.
- `statistical_report.txt` – ANOVA tables (η² effect sizes) and descriptive
  statistics for each factor.
- `manifest.json` – paths to every generated artifact.

Tip: the explorer simulator always wins game-level metrics because it receives
perfect knowledge of the AI distribution each round. Adjust the simulator (or
mask the AI metadata) if you need a bounded adversary for day-to-day tuning.

---

## Custom Experiments

- **Parameter tuning** – edit the difficulty/strategy/personality presets in
  `rps_ai_system.py`, run `pytest`, then re-run the performance tester to compare
  before/after reports.
- **New simulators** – add a subclass in `human_simulators.py`, register it in
  `SIMULATOR_REGISTRY`, and the tester will include it automatically.
- **Frontend hooks** – expose `difficulty_notes`, `strategy_notes`, and
  `personality_notes` (returned by `game_context.get_ai_prediction`) to display
  richer opponent descriptions in the UI.

---

## Roadmap Ideas

- Introduce a "bounded explorer" that has incomplete information, to create a
  more realistic yet challenging adversary.
- Replace the legacy fallback logic in `webapp/app.py` with a lighter-weight
  sampling policy or remove it entirely once the adaptive engine is stable for
  deployment.
- Automate nightly simulation runs and collect historical trends for each
  opponent configuration.

---

## License

This project currently has no explicit license. Treat it as internal work in
progress unless a license is added.
