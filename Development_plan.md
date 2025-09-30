# AI RPS -- Development Plan (Streamlined)

> **Goal:** Ship a web-first Rock--Paper--Scissors game that uses AI to
> (1) predict the human's next move (updating every 5 rounds), and (2)
> analyze & explain strategy shifts in plain language. Design is
> offline-first, small, and explainable, with optional iOS/macOS later.

------------------------------------------------------------------------

## 1) Objectives (Product & Learning)

-   **Challenge the Player:**\
    AI predicts likely next human move by learning **short-term habits**
    and **long-term tendencies**; adapts every 5 rounds to avoid being
    exploitable.
-   **Build Awareness:**\
    On demand, app analyzes both players' histories, **detects strategy
    shifts** (e.g., repeating → cycling; defensive → aggressive), and
    explains them in **simple language** with small visuals.
-   **Educate through Play:**\
    Demonstrate how AI **detects, exploits, and adapts** to patterns;
    show that even simple models can reveal human predictability.

------------------------------------------------------------------------

## 2) Architecture & Tech Choices

**Stack & tooling** - **Web first:** TypeScript/React (Vite), Zustand
(state), Vitest.\
- **Engine packages:** TS for gameplay & analyzers; Python for
training/export (LSTM/HMM).\
- **ML:** Markov (TS) baseline; LSTM (PyTorch → ONNX/TF-Lite export);
HMM (hmmlearn/pomegranate); change-point (ruptures/χ²).\
- **Packaging:** ONNX/TF-Lite for LSTM; JSON for HMM config.\
- **CI:** GitHub Actions --- lint, tests, **size/perf budgets**.

**Performance & size budgets** - **Inference:** \< **5 ms** per turn.\
- **Update burst (every 5 rounds):** \< **80 ms**.\
- **Model/bundle:** ≤ **300 KB** (quantized).\
- **Local-only by default**; no PII.

------------------------------------------------------------------------

## 3) Monorepo Layout (Top-Level)

    ai-rps/
    ├─ apps/web/                # React app (web-first)
    ├─ packages/engine-js/      # TS engine: predictor, analyzer, coach
    ├─ packages/sim/            # TS simulators & harness
    ├─ python/engine_py/        # PyTorch LSTM, HMM, exporters
    ├─ python/sim_py/           # Python simulators (parity)
    ├─ models/                  # exported ONNX/TF-Lite & HMM configs
    └─ .github/workflows/       # CI pipelines (lint/test/budgets)

*(Keep the detailed tree you already have; it's excellent for
onboarding.)*

------------------------------------------------------------------------

## 4) Core Engine & Models

### 4.1 Contracts (shared types)

-   **Move:** `R | P | S`\
-   **Round:** `{ id, num, human, bot, winner, ts, latency_ms? }`\
-   **PredictRequest/Response:** `[p(R), p(P), p(S)]`, `botMove`,
    `confidence`\
-   **StrategyReport:** `segments[]`, `changes[]`, `tips[]`,
    `bot_adaptations?[]`

### 4.2 Predictor (robot next-move)

-   **Baseline:** Variable-order **Markov (n=1..3)** with **decayed
    counts** (λ≈0.95/round) and **Katz-style backoff**.\
-   **Primary:** **Tiny LSTM** (embed=8, hidden=24, 1--2 layers,
    dropout≤0.1).\
-   **Policy:** choose counter to argmax; if `max(p_next)< τ(0.45)` add
    ε-greedy randomness (ε=0.1) to prevent unfairness/exploitability.\
-   **Adaptation cadence:** every **5 rounds** --- short fine-tune steps
    on recent window (≤200 rounds, recency-weighted).\
-   **Export:** PyTorch → ONNX/TF-Lite (quantized); load in web
    (onnxruntime-web or TF.js).

**High-level implementation guidance** - JS **Markov**: one file +
tests; expose `update(history[])` and `predict(lastK[])`.\
- Py **LSTM**: tiny network + brief training loop; export; keep a CLI
for batch export.\
- Web integration: feature flag `predictor.mode = 'markov' | 'lstm'`.

### 4.3 Analyzer (on-demand)

-   **Change-points:** Offline PELT or simple **χ²** drift on move
    distributions + **windowed features** (`repeat_prob_5`,
    `cycle_score_5`, etc.).\
-   **HMM states:** 3--5 hidden states; map to labels {**defensive**,
    **cyclic**, **randomized**, **exploitative**}.\
-   **Clustering:** 10-round segments; k-means (k=3--5) → label by
    centroid features.\
-   **Coach:** turn report into 3--5 **plain-language tips** and "**what
    to try next**" experiments.

**High-level guidance** - TS analyzer: start with χ² change-points +
label mapping; add HMM later.\
- Python analyzer: fit HMM for research/offline validation; export label
map JSON.

------------------------------------------------------------------------

## 5) Web Application

### 5.1 Scaffolding (web)

-   **Pages:** *Play*, *Insights* (tabs).\
-   **Components:** HUD (confidence chip), Move buttons, Timeline,
    Tips.\
-   **State:** rounds\[\] + derived metrics; persistent local store
    (IndexedDB).\
-   **Controls:** Reset, Undo (for demos), "Insight now".

### 5.2 UX Design (user-facing GUI)

-   **Top bar:** title, tabs, confidence chip (Low/Med/High).\
-   **Play panel:** round header; big move buttons; **Result card** with
    brief "**why**" (e.g., "predicted repeat, 58% confidence");
    micro-telemetry (confidence, randomness, streak).\
-   **Insights panel:** strategy **timeline** (bands + change markers),
    **metrics cards** (predictability, cycle, win rate by phase, bot
    adaptation), **coach notes** (3--5 bullets), **what to try next**,
    **export report** (JSON/PDF).\
-   **Settings:** difficulty (entropy/adaptation), hints toggle, local
    storage/export, short "About AI".

**Empty states** - Rounds \< 5 → "warming up."\
- No shifts yet → "we'll highlight once stable."\
- Low confidence → "your mix is hard to predict (good!)."

**Accessibility** - Text + emoji labels; ≥44px targets; high-contrast
mode; screen-reader strings ("You: Rock. AI: Paper. AI wins. Confidence
high because you repeated.").

### 5.3 Developer Metrics Console (debug & optimization)

> Toggle via **Settings → Developer Mode** (hidden in production unless
> `DEV_MODE`). Local-only; no PII.

**Predictor performance** - **Top-1 accuracy** (rolling N=50 / session)\
- **Log loss / Brier** (calibration)\
- **Confidence histogram** (max p_next)\
- **Exploitability index** (% wins from exact counters)\
- **Adaptation latency** (rounds to recover ≥80% accuracy after shift)

**Analyzer quality** - **Change-point precision/recall** (±2 rounds vs
synthetic/consensus)\
- **Segment stability** (under small parameter perturbations)\
- **Label distribution** across {defensive, cyclic, randomized,
exploitative}

**Gameplay & UX** - **Win-rate by phase**, **avg streak length**,
**repeat rate** (last 10/50)\
- **Hint funnel** (impressions → opens → behavior change)\
- **Frustration heuristics** (rapid resets/undos; exit after losing
streak \>K)

**Perf & footprint** - **Inference p50/p95** (\<5 ms), **update burst**
(\<80 ms), **bundle/model size** (warn \>300 KB), **peak memory** during
bursts

**Drift & health** - **Concept drift** (KL vs prior window),
**Calibration drift (ECE)**, **Anomaly log** (e.g., high confidence +
repeated misses)

**Views & controls** - Views: *Live* (last 50), *Session*, *Compare*
(A/B τ, ε)\
- Controls: sampling rate, window sizes, thresholds (τ, ε, χ²)\
- Exports: CSV/JSON time series; snapshot compare between model versions

**Acceptance** - Console render \<50 ms; no impact on turn latency\
- All metrics exportable to JSON; ≥1 compare report per build

------------------------------------------------------------------------

## 6) Simulation, Testing & CI

**Synthetic human types (TS & Py)** - **Repeater**, **Cycler**,
**Mirror-after-loss**, **Random-with-bias**, **Shifter** (switches
strategy every N rounds)

**Harness** - Batch matches vs predictors; report **accuracy**,
**time-to-adapt**, **win-rate by phase**.

**Assertions** - Predictor ≥ **+10pp** over Markov(1) on Cycler/Shifter\
- Change-point detection within **±2 rounds** on Shifter\
- Coach yields **≥3 actionable tips** on typical session

**CI rules** - `lint` + `test` for JS/Py\
- **Budget checks:** bundle ≤300 KB; avg predict \<5 ms on Node (sim)\
- Artifact: **StrategyReport** JSON sample from simulated run

------------------------------------------------------------------------

## 7) Product Flags & Settings

-   `predictor.mode`: `'markov' | 'lstm'`\
-   `policy`: `{ tau: 0.45, epsilon: 0.1 }`\
-   `analyzer`: `{ window: 10, chi2_thresh: 6 }`\
-   Toggles: **hints**, **insights**, **export**, **developer mode**

------------------------------------------------------------------------

## 8) Roadmap & Milestones

**M0 -- Core loop** - Markov predictor + policy\
- Game UI + HUD + storage\
- Basic metrics (accuracy, win rate)

**M1 -- LSTM** - Train TinyLSTM on synthetic players (Python)\
- Export ONNX; load in web (onnxruntime-web / TF.js)\
- Hook periodic fine-tune queue (every 5 rounds)

**M2 -- Analyzer** - χ² change-point + features (TS)\
- Segment clustering + labels (TS)\
- Coach tips generator (TS)

**M3 -- UX & size** - Timeline chart + markers\
- Quantization & bundle budget check\
- Settings panel (entropy/hints)

**M4 -- Validation** - TS/Py simulators parity tests\
- A/B thresholds (τ, ε) on sims\
- Generate sample **StrategyReport** (export)

------------------------------------------------------------------------

## 9) Evaluation & Acceptance (Consolidated)

**Predictor**\
- **Top-1 next-move accuracy** ≥ baseline +10pp (on sims)\
- **Calibration** reasonable (Brier/log-loss)\
- **Adaptation** improvement ≤ **10 rounds** after a shift

**Analyzer & Insights**\
- **≥80%** of simulated change-points detected within **±2 rounds**\
- Segments stable; labels non-degenerate\
- Tips: **clear, non-jargony**, rated helpful in quick user tests

**Performance & Size**\
- Inference \<5 ms; update burst \<80 ms; model/bundle ≤300 KB

**Developer Console**\
- Renders \<50 ms; JSON export works; compare report generated

------------------------------------------------------------------------

## 10) Release & Market Readiness

**App Stores & Education** - Offline-first; no PII; clear privacy
policy\
- Disclose "AI Coach Insights" as **predictive & probabilistic**\
- Classroom/kiosk flag; **report export** (JSON/PDF)

**Monetization paths (optional)** - **Casual:** free + premium insights
(\$2.99--\$4.99)\
- **Education:** \$49--\$199 per classroom\
- **Consulting/demo:** \$2K--\$10K per workshop

------------------------------------------------------------------------

## 11) Implementation Tips (High-Level, No Big Code)

-   **Keep engine pure & pluggable.** Engine has no UI deps; web app
    uses thin adapters.\
-   **Start with Markov.** Simple, predictable baseline with tests; wire
    predictors via `predictor.mode`.\
-   **Gate LSTM behind a flag.** Ship baseline; add ONNX model loading +
    tiny fine-tune loop when stable.\
-   **Analyzer in steps.** χ² change-points first → add labels →
    (optionally) HMM for richer segmentation.\
-   **Metrics early.** Add accuracy/log-loss + time-to-adapt in M0;
    don't wait for M2.\
-   **Budgets in CI.** Fail PRs that exceed 300 KB bundle or 5 ms
    average predict.\
-   **Explainability first.** Always pair the result with a **why** line
    & confidence chip; throttle hints.\
-   **Export everywhere.** StrategyReport JSON; metrics CSV/JSON for
    debugging and education.

------------------------------------------------------------------------

### Appendix: Feature & Metric Glossary

-   `repeat_prob_5`: probability of repeating previous move (last 5
    rounds)\
-   `switch_prob_5`: probability of switching gesture (last 5)\
-   `cycle_score_5`: tendency to rotate R→P→S (or inverse)\
-   **Exploitability index:** fraction of AI wins caused by exact
    counters (vs random)\
-   **Adaptation latency:** rounds from detected shift → restored
    accuracy\
-   **Concept drift (KL):** divergence between current vs prior move
    distributions\
-   **ECE:** expected calibration error of predictor confidence
