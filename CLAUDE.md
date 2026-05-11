# CLAUDE.md — UFC Fight Predictor

This file gives an AI assistant full context on the project: what it does,
how it's structured, key design decisions, and what to watch out for.

---

## Project Goal

Build a continuously-updating ML pipeline that predicts UFC fight outcomes.
The system scrapes historical fight data, engineers pre-fight features, trains
a gradient boosted classifier, and outputs win probabilities for upcoming fights.

**Secondary goal:** Use this project as a vehicle to learn the underlying math,
statistics, and ML theory — not just to produce a working model, but to
understand why each design decision was made.

---

## Architecture Overview

```
ufcstats.com                   bestfightodds.com
     │                               │
     ▼                               ▼
scraper.py          odds_scraper.py  ← scrapes closing moneyline odds
     │                    │             (fighter BFS discovery, ~256 UFC events)
     ▼                    │
data/ufc.db  ←────────────┘  (writes odds table, merges diff_closing_prob → features)
     │
     ├── features.py ← builds ML-ready feature vectors (rolling pre-fight averages)
     │
     ├── elo.py      ← computes per-fighter Elo ratings (opponent quality signal)
     │
     └── model.py    ← grid search + GBM training, probability calibration, prediction
          │
          ▼
     models/ufc_model_YYYYMMDD.pkl
```

---

## Modules

### `scraper.py`
Scrapes [ufcstats.com](http://ufcstats.com) and writes to `data/ufc.db`.

**Key behaviors:**
- `run_pipeline()` is the entry point. Pass `--full` on first run to backfill all history back to UFC 1. Subsequent runs are incremental — only new events are scraped by diffing against `events` table.
- `init_db()` creates the four-table schema on first run (idempotent).
- Uses `ThreadPoolExecutor(MAX_WORKERS=10)` with a thread-safe `_RateLimiter(RATE_LIMIT=10)` — capped at 10 req/s globally across all threads. Do not raise these limits; this is courtesy to ufcstats.com.
- Uses `INSERT OR REPLACE` (upsert) so re-runs don't create duplicate rows.
- WAL mode is enabled on SQLite for better concurrent read performance.

**Scraped data:**
- `events` — event name, URL, date (ISO format), location
- `fighters` — height, reach, stance, win/loss/draw record, DOB, and career stats (SLpM, Str. Acc., SApM, Str. Def., TD Avg., TD Acc., TD Def., Sub. Avg.). Career stats come from individual profile pages; on incremental runs only fighters from new events are re-fetched.
- `fights` — result (outcome mapped: `W`→`win`, `L`→`loss`), method, round, time, plus raw stat strings for both fighters (knockdowns, sig strikes, takedowns, control time, submission attempts)

**Raw stat format:** Stats come in as strings like `"45 of 102"` or `"4:32"`. Parsing happens downstream in `features.py`, not here.

---

### `features.py`
Reads raw fight data from `ufc.db` and writes a `features` table back to the same DB.

**Critical design principle — no data leakage:**
Fights are processed strictly in chronological order. When building the feature
vector for Fight N, only fights 1 through N-1 are used. After features are built,
Fight N is added to the history. This is enforced by the sequential loop in
`build_feature_dataframe()` — do not refactor this into a vectorized operation
without carefully preserving this ordering guarantee.

**Feature construction:**
- Rolling 7-fight averages per fighter: sig strikes landed/attempted, takedowns, control time, knockdowns, sub attempts
- Derived rates: strike accuracy, TD accuracy, win rate, finish rate
- Physical attributes from `fighters` table: height (inches), reach (inches), stance (one-hot)
- All features are **differenced** (f1 - f2). This encodes relative advantage and makes the model symmetric — if you swap fighter order, the prediction flips correctly.

**Rolling window:** `n_fights=7` (tested 3/5/7/10 — 7 wins on AUC).

**Label balancing:** ufcstats.com always lists the winner as fighter_1. To prevent the model from learning a trivial "always pick f1" shortcut, fighter order is randomly swapped 50% of the time (seeded with `np.random.RandomState(42)` for reproducibility). Labels are flipped accordingly. Result: ~50/50 label balance.

**Label:** `1` if fighter_1 won (`outcome == "win"`), `0` otherwise. Draws and No Contests are excluded from training.

---

### `odds_scraper.py`
Scrapes historical closing moneyline odds from bestfightodds.com and merges
them into the `features` table as `diff_closing_prob`.

**Key behaviors:**
- `collect_all_ufc_slugs()` — discovers all UFC event slugs via BFS:
  1. Seeds from archive page (20 most recent events, typically 3-4 UFC)
  2. Fetches each UFC event page to collect fighter profile links
  3. Fetches each fighter profile to discover more UFC event slugs
  4. Typically finds ~256 UFC events spanning 2012–present (~2 min, 70 HTTP requests)
- `_extract_fights(soup)` — parses the moneyline table. BFO pages have two
  identical table sections; the first ~1,635 rows are navigation-only (1 cell
  per row). Actual odds rows have 8+ cells. Filter: `len(cells) >= 5`.
- `_build_fight()` — averages multi-book odds, removes vig with `remove_vig()`,
  returns no-vig implied probabilities summing to 1.0.
- `match_name()` — fuzzy SequenceMatcher matching (0.82 threshold) to align
  BFO fighter names with ufcstats names in the fighters table.
- `merge_odds_into_features()` — joins `odds` table onto `features` table
  by fighter name (handling 50% random swap), adds three columns:
  `f1_closing_prob`, `f2_closing_prob`, `diff_closing_prob`.
- Called automatically at end of `run_pipeline()`.

**Usage:**
```bash
python odds_scraper.py              # incremental (new events only)
python odds_scraper.py --full       # rescrape everything
python odds_scraper.py --test ufc-perth-4079   # debug single event
```

**`odds` table schema:**
```sql
odds (bfo_slug PK, bfo_event_name, bfo_f1_name, bfo_f2_name,
      db_f1_name, db_f2_name,
      f1_avg_odds, f2_avg_odds, f1_implied_prob, f2_implied_prob, scraped_at)
```

---

### `elo.py`
Computes per-fighter Elo ratings from chronological fight history and writes
them into the `features` table as additional columns.

**Elo update formula:**
```
expected = 1 / (1 + 10^((opponent_elo - your_elo) / 400))
new_elo  = old_elo + K * (actual - expected)
```

**Key design decisions:**
- Ratings are maintained **per weight class** — a heavyweight Elo has no bearing on a lightweight Elo.
- **Granular K-factor by method:** K scales with how decisive the victory was:
  - `KO/TKO`, `SUB` → `K_BASE + K_FINISH_BONUS = 28` (dominant finish)
  - `U-DEC`, `M-DEC` → `K_BASE + K_DECISION_BONUS = 24` (clear judge win)
  - `S-DEC` → `K_BASE = 20` (one judge disagreed on the winner)
  - `DQ`, `Overturned`, `CNC`, `Other` → no rating update (non-competitive)
- **Inactivity decay:** After 365 days of inactivity, a fighter's rating drifts toward the 1500 baseline at `DECAY_RATE = 0.01` per year. This represents uncertainty about current form after a long layoff.
- Pre-fight Elo snapshot is taken **before** updating — no leakage.

**Tunable hyperparameters** (top of `elo.py`, tuned via grid search):
```python
BASE_RATING      = 1500
K_BASE           = 28    # split decision (totals: split=28, dec=30, fin=32)
K_DECISION_BONUS = 2     # added for U-DEC / M-DEC
K_FINISH_BONUS   = 4     # added for KO/TKO / SUB
SCALE            = 400
DECAY_RATE       = 0.01
DECAY_THRESHOLD  = 365
```

**Diagnostics:**
```python
from elo import get_top_rated, plot_rating_history
get_top_rated(weight_class="Lightweight", n=10)  # sanity check
plot_rating_history("Khabib Nurmagomedov")       # career arc
```

**Features added to model:**
- `diff_elo` — rating differential (f1 - f2)
- `f1_elo`, `f2_elo` — raw ratings (absolute skill level, not just relative)
- `f1_elo_peak`, `f2_elo_peak` — career peak rating (ceiling/upside proxy)

---

### `model.py`
Trains a `GradientBoostingClassifier` (scikit-learn) on the `features` table
and exposes a prediction interface.

**Model:** sklearn `GradientBoostingClassifier` wrapped in a `Pipeline` with
`SimpleImputer` (median) and `StandardScaler`, then wrapped in
`CalibratedClassifierCV` (Platt scaling) for probability calibration.

**Hyperparameter tuning:**
Hyperparameters are tuned via `GridSearchCV` with temporal fold indices
(preserving walk-forward ordering). The grid:
```python
PARAM_GRID = {
    "clf__n_estimators":  [100, 200, 300],
    "clf__max_depth":     [2, 3, 4],
    "clf__learning_rate": [0.01, 0.05, 0.1],
    "clf__subsample":     [0.8, 1.0],
}
```
54 combos × 7 folds = 378 fits. Scoring metric: `neg_log_loss`. Best params
are used for the final model. Takes ~5-10 min with `n_jobs=-1`.

**Probability calibration:**
GBMs produce overconfident probabilities. After training, the pipeline is
wrapped in `CalibratedClassifierCV` using Platt scaling (`method="sigmoid"`)
on a dedicated calibration set (middle 15% of data, never used for training
or evaluation). This maps raw scores to true frequencies so "70%" means the
fighter actually wins ~70% of the time.
Uses `FrozenEstimator` to wrap the pre-fitted pipeline (sklearn 1.6+ API).

**Walk-forward cross-validation (`walk_forward_cv`):**
Splits the training set (first 70% of data) into temporal folds. Fold i trains
on the first `i * fold_size` fights and tests on the next `fold_size`. This is
the only correct evaluation strategy for time-series sports data — random splits
cause data leakage. CV never sees the calibration set or the holdout.

**Training flow (`train()`):**
1. Split data into 3 temporal segments: `TRAIN_FRAC=0.70`, `CALIB_FRAC=0.15`, holdout=last 15%.
2. Run `walk_forward_cv()` on the training set only (includes grid search) to find best params.
3. Fit the full pipeline on the training set with best params.
4. Fit `CalibratedClassifierCV(FrozenEstimator(pipeline), method="sigmoid")` on the calibration set.
5. Save calibrated model. Holdout is never touched.

**Key constants:**
```python
TRAIN_FRAC = 0.70   # GBM training + walk-forward CV
CALIB_FRAC = 0.15   # Platt scaling only
# last 15% → holdout, evaluated in backtest.py / bet_backtest.py
```

**Prediction interface:**
```python
from model import predict_fight
# Basic usage
result = predict_fight("Islam Makhachev", "Charles Oliveira")
# With current BFO odds (no-vig implied prob for fighter_1, 0–1)
result = predict_fight("Islam Makhachev", "Charles Oliveira", closing_odds_f1=0.71)
```
Returns calibrated win probabilities plus per-fighter stat profiles:
```python
{
    "fighter_1": "...", "fighter_2": "...",
    "fighter_1_win_prob": 0.643, "fighter_2_win_prob": 0.357,
    "predicted_winner": "...", "confidence": 0.643,
    "elo": {
        "Fighter 1": {"current": 1682.3, "peak": 1701.5},
        "Fighter 2": {"current": 1614.1, "peak": 1650.2},
    },
    "market_consensus": {"Fighter 1": 0.71, "Fighter 2": 0.29},  # only if closing_odds_f1 passed
    "f1_profile": {
        "record":            {"win_rate": 92.3, "finish_rate": 61.5, "fights": 26},
        "striking":          {"sig_str_acc": 52.1, ...},
        "striking_position": {"distance_pct": ..., "clinch_pct": ..., "ground_pct": ...},
        "grappling":         {"td_acc": ..., "ctrl_pct": ...},
        "physical":          {"height_inches": 70.0, "reach_inches": 70.5},
    },
    "f2_profile": { ... },
}
```
Fighter names must match ufcstats.com exactly (check `fighters` table if uncertain).
`closing_odds_f1`: no-vig implied win probability for fighter_1. Convert from American
moneylines using `american_to_prob` + `remove_vig` from `odds_scraper.py`.

**Training output metrics:**
- **Pick accuracy** — how often the model picks the right winner
- **Prob accuracy** — `exp(-log_loss)`, the average probability assigned to the
  correct outcome (random = 50%, perfect = 100%). Reported for both raw and
  calibrated probabilities so you can see the calibration improvement.

**Model persistence:** Saved as `models/ufc_model_YYYYMMDD.pkl`. `load_latest_model()` always loads the most recently dated file.

**Saved artifact contents:**
```python
{
    "pipeline":           CalibratedClassifierCV,  # calibrated wrapper
    "features":           [...],                    # feature column names
    "best_params":        {...},                    # tuned hyperparameters
    "cv_results":         {...},                    # per-fold scores + means
    "feat_importance":    {...},                    # feature importances dict
    "calibration_method": "sigmoid",
    "trained_on":         "...",
}
```
Feature importance: access via `pipeline.estimator.named_steps["clf"]` since
the saved pipeline is a `CalibratedClassifierCV` wrapping a `FrozenEstimator`
wrapping the sklearn `Pipeline`.

---

## Database Schema (`data/ufc.db`)

```sql
events    (event_url PK, event_name, date, location)
fighters  (fighter_url PK, first_name, last_name, nickname,
           height, weight, reach, stance, wins, losses, draws,
           dob, slpm, str_acc, sapm, str_def, td_avg, td_acc, td_def, sub_avg)
fights    (fight_url PK, event_url FK, fighter_1, fighter_2,
           outcome, method, round, time,
           tot_* and sig_* stat columns for f1 and f2)
features  (fight_url PK FK, event_date, fighter_1, fighter_2,
           label, diff_* columns, f1_*/f2_* stat columns,
           f1_elo, f2_elo, diff_elo, f1_elo_peak, f2_elo_peak,
           f1_closing_prob, f2_closing_prob, diff_closing_prob)
odds      (bfo_slug PK, bfo_event_name,
           bfo_f1_name, bfo_f2_name, db_f1_name, db_f2_name,
           f1_avg_odds, f2_avg_odds, f1_implied_prob, f2_implied_prob, scraped_at)
```

Indexes on `fights(fighter_1)`, `fights(fighter_2)`, `fights(event_date)`,
`features(event_date)`.

---

## Running the Pipeline

```bash
# Install dependencies
pip install -r requirements.txt

# First run — full historical backfill (30–60 min)
python scraper.py --full

# Build feature matrix (rolling 7-fight averages, layoff, age, 5-round flag)
python features.py

# Compute Elo ratings (per weight class, granular K-factors)
python elo.py

# Scrape closing odds from bestfightodds.com (~15 min first run, fast incremental)
# Also merges diff_closing_prob into features table automatically
python odds_scraper.py

# Train model + walk-forward CV
python model.py

# Incremental update (run after each UFC event)
python scraper.py
python features.py
python elo.py
python odds_scraper.py
python model.py
```

---

## Known Limitations & Planned Improvements

| Gap | Why it matters | Fix |
|---|---|---|
| ~~No pre-fight betting odds~~ | ~~Closing line is a strong predictor and the best benchmark to beat~~ | ✅ Done — `odds_scraper.py` scrapes bestfightodds.com; `diff_closing_prob` added as feature; AUC 0.793 → 0.835 |
| ~~No layoff/days-since-last-fight feature~~ | ~~Long inactivity is a meaningful signal~~ | ✅ Done — `diff_days_since_last_fight`, `f1/f2_days_since_last_fight` |
| ~~No fighter age at fight time~~ | ~~Decline curves are real~~ | ✅ Done — `diff_age`, `f1/f2_age` |
| Style matchups not encoded | Wrestler vs striker is invisible to the model | Would require tagging fighters by style (manual or NLP on bios) |
| ~~No title fight flag~~ | ~~Both affect fight dynamics~~ | ✅ Done — `is_5round_fight` + `diff_five_round_exp` (proxy: scheduled_rounds == 5) |
| ~~Elo K-factor values not tuned~~ | ~~Granular K-factors by method are implemented but values were defaults~~ | ✅ Done — grid searched; best: K_BASE=28, K_DECISION_BONUS=2, K_FINISH_BONUS=4 (totals: 28/30/32) |
| ~~Rolling window fixed at 5~~ | ~~Arbitrary — may not be optimal~~ | ✅ Done — tested 3/5/7/10; n=7 wins (AUC 0.7999, log loss 0.5496) |

---

## Design Principles

1. **No data leakage.** Every feature must be computable from fights that occurred strictly before the fight being predicted. This is enforced chronologically in `features.py` and `elo.py`. Never use random train/test splits on this dataset.

2. **Incremental by default.** The scraper only fetches new events on each run. This keeps runtime short and is polite to ufcstats.com.

3. **Single DB as source of truth.** All modules read from and write to `data/ufc.db`. No intermediate CSVs.

4. **Differenced features.** All rolling stats are expressed as `f1 - f2`. This encodes relative advantage, removes scale bias, and ensures the model is symmetric with respect to fighter order.

5. **Free data sources only.** ufcstats.com via scraping. No paid APIs.

---

## Current Status (as of 2026-05-10)

**What's done:**
- Full scraping pipeline (incremental + backfill)
- Feature engineering with rolling 7-fight averages (tested 3/5/7/10 — 7 wins), strike targets/positions, control time pct
- Layoff feature (`days_since_last_fight`) and fighter age at fight time — both in model
- Title/main-event flag (`is_5round_fight`) + 5-round experience (`five_round_exp`)
- Per-weight-class Elo with tuned K-factors: K_BASE=28, K_DECISION_BONUS=2, K_FINISH_BONUS=4 (totals: split=28, dec=30, fin=32)
- `apply_decay` bug fixed — ratings already ≤ 1500 are no longer pulled upward
- GBM with GridSearchCV hyperparameter tuning (54 combos × 7 temporal folds)
- 3-way temporal split: 70% train (GBM + CV), 15% calibrate (Platt), 15% holdout (evaluation only) — CV and calibration never see the holdout
- Probability calibration via Platt scaling (CalibratedClassifierCV + FrozenEstimator) on dedicated calibration set
- Moneyline averaging bug fixed: averages implied probabilities across books (not raw American lines) before vig removal
- Betting backtest default threshold set to 15% edge (best ROI/volume tradeoff from bucket analysis)
- predict_fight() accepts `is_title_fight=True/False`, returns calibrated probs + age/layoff + stat profiles
- `build_feature_dataframe` accepts `write_db=False` for in-memory experiments
- `backtest.py` for holdout evaluation; `bet_backtest.py` for betting simulation (flat betting, 15% edge default, market prob >= 25% default)
- `bet_backtest.py` flags: `--threshold` (edge), `--min-market-prob` (cuts extreme longshots; default 0.25 gives +44.7% ROI vs +39.9% unfiltered)

**Current model metrics (ufc_model_20260509.pkl, 976-fight holdout, 2024-02-17 onward):**
- Pick accuracy: 74.7% | AUC: 0.834 | Log loss: 0.5028
- 75%+ confidence bucket: 87.0% accuracy (483 fights)
- Retrained with `diff_closing_prob`, sportsbook-only odds (Kalshi/Polymarket excluded)
- Previous baseline (no odds feature): 73.2% / 0.793 / 0.5285
- Betting backtest (15% edge, 189 bets): +39.9% ROI, max drawdown -6.94u

**What needs to happen next:**
- Style matchup encoding (wrestler vs striker) — requires fighter tagging
- Improve pre-2021 odds coverage: BFO name matching fails for ~99% of 2014-2020 fights (only 1-4 matches/year vs 60-73% in 2021+)
