# UFC Fight Predictor

Scrapes [ufcstats.com](http://ufcstats.com) incrementally, engineers ML features
with zero data leakage, and trains a gradient-boosted classifier to predict UFC fight outcomes.

**Current model (ufc_model_20260509.pkl — 976-fight holdout, 2024-02-17 onward):**
- Pick accuracy: **74.7%** | AUC: **0.834** | Log loss: **0.503**
- 75%+ confidence bucket: **87.0%** accuracy (483 fights)

---

## Project Structure

```
scraper.py          # Scrapes events, fights, fighter profiles + career stats
features.py         # Builds ML-ready feature vectors (rolling pre-fight stats)
elo.py              # Computes per-fighter Elo ratings by weight class
odds_scraper.py     # Scrapes closing moneylines from bestfightodds.com
model.py            # Trains GBM, walk-forward CV, prediction interface
backtest.py         # Holdout evaluation: accuracy, AUC, log loss by confidence bucket
bet_backtest.py     # Betting simulation: flat betting at configurable edge threshold
requirements.txt
data/
  ufc.db            # SQLite database (single source of truth)
models/             # Saved model .pkl files
```

---

## Quickstart

```bash
pip install -r requirements.txt

# First run: backfill ALL historical data (UFC 1 -> present, ~20-25 min)
python scraper.py --full

# Build feature matrix (rolling 7-fight averages, Elo, layoff, age, 5-round flag)
python features.py

# Compute Elo ratings (per weight class, tuned K-factors)
python elo.py

# Scrape closing odds from bestfightodds.com (~15 min first run, fast incremental)
# Automatically merges diff_closing_prob into features table
python odds_scraper.py

# Train model + walk-forward cross-validation
python model.py
```

### Incremental updates (after each UFC event)

```bash
python scraper.py
python features.py
python elo.py
python odds_scraper.py
python model.py
```

### Evaluate the model

```bash
# Holdout accuracy, AUC, log loss by confidence bucket
python backtest.py

# Betting simulation (flat betting, default 15% edge threshold)
python bet_backtest.py
python bet_backtest.py --threshold 0.10   # override threshold
```

---

## Predict a Fight

```python
from model import predict_fight

# Basic usage
result = predict_fight("Islam Makhachev", "Charles Oliveira")

# With closing odds (no-vig implied probability for fighter_1)
result = predict_fight("Islam Makhachev", "Charles Oliveira", closing_odds_f1=0.71)

print(result)
# {
#   "fighter_1": "Islam Makhachev",
#   "fighter_2": "Charles Oliveira",
#   "fighter_1_win_prob": 0.673,
#   "fighter_2_win_prob": 0.327,
#   "predicted_winner": "Islam Makhachev",
#   "confidence": 0.673,
#   "elo": {"Islam Makhachev": {"current": 1820.1, "peak": 1831.4}, ...},
#   "market_consensus": {"Islam Makhachev": 0.71, ...},  # only if closing_odds_f1 passed
#   "f1_profile": {...},
#   "f2_profile": {...},
# }
```

Fighter names must match ufcstats.com exactly (check the `fighters` table in `data/ufc.db` if unsure).

---

## How It Works

### Data Collection (`scraper.py`)
- Scrapes **events** (name, date, location), **fight results** (outcome, method, round, time), and **per-fight stats** (knockdowns, sig strikes, takedowns, control time, sub attempts)
- Scrapes **fighter profiles**: physical attributes (height, reach, stance) and career stats (SLpM, Str. Acc., SApM, Str. Def., TD Avg., TD Acc., TD Def., Sub. Avg., DOB)
- Incremental by default: only scrapes new events each run. `--full` backfills all history
- All data stored in a single SQLite database (`data/ufc.db`)

### Feature Engineering (`features.py`)
For each fight, features are built using only fights that occurred **before** that fight (no leakage):
- Rolling 7-fight averages: significant strikes, takedowns, control time, KDs, sub attempts
- Derived rates: strike accuracy, TD accuracy, win rate, finish rate
- Strike breakdown by target (head, body, leg) and position (distance, clinch, ground)
- Physical: height/reach differential, stance one-hot encoded
- Layoff (`days_since_last_fight`), fighter age at fight time, 5-round experience
- All rolling stats are **differenced** (f1 - f2) to capture relative advantage
- **Random fighter swap** (seeded): labels are ~50/50 (ufcstats.com always lists the winner as fighter_1)

### Elo Ratings (`elo.py`)
- Per-weight-class Elo ratings computed chronologically — no cross-class contamination
- **Granular K-factors** (tuned via grid search):
  - KO/TKO or Submission: K = 32 (most decisive)
  - Unanimous/Majority Decision: K = 30
  - Split Decision: K = 28 (barely a win)
- Inactivity decay: ratings drift toward 1500 after 365+ days without a fight (1%/year)
- Pre-fight Elo snapshot (no leakage) written into features: `diff_elo`, `f1_elo`, `f2_elo`, `f1_elo_peak`, `f2_elo_peak`

### Closing Odds (`odds_scraper.py`)
- Scrapes historical closing moneylines from bestfightodds.com via BFS slug discovery
- Covers ~256 UFC events (2012–present); best coverage 2021+ (~60-73%)
- Averages sportsbook odds only — Kalshi/Polymarket columns are dynamically detected and excluded
- Removes vig to produce no-vig implied probabilities
- Writes `f1_closing_prob`, `f2_closing_prob`, `diff_closing_prob` into the `features` table

### Model (`model.py`)
- **Gradient Boosting Classifier** (scikit-learn) with `SimpleImputer → StandardScaler → GBM` pipeline
- **GridSearchCV** over 54 hyperparameter combos × 7 temporal folds (scoring: neg_log_loss)
- **3-way temporal split**: 70% train (GBM + CV), 15% calibrate (Platt scaling), 15% holdout (evaluation only)
- **Probability calibration** via Platt scaling (`CalibratedClassifierCV`) on the middle 15% — never sees the holdout
- **Walk-forward cross-validation**: trains on past events, tests on future events — the only valid evaluation method for time-series sports data

---

## Database Schema

```sql
events    (event_url PK, event_name, date, location)
fighters  (fighter_url PK, first_name, last_name, nickname,
           height, weight, reach, stance, wins, losses, draws,
           dob, slpm, str_acc, sapm, str_def, td_avg, td_acc, td_def, sub_avg)
fights    (fight_url PK, event_url FK, fighter_1, fighter_2,
           outcome, method, round, time,
           tot_*/sig_* stat columns for f1 and f2)
features  (fight_url PK FK, event_date, fighter_1, fighter_2,
           label, diff_* columns, f1_*/f2_* stat columns,
           f1_elo, f2_elo, diff_elo, f1_elo_peak, f2_elo_peak,
           f1_closing_prob, f2_closing_prob, diff_closing_prob)
odds      (bfo_slug PK, bfo_event_name,
           bfo_f1_name, bfo_f2_name, db_f1_name, db_f2_name,
           f1_avg_odds, f2_avg_odds, f1_implied_prob, f2_implied_prob, scraped_at)
```
