# Codebase Guide — UFC Fight Predictor

A walkthrough of every script in the pipeline: what it does, how the math works,
and what assumptions are baked into each decision.

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [scraper.py — Data Collection](#scraperpy--data-collection)
3. [features.py — Feature Engineering](#featurespy--feature-engineering)
4. [elo.py — Elo Rating System](#elopy--elo-rating-system)
5. [model.py — Training and Prediction](#modelpy--training-and-prediction)
6. [Assumptions and Limitations](#assumptions-and-limitations)

--- 

## Pipeline Overview

The pipeline runs in four sequential steps. Each reads from and writes to
a single SQLite database (`data/ufc.db`).

```
scraper.py --full     Populate the DB with ~7,000 fights and ~4,400 fighters
        |
        v
features.py           For each fight, compute rolling pre-fight stats for both
                      fighters and write a features table
        |
        v
elo.py                Replay all fights chronologically to compute Elo ratings,
                      then merge them into the features table
        |
        v
model.py              Train a gradient boosted classifier on the features table,
                      evaluate with walk-forward CV, expose predict_fight()
```

Every step enforces a strict rule: **no information from the future is used to
describe the past**. This is called preventing data leakage. If you break this
rule, the model will look great in testing but fail completely on real fights.

---

## scraper.py — Data Collection

### What it scrapes

Three layers of data from ufcstats.com:

| Layer | Source page | What we get |
|---|---|---|
| **Events** | `/statistics/events/completed?page=all` | Name, date (ISO), location for every UFC event ever held |
| **Fights** | Each event's detail page | Fighter names, outcome, weight class, method, round, time, plus per-fight stat tables |
| **Fighter profiles** | Each fighter's individual page | Height, reach, stance, record, career averages (SLpM, Str Acc, SApM, etc.), DOB |

### How fight stats are stored

Per-fight stats come from two HTML tables on each fight detail page:

1. **Totals table** — knockdowns, sig strikes, total strikes, takedowns, sub attempts, reversals, control time
2. **Significant strikes table** — sig strike totals, plus breakdown by **target** (head, body, leg) and **position** (distance, clinch, ground)

Additionally, the fight detail page contains a "Time format" field
(e.g., `"3 Rnd (5-5-5)"` or `"5 Rnd (5-5-5-5-5)"`) which tells us the
**scheduled number of rounds**. This is stored as `scheduled_rounds` and used
downstream to normalize the `avg_round_ended` feature.

**Weight class** is scraped from column 6 of the event page fight table. It comes
directly from the website (e.g., `"Lightweight"`, `"Heavyweight"`) — it is not
inferred from the event name. This field is used by `elo.py` to maintain separate
Elo pools per weight class.

Each stat has two values (one per fighter), stored as raw strings:
- `"45 of 102"` for landed/attempted
- `"4:32"` for control time
- `"67%"` for percentages

Parsing these into numbers happens later in `features.py`, not here. The scraper
just stores the raw text.

### Outcome mapping

ufcstats.com uses abbreviations: `W`, `L`, `D`, `NC`. The scraper maps these
to lowercase words (`win`, `loss`, `draw`, `nc`) because the downstream scripts
check `outcome == "win"`.

### Concurrency

The scraper uses `ThreadPoolExecutor` with 10 worker threads and a global rate
limiter (10 requests/second) to parallelize the three expensive phases:

| Phase | Requests | Approx time |
|---|---|---|
| Event fight lists | ~700 | ~70 sec |
| Fight detail pages | ~8,000 | ~13 min |
| Fighter profiles | ~4,400 | ~7 min |

The rate limiter (`_RateLimiter` class) is thread-safe and ensures no more than
10 requests per second across all threads, regardless of how many workers are
active. This keeps the overall load on ufcstats.com reasonable.

`MAX_WORKERS` and `RATE_LIMIT` are configurable constants at the top of the file.

### Incremental vs full

- `--full`: Treats every event as new. Scrapes all events, all fights (with
  per-fight stat pages), and all ~4,400 fighter profiles. Takes ~20-25 minutes.
- Default (no flag): Only scrapes events not already in the `events` table.
  Fighter profiles are only fetched for fighters who appeared in new events.

### Key assumption

**ufcstats.com always lists the winner as fighter_1.** This is critically
important — it means the raw `outcome` column is `win` for ~98% of fights
(the other 2% being draws/NCs). The `features.py` script handles this with
random swapping (explained below).

---

## features.py — Feature Engineering

This is where raw fight data becomes ML-ready feature vectors. Every design
decision here exists to prevent data leakage.

### Step 1: Parse raw strings into numbers

`extract_fight_stats()` converts the raw stat strings from the scraper:

| Raw column | Parsed into | Example |
|---|---|---|
| `tot_sig_str_f1 = "45 of 102"` | `sig_str_landed_f1 = 45`, `sig_str_att_f1 = 102` | Split on " of " |
| `tot_td_f1 = "3 of 7"` | `td_landed_f1 = 3`, `td_att_f1 = 7` | Split on " of " |
| `tot_ctrl_f1 = "4:32"` | `ctrl_secs_f1 = 272` | Minutes * 60 + seconds |
| `tot_kd_f1 = "2"` | `kd_f1 = 2` | Direct numeric conversion |
| `tot_sub_att_f1 = "1"` | `sub_att_f1 = 1` | Direct numeric conversion |
| `sig_head_f1 = "9 of 12"` | `head_landed_f1 = 9`, `head_att_f1 = 12` | Strike target |
| `sig_body_f1 = "3 of 5"` | `body_landed_f1 = 3`, `body_att_f1 = 5` | Strike target |
| `sig_leg_f1 = "2 of 4"` | `leg_landed_f1 = 2`, `leg_att_f1 = 4` | Strike target |
| `sig_distance_f1 = "8 of 15"` | `distance_landed_f1 = 8`, `distance_att_f1 = 15` | Strike position |
| `sig_clinch_f1 = "2 of 3"` | `clinch_landed_f1 = 2`, `clinch_att_f1 = 3` | Strike position |
| `sig_ground_f1 = "4 of 4"` | `ground_landed_f1 = 4`, `ground_att_f1 = 4` | Strike position |

### Step 2: Build rolling averages (the main loop)

This is the core of the feature pipeline. Fights are sorted chronologically
and processed one at a time in a `for` loop. This loop **cannot** be
vectorized or parallelized without breaking the leakage guarantee.

For each fight:

1. Look up both fighters in the `history` dictionary
2. If both have at least one prior fight, compute rolling averages from
   their last `n_fights=7` fights
3. Build the feature row
4. **After** building the feature row, add this fight to both fighters' history

The rolling average (`rolling_avg()`) computes these stats from the last 7 fights:

| Feature | Formula | What it captures |
|---|---|---|
| `win_rate` | Mean of `won` (0 or 1) over last 7 | Recent form |
| `finish_rate` | Fraction of last 7 won by KO/TKO or Submission | Finishing ability |
| `avg_sig_str_landed` | Mean sig strikes landed per fight | Striking volume |
| `avg_sig_str_att` | Mean sig strikes attempted | Striking activity |
| `avg_td_landed` | Mean takedowns landed | Wrestling pressure |
| `avg_td_att` | Mean takedowns attempted | Wrestling activity |
| `avg_ctrl_secs` | Mean control time in seconds | Grappling dominance |
| `avg_kd` | Mean knockdowns per fight | Power |
| `avg_sub_att` | Mean submission attempts | Submission threat |
| `sig_str_acc` | `avg_landed / avg_attempted` | Striking precision |
| `td_acc` | `avg_td_landed / avg_td_att` | Takedown efficiency |
| `avg_round_ended` | Mean normalized round fraction (see below) | Durability / finishing speed |
| `fights_count` | Total career fights (not rolling) | Experience |
| `avg_head_landed` | Mean head strikes landed | Head targeting volume |
| `avg_body_landed` | Mean body strikes landed | Body targeting volume |
| `avg_leg_landed` | Mean leg strikes landed | Leg kick volume |
| `avg_distance_landed` | Mean strikes landed at distance | Range fighting volume |
| `avg_clinch_landed` | Mean strikes landed in clinch | Clinch fighting volume |
| `avg_ground_landed` | Mean strikes landed on ground | Ground & pound volume |
| `head_acc` | `avg_head_landed / avg_head_att` | Head strike precision |
| `body_acc` | `avg_body_landed / avg_body_att` | Body strike precision |
| `leg_acc` | `avg_leg_landed / avg_leg_att` | Leg kick precision |
| `distance_pct` | `avg_distance_landed / avg_sig_str_landed` | Fraction of work done at range |
| `clinch_pct` | `avg_clinch_landed / avg_sig_str_landed` | Fraction of work done in clinch |
| `ground_pct` | `avg_ground_landed / avg_sig_str_landed` | Fraction of work done on ground |

**Round normalization (`avg_round_ended`):** Raw round numbers are misleading
because main events are 5 rounds while undercard fights are 3. A round 3
decision (3-rounder) and a round 5 decision (5-rounder) both mean "went the
distance," but the raw numbers are 3 vs 5.

The `_round_fraction()` function normalizes each fight's round to a 0–1 scale
by dividing by the scheduled rounds. It uses the scraped `scheduled_rounds`
value when available, with a fallback heuristic for older data:

| Scenario | Raw round | Normalized |
|---|---|---|
| Round 1 KO (3-round fight) | 1 | 0.33 |
| Round 3 Decision (3-round fight) | 3 | 1.0 |
| Round 5 Decision (5-round fight) | 5 | 1.0 |
| Round 4 TKO (5-round fight) | 4 | 0.8 |

### Step 3: Differencing

All rolling stats are expressed as **f1 minus f2**. For example:

```
diff_avg_sig_str_landed = f1_avg_sig_str_landed - f2_avg_sig_str_landed
```

This does three useful things:
1. Encodes **relative advantage** directly — a positive value means f1 is better
2. Makes the model **symmetric** — swapping fighter order flips the sign
3. Reduces the number of features the model needs to learn from

The individual f1/f2 values are also kept in the table, but the model primarily
uses the `diff_*` columns.

### Step 4: Random fighter swap (label balancing)

Because ufcstats.com always lists the winner first, without intervention every
training example would have `label=1`. The model would just learn "always pick
fighter_1" and get ~98% training accuracy while learning nothing useful.

The fix: for each fight, flip a coin (seeded with `RandomState(42)` for
reproducibility). If heads, swap fighter_1 and fighter_2:
- Swap the names
- Swap the rolling stat dictionaries
- Flip the label (1 becomes 0, 0 becomes 1)

The history dictionary is **always updated using the original fighter order**
from ufcstats.com. The swap only affects the feature row output. This means
the rolling averages are computed correctly regardless of the swap.

Result: ~50% of rows have label=1, ~50% have label=0. The model must now
learn from the actual features to predict correctly.

### Step 5: Physical attributes

After the main loop, height, reach, and stance are merged from the `fighters`
table:
- Height and reach are converted to inches (e.g., `6' 1"` becomes `73.0`)
- Stance is one-hot encoded: `Orthodox`, `Southpaw`, `Switch` each get a 0/1 column
- Height and reach differentials are computed (f1 - f2)

### What gets skipped

A fight is **excluded** from the features table if:
- Either fighter has **zero prior fights** in the database (no history to average)
- Fighter names are missing or null
- Outcome is draw or NC (these rows still update fighter history but don't generate a training example)

This means debuting fighters never appear as a training row. At prediction time,
`predict_fight()` uses whatever history is available, so you can still predict
fights involving relatively new fighters — but the model had no examples like
that during training.

---

## elo.py — Elo Rating System

Elo ratings encode something the rolling stats don't: **who you beat matters**.
A fighter who went 4-1 against top-10 opponents is much better than one who
went 4-1 against regional-level competition. Elo captures this because beating
a high-rated opponent gives you more points.

### The core formula

Before each fight, compute the expected score for fighter A against fighter B:

```
E(A) = 1 / (1 + 10^((R_B - R_A) / 400))
```

This is a sigmoid function centered on the rating difference. If A and B have
equal ratings, `E(A) = 0.5`. If A is 400 points above B, `E(A) ≈ 0.91`.

After the fight, ratings update:

```
R_A_new = R_A + K * (S - E(A))
```

Where `S = 1` if A won, `S = 0` if A lost, and `K` controls how much a single
fight moves the rating.

The points are **zero-sum**: whatever the winner gains, the loser loses.

### Design choices

**Per-weight-class ratings.** A fighter's Middleweight Elo has nothing to do
with their Heavyweight Elo (if they ever fought there). This is because the
competition pool is different. The canonical weight classes are mapped from
ufcstats.com's strings (e.g., "UFC Lightweight Championship" maps to
"Lightweight").

**Granular K-factors.** K scales with how decisive the victory was (tuned via grid search):
- KO/TKO or Submission: `K = 32` — most dominant finish
- Unanimous/Majority Decision: `K = 30` — clear judge win
- Split Decision: `K = 28` — one judge disagreed

The reasoning: a finish is a more decisive signal of dominance than a split decision.
Granular K-factors let decisive wins shift ratings more than marginal ones.

**Inactivity decay.** If a fighter hasn't fought in over 365 days, their rating
starts drifting back toward 1500 (the baseline). The formula:

```
decayed = rating + (1500 - rating) * (1 - 0.99^years_inactive)
```

After 1 year of inactivity: ~1% of the gap to 1500 is closed.
After 5 years: ~5% closed. This is intentionally slow — ratings only
already built up through many fights should be hard to fully erase.

**Pre-fight snapshot.** The Elo values written into the features table are
captured **before** the fight's ratings update. This prevents leakage — the
model sees only what was knowable before the fight happened.

### Features added to the model

| Feature | Meaning |
|---|---|
| `diff_elo` | f1 Elo minus f2 Elo — relative skill level |
| `f1_elo`, `f2_elo` | Absolute ratings — encodes overall skill floor |
| `f1_elo_peak`, `f2_elo_peak` | Career-high ratings — encodes ceiling/upside |

### Hyperparameters (tuned via grid search)

```
BASE_RATING      = 1500    Starting rating for all fighters
K_BASE           = 28      Points transferred for a split decision win
K_DECISION_BONUS = 2       Extra K for unanimous/majority decision
K_FINISH_BONUS   = 4       Extra K for KO/TKO or Submission
SCALE            = 400     Controls how steep the expected score curve is
DECAY_RATE       = 0.01    1% decay toward baseline per year of inactivity
DECAY_THRESHOLD  = 365     Days of inactivity before decay kicks in
```

K-factor values were tuned via grid search optimizing holdout AUC.
The totals (split=28, decision=30, finish=32) outperform the standard chess K=32/24 split.

---

## model.py — Training and Prediction

### The classifier

A **Gradient Boosting Machine** (GBM) from scikit-learn, wrapped in a pipeline:

```
SimpleImputer(strategy="median")  →  StandardScaler()  →  GradientBoostingClassifier
```

1. **Imputer**: Fills missing values with the column median. This handles fighters
   who have some stats missing (e.g., no takedown attempts in their last 5 fights).
2. **Scaler**: Standardizes features to mean=0, std=1. GBMs don't technically
   need this (they use decision boundaries, not distances), but it doesn't hurt
   and makes feature importance more comparable.
3. **GBM**: Builds sequential decision trees (number tuned via grid search),
   each correcting the errors of the previous ones.

### GBM hyperparameters

Hyperparameters are selected via `GridSearchCV` over 54 combinations × 7 temporal folds,
scored by neg_log_loss. The search grid:

```python
PARAM_GRID = {
    "clf__n_estimators":  [100, 200, 300],
    "clf__max_depth":     [2, 3, 4],
    "clf__learning_rate": [0.01, 0.05, 0.1],
    "clf__subsample":     [0.8, 1.0],
}
```

**Why these ranges?**
- `max_depth 2-4`: Shallow trees prevent overfitting. Each tree captures simple
  interactions (e.g., "if Elo diff > 100 AND reach diff > 3, lean toward f1").
  Deeper trees would memorize noise.
- `learning_rate 0.01-0.1`: Small learning rate + many trees converges more
  smoothly than large learning rate + few trees.
- `subsample 0.8-1.0`: Stochastic boosting — each tree only sees a fraction of
  the data, acting like bagging to reduce overfitting.

### Walk-forward cross-validation

This is the evaluation strategy. It simulates how the model would perform if
you trained it at various points in history and predicted future fights.

With `n_splits=7`, the data is divided into 8 equal chunks by date:

```
|--chunk 1--|--chunk 2--|--chunk 3--|--chunk 4--|--chunk 5--|--chunk 6--|--chunk 7--|--chunk 8--|

Fold 1: train=[1]                   test=[2]
Fold 2: train=[1,2]                 test=[3]
Fold 3: train=[1,2,3]               test=[4]
Fold 4: train=[1,2,3,4]             test=[5]
Fold 5: train=[1,2,3,4,5]           test=[6]
Fold 6: train=[1,2,3,4,5,6]         test=[7]
Fold 7: train=[1,2,3,4,5,6,7]       test=[8]
```

Each fold trains only on fights that happened **before** the test fights. This
is the only valid evaluation method for time-series sports data. Random
train/test splits would let the model train on a 2024 fight to predict a 2020
fight, which is leakage.

### Metrics reported

| Metric | What it means |
|---|---|
| **Accuracy** | Fraction of fights where the predicted winner was correct |
| **AUC (ROC)** | How well the model ranks fights by confidence. 0.5 = random, 1.0 = perfect. Measures discrimination ability independent of threshold |
| **Log loss** | Penalizes confident wrong predictions. Lower is better. This is the metric that most directly measures probability calibration |

### How predict_fight() works

When you call `predict_fight("Israel Adesanya", "Joe Pyfer")`:

1. Loads the latest saved model from `models/`
2. Looks up each fighter's **most recent appearance** in the features table to
   get their last known rolling stats
3. Recomputes current Elo ratings by replaying all fights through `elo.compute_elo()`
4. Builds a single feature row: computes all differentials (f1 - f2), adds
   stance one-hots, adds Elo features
5. Feeds the feature row through the pipeline (impute, scale, predict)
6. Returns win probabilities for both fighters

### What the final model trains on

The data is split into three temporal segments:

1. **Training set (first 70%)** — GBM is fitted here; walk-forward CV and grid
   search also run on this portion only. CV scores are honest because the
   calibration and holdout sets are never touched during this phase.
2. **Calibration set (middle 15%)** — The fitted GBM is wrapped in
   `CalibratedClassifierCV` (Platt scaling) here. This corrects the GBM's
   overconfident raw scores into well-calibrated probabilities.
3. **Holdout (last 15%)** — Used only in `backtest.py` and `bet_backtest.py`
   for honest evaluation. Never seen during training or calibration.

---

## Assumptions and Limitations

These are the assumptions baked into the model. Every one of them is a potential
source of error.

### Data assumptions

1. **ufcstats.com is accurate.** We trust that the scraped stats (strikes,
   takedowns, control time) are correct. There's no cross-validation against
   another source.

2. **Fighter names are consistent.** The system matches fighters by full name
   string. If ufcstats.com spells a name differently across events (e.g.,
   nickname changes, name corrections), those get treated as different fighters.

3. **Outcome = "win" always means fighter_1 won.** The entire label system
   depends on ufcstats.com consistently listing the winner first.

### Feature assumptions

4. **Last 7 fights are representative.** The rolling window of 7 is arbitrary.
   A fighter who had one bad performance in their last 7 gets that weighted the
   same as their other 6 fights. A fighter on a 10-fight win streak looks the
   same as one on a 7-fight win streak (the `win_rate` over 7 is identical).

5. **All fights are weighted equally.** A fight against the #1 contender counts
   the same as a fight against a debuting opponent in the rolling averages.
   (Elo partially compensates for this, but the rolling stats don't.)

6. **Raw volume stats are meaningful across eras.** Striking volume has
   increased over UFC's history. A fighter averaging 50 sig strikes in 2010
   was elite; in 2024 that's average. The model doesn't adjust for this.

7. **Physical attributes are static.** Height and reach don't change, but the
   model treats them as fixed values from the most recent scrape. This is fine
   for height/reach but wouldn't work for attributes that change (e.g., weight).

8. **Stance one-hot encoding is simplistic.** The model knows Orthodox vs
   Southpaw but doesn't model the **matchup** (Southpaw vs Orthodox is a
   meaningfully different dynamic than Orthodox vs Orthodox). It just sees
   each fighter's stance independently.

### Elo assumptions

9. **Elo transfers are zero-sum within weight class.** A fighter moving up
   in weight starts from scratch at 1500. Their Lightweight Elo doesn't
   carry over to Welterweight.

10. **K-factor is the same for all fighters.** In chess, new players have a
    higher K (ratings move faster) and established players have a lower K.
    Our model uses the same K (28/30/32) for everyone. A debuting fighter and a
    20-fight veteran are treated identically.

11. **The K-factor bonus is a blunt instrument.** A first-round KO against the
    champion and a third-round submission against a debuting opponent both get
    K=32. Ideally the bonus would scale with opponent quality.

12. **Inactivity decay is linear and uniform.** A 365-day layoff due to injury
    is treated the same as a 365-day layoff due to suspension or choice.
    The decay rate (1%/year, `DECAY_RATE=0.01`) is a guess, not empirically optimized.

### Model assumptions

13. **Probability calibration has limits.** The model applies Platt scaling
    (`CalibratedClassifierCV`) on a dedicated calibration set (middle 15% of
    data, separate from the evaluation holdout), which maps raw GBM scores
    to better-calibrated probabilities. However, the model's probability
    distribution is still more dispersed than the market's (~0.31 std vs ~0.18
    std for closing line), so high-confidence outputs should be treated with
    some skepticism.

14. **Missing features are filled with median.** If a feature is null (e.g., a
    fighter has no takedown attempts), it's replaced with the median of that
    column across all fights. This is a reasonable default but means the model
    can't distinguish "no data" from "average."

15. **No interaction features.** The model sees `diff_avg_td_landed` and
    `diff_avg_sig_str_landed` independently. It can learn simple interactions
    (GBMs do this via tree splits) but it has no explicit "wrestler vs striker"
    matchup feature.

16. **3-way temporal split.** The walk-forward CV runs on the first 70% of
    data only, so the CV score isn't contaminated by calibration or holdout
    fights. The middle 15% is used solely for Platt scaling. The final 15% is
    a true holdout — never seen during training, grid search, or calibration.
    The CV score is an honest generalization estimate, not inflated by leakage
    into the holdout.

17. **The model has limited concept of context.** Scheduled rounds (3 vs 5) are
    now used to normalize `avg_round_ended`, but other contextual factors —
    title fight stakes, altitude, short-notice replacements, weight misses —
    are not in the feature set. They all affect outcomes.

### What this means in practice

The model captures the most important predictors of UFC outcomes (recent form,
striking/grappling ability, physical attributes, opponent-adjusted skill via Elo,
and closing odds consensus). It is probability-calibrated via Platt scaling and
achieves **62.5% pick accuracy on a 997-fight holdout (AUC 0.680)**. It has usable
discrimination only in its 65%+ confidence buckets and **does not beat the closing
line** (model holdout log loss 0.64 vs market 0.58). Treat its outputs as a
calibrated starting point for analysis, not as a betting edge — the edge-filtered
strategy loses money in backtest.

*(An earlier version of this guide cited 74.7% / AUC 0.834. Those numbers came from
a label leak in the Elo features, fixed 2026-09-07 — see CLAUDE.md.)*

