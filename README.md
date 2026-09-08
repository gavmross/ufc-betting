## **UFC Fight Outcome Prediction & Betting System**

A systematic pipeline for predicting UFC fight outcomes and testing for edge against
closing moneylines. A gradient-boosted classifier is trained on 30 years of UFC fight
data with strict temporal validation. The betting layer filters for model edge against
no-vig market prices and is backtested on a ~1,000-fight out-of-sample holdout.

> **Result:** the model predicts winners at **~62% out-of-sample** but **does not beat
> the closing line**, and the edge-filtered betting strategy **loses money** in backtest
> (Quarter-Kelly: $100 → $15 over the holdout; $100 → $26 YTD 2026). This is a negative
> result. See [Betting Strategy & Backtest Results](#betting-strategy--backtest-results)
> and [A note on an earlier bug](#a-note-on-an-earlier-bug).

---

### **Data Foundation**

Built on a purpose-built SQLite database covering **every UFC event from UFC 1 (1994)
through present**. All data is scraped, cleaned, and stored locally.

| Dataset | Rows | Source |
|---|---|---|
| Events | 788 | ufcstats.com |
| Fights (with full per-fight stats) | 8,874 | ufcstats.com |
| Fighter profiles | 4,617 | ufcstats.com |
| ML feature vectors | 6,641 | Computed (features.py) |
| Feature vectors with closing odds | 1,604 | bestfightodds.com (2012–present) |
| Per-fight stats collected | 40+ columns | Strikes, TDs, control time, sub attempts, KDs |

Per-fight stats include full strike breakdowns by **target** (head, body, leg) and
**position** (distance, clinch, ground), scheduled rounds, and fighter career stats
(SLpM, Str Acc, SApM, TD Avg, Sub Avg, DOB).

ufcstats.com now fronts every page with a JavaScript proof-of-work interstitial, so the
scraper drives a headless Chromium via Playwright (`scraper.py`) rather than plain HTTP.

---

### **The Model**

**Holdout: 62.5% pick accuracy | AUC 0.680 | log loss 0.644**
*(997-fight out-of-sample holdout, May 2024 – Sep 2026)*

**Walk-forward CV: 55.1% pick accuracy | AUC 0.584 | prob accuracy 50.3%**
*(temporal folds on the train+calibrate window; barely above coin-flip)*

**Architecture:** `SimpleImputer → StandardScaler → GradientBoostingClassifier`, wrapped
in `CalibratedClassifierCV` (Platt scaling) for calibrated probabilities.

**Training protocol — strict 3-way temporal split, zero leakage:**

```
|──────────── 70% train ────────────|── 15% calibrate ──|── 15% holdout ──|
  Walk-forward CV + grid search        Platt scaling        Never touched
  (18 combos × 7 temporal folds)       only                 until backtest
```

- Walk-forward CV — no random splits
- Calibration set is separate from the holdout (GBM probabilities are overconfident without this)
- Holdout is never seen during training, grid search, or calibration
- Best params: `learning_rate=0.01, max_depth=4, n_estimators=200, subsample=0.8`

**Feature engineering (48 features, no leakage):**

For each fight, features are built using only fights that occurred *before* that fight.
Rolling 7-fight windows (tested 3/5/7/10 — 7 wins on AUC) capture recent form without
lookahead. All rolling stats are differenced (f1 − f2). Fighter order is randomly
swapped 50% of the time to balance labels; every swap-aware feature (and the Elo
columns) is oriented to the swapped order.

| Feature group | Features |
|---|---|
| Rolling averages | Sig strikes, TDs, control time, KDs, sub attempts (landed + attempted) |
| Derived rates | Strike accuracy, TD accuracy, win rate, finish rate |
| Strike breakdown | Head/body/leg accuracy; distance/clinch/ground distribution |
| Elo ratings | Per-weight-class, granular K-factors (tuned), inactivity decay, pre-fight snapshot |
| Physical | Height/reach differential, stance one-hot encoded |
| Context | Layoff (days since last fight), fighter age at fight time, 5-round experience |
| Market | Closing no-vig implied probability (bestfightodds.com, sportsbooks only) |

**Elo system (elo.py):** Per-weight-class ratings computed chronologically with granular
K-factors tuned via grid search — KO/TKO/Sub: K=32, U-DEC/M-DEC: K=30, S-DEC: K=28.
Inactivity decay toward 1500 after 365+ days (1%/year). Pre-fight snapshot written into
features before ratings update. Elo columns are written to the `features` table in the
same randomly-swapped fighter order as the rest of the row (see
[A note on an earlier bug](#a-note-on-an-earlier-bug)).

**Top features by importance:**

| Rank | Feature | Importance |
|---|---|---|
| 1 | diff_elo | 8.6% |
| 2 | diff_avg_td_att | 8.1% |
| 3 | diff_win_rate | 6.0% |
| 4 | f1_days_since_last_fight | 3.8% |
| 5 | diff_leg_acc | 3.6% |
| 6 | diff_avg_td_landed | 3.6% |
| 7 | diff_fights_count | 3.5% |
| 8 | diff_five_round_exp | 3.5% |
| 9 | diff_closing_prob | 3.4% |

Importance is spread thin across many weak features — no single signal dominates.

**Holdout accuracy by confidence bucket:**

| Confidence | Fights | Accuracy |
|---|---|---|
| 50–55% | 255 | 52.9% |
| 55–60% | 181 | 58.0% |
| 60–65% | 207 | 57.0% |
| 65–70% | 156 | 76.3% |
| 70–75% | 111 | 70.3% |
| 75%+ | 87 | 78.2% |

The model has usable discrimination only when it is fairly confident (65%+); in the
50–65% band it is close to a coin-flip. It cannot out-predict the market on the same
fights — market log loss on the holdout is ~0.58 vs the model's 0.64.

---

### **Betting Strategy & Backtest Results**

**Edge filter:** Bet when `model_prob − no_vig_market_prob ≥ 15%` and market probability
of the bet side ≥ 25%. Sizing: Quarter Kelly — `f* = (b·p − q) / b × 0.25`, capped at
15% per bet. Payouts use actual closing moneylines (with book vig). Edge is calculated
against no-vig implied probabilities.

![Equity Curve](equity_curve.png)

**Full holdout (June 2024 – Sep 2026):**

| Metric | Flat stake | Quarter Kelly |
|---|---|---|
| Starting bankroll | $100 | $100 |
| Ending bankroll | — | **$15  (−85%)** |
| ROI per bet | **−29.0%** | — |
| Win rate | 31.5% (23 / 73 bets) | |
| Max drawdown | −25.5u | −89.6% |
| Sharpe (annualized) | — | −1.61 |
| Avg edge "taken" | 20.6% | |
| Avg decimal odds | 2.41x | |

**Year-to-date 2026 (Jan – Sep, Quarter Kelly, `ytd_backtest.py`):**

| Metric | Value |
|---|---|
| Starting bankroll | $100 |
| Ending bankroll | **$26.25  (−73.8%)** |
| Bets | 28  (5 W – 23 L) |
| Win rate | **17.9%** |
| Flat ROI per bet | −60.1% |
| Max drawdown | −77.2% |

**Results by edge bucket (full holdout):**

| Edge | Bets | Win rate | ROI |
|---|---|---|---|
| 15–20% | 39 | 35.9% | −23.0% |
| 20%+ | 34 | 26.5% | −35.8% |

**Favorites vs underdogs (full holdout):**

| | Bets | Win rate | ROI |
|---|---|---|---|
| Favorites (market > 50%) | 15 | 53.3% | −3.6% |
| Underdogs (market ≤ 50%) | 58 | 25.9% | −35.5% |

Every slice is negative. When the model disagrees with the closing line by 15+ points,
the market is right and the model is wrong — the "edge" signal is anti-predictive,
especially on underdogs where the long odds amplify the losses. The closing line is
efficient and this feature set does not beat it.

---

### **A note on an earlier bug**

Earlier versions of this README reported a market-beating model (~75% accuracy, AUC
0.83) and a wildly profitable backtest (Quarter-Kelly $100 → ~$363k, +362,761%). Those
numbers were the product of a **label leak in the Elo features**.

`features.py` randomly swaps `fighter_1`/`fighter_2` 50% of the time and flips the
label. `elo.py` computed Elo in fights-table order (winner always first) and wrote the
`f1_elo` / `f2_elo` / `diff_elo` / `*_elo_peak` columns into the `features` table
**without** applying that swap. So `sign(diff_elo)` encoded which fighter actually won —
the label — and the GBM learned to read it. The Elo columns accounted for ~38% of
feature importance; zeroing them collapsed holdout accuracy from ~79% to ~52%.

The fix orients the Elo columns to the same swapped order as the rest of the row. After
retraining, the model's real out-of-sample performance is what is documented above:
a modest winner-predictor that does not beat the market. `predict_fight()` was
unaffected — it rebuilds the Elo features itself in caller order.

---

### **Technical Architecture**

```
ufcstats.com                      bestfightodds.com
     |  (Playwright headless Chromium,      |
     |   clears JS proof-of-work wall)      |
     v                                      v
scraper.py                        odds_scraper.py
(4 browser pages, rate-limited)   (BFS slug discovery, ~256 events)
     |                                      |
     +-------------------+------------------+
                         |
                    data/ufc.db  (SQLite WAL)
                         |
              +----------+----------+
              |                     |
         features.py            elo.py
    (rolling 7-fight avgs,   (per-weight-class,
     48 features, no leak)    tuned K-factors,
              |               swap-aware write)
              +----------+----------+
                         |
                     model.py
          (GBM + walk-forward CV + Platt scaling)
                         |
              models/ufc_model_YYYYMMDD.pkl
```

All data stored in a single SQLite database (`data/ufc.db`). No intermediate CSVs.
Incremental by default — each module only processes new events on subsequent runs.

---

### **Stack**

Python 3.13 · SQLite WAL · scikit-learn · pandas · numpy · Playwright · BeautifulSoup ·
difflib (fuzzy name matching) · matplotlib

---

### **Running It**

```bash
pip install -r requirements.txt
python -m playwright install chromium

# First run: full historical backfill (UFC 1 -> present)
python scraper.py --full
python features.py
python elo.py
python odds_scraper.py
python model.py

# Incremental update (run after each UFC event)
python scraper.py
python features.py
python elo.py
python odds_scraper.py
python model.py

# Repair fights that have a row but blank per-fight stats (anti-bot outage)
python rescrape_missing_stats.py

# Evaluate
python backtest.py                              # holdout accuracy, AUC, log loss
python bet_backtest.py                          # flat-unit edge analysis (15% edge, mkt >= 25%)
python gen_equity.py                            # Quarter Kelly equity curve → equity_curve.png
python ytd_backtest.py --start 2026-01-01       # year-to-date bankroll simulation

# Predict an upcoming fight
python -c "from model import predict_fight; import json; print(json.dumps(predict_fight('Islam Makhachev', 'Charles Oliveira'), indent=2))"
```

---

### **Predict a Fight**

```python
from model import predict_fight

result = predict_fight("Islam Makhachev", "Charles Oliveira", closing_odds_f1=0.71)
# {
#   "fighter_1": "Islam Makhachev",
#   "fighter_2": "Charles Oliveira",
#   "fighter_1_win_prob": 0.61,
#   "fighter_2_win_prob": 0.39,
#   "predicted_winner": "Islam Makhachev",
#   "confidence": 0.61,
#   "elo": {"Islam Makhachev": {"current": 1820.1, "peak": 1831.4}, ...},
#   "market_consensus": {"Islam Makhachev": 0.71, "Charles Oliveira": 0.29},
#   "f1_profile": {"record": {...}, "striking": {...}, "grappling": {...}, ...},
#   "f2_profile": {...}
# }
```

Fighter names must match ufcstats.com exactly (check the `fighters` table in
`data/ufc.db` if unsure).

---

### **Status**

| Component | Status |
|---|---|
| Data pipeline (UFC 1 – present, 8,874 fights) | Complete |
| Scraper hardened against ufcstats.com anti-bot wall | Complete (Playwright) |
| Feature engineering (48 features, no leakage) | Complete |
| Per-weight-class Elo with tuned K-factors | Complete |
| Closing odds scraper (BFO, sportsbooks only) | Complete |
| GBM + walk-forward CV + Platt calibration | Complete |
| Holdout backtest (997 fights) | Complete — 62.5% accuracy, AUC 0.680 |
| Betting backtest | Complete — **negative edge, strategy loses money** |
| Live prediction interface | Complete — `predict_fight()` |
| Improve pre-2021 odds coverage (BFO name matching) | Planned |
| Style matchup encoding (wrestler vs striker) | Planned |
| Features with genuine edge over the closing line | Open problem |
