import sqlite3, pickle, numpy as np, pandas as pd, sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

def american_to_decimal(ml):
    return 1 + ml / 100 if ml >= 0 else 1 + 100 / abs(ml)

models = sorted(Path("models").glob("ufc_model_[0-9]*.pkl"))
with open(models[-1], "rb") as f:
    artifact = pickle.load(f)

pipeline     = artifact["pipeline"]
feature_cols = artifact["features"]
train_frac   = artifact.get("train_frac", 0.80)
calib_frac   = artifact.get("calib_frac", 0.00)

conn = sqlite3.connect("data/ufc.db")
df       = pd.read_sql("SELECT * FROM features ORDER BY event_date", conn)
odds_raw = pd.read_sql(
    "SELECT db_f1_name, db_f2_name, f1_avg_odds, f2_avg_odds FROM odds WHERE f1_avg_odds IS NOT NULL",
    conn)
conn.close()

df = df.sort_values("event_date").dropna(subset=["label"])
holdout_start = int(len(df) * (train_frac + calib_frac))
holdout = df.iloc[holdout_start:].copy()

X = holdout[[c for c in feature_cols if c in holdout.columns]].fillna(0)
for col in feature_cols:
    if col not in X.columns:
        X[col] = 0
X = X[feature_cols]

proba = pipeline.predict_proba(X)
holdout["prob_f1"] = proba[:, 1]
holdout["prob_f2"] = 1.0 - holdout["prob_f1"]
bettable = holdout[holdout["diff_closing_prob"].notna()].copy()

ml_lookup = {}
for _, r in odds_raw.iterrows():
    key = frozenset([r["db_f1_name"], r["db_f2_name"]])
    ml_lookup[key] = (r["db_f1_name"], r["f1_avg_odds"], r["f2_avg_odds"])

def get_ml(row):
    key = frozenset([row["fighter_1"], row["fighter_2"]])
    if key not in ml_lookup:
        return pd.Series({"ml_f1": np.nan, "ml_f2": np.nan})
    bfo_f1, ml1, ml2 = ml_lookup[key]
    if bfo_f1 == row["fighter_1"]:
        return pd.Series({"ml_f1": ml1, "ml_f2": ml2})
    else:
        return pd.Series({"ml_f1": ml2, "ml_f2": ml1})

bettable = bettable.join(bettable.apply(get_ml, axis=1))
bettable = bettable.dropna(subset=["ml_f1", "ml_f2"])
bettable["edge_f1"] = bettable["prob_f1"] - bettable["f1_closing_prob"]
bettable["edge_f2"] = bettable["prob_f2"] - bettable["f2_closing_prob"]

EDGE = 0.15
def pick_bet(row):
    if row["edge_f1"] >= EDGE and row["edge_f1"] >= row["edge_f2"]:
        return "f1"
    if row["edge_f2"] >= EDGE and row["edge_f2"] > row["edge_f1"]:
        return "f2"
    if row["edge_f1"] >= EDGE:
        return "f1"
    return None

bettable["bet_side"] = bettable.apply(pick_bet, axis=1)
bets_raw = bettable[bettable["bet_side"].notna()].copy()

rows = []
for _, row in bets_raw.iterrows():
    if row["bet_side"] == "f1":
        won = int(row["label"]) == 1
        dec = american_to_decimal(row["ml_f1"])
        mkt = row["f1_closing_prob"]
        p   = row["prob_f1"]
    else:
        won = int(row["label"]) == 0
        dec = american_to_decimal(row["ml_f2"])
        mkt = row["f2_closing_prob"]
        p   = row["prob_f2"]
    pnl = (dec - 1) if won else -1.0
    rows.append({"won": won, "decimal_odds": dec, "market_prob": mkt,
                 "event_date": pd.to_datetime(row["event_date"]),
                 "pnl": pnl, "model_prob": p})

all_bets = pd.DataFrame(rows)
bets = all_bets[all_bets["market_prob"] >= 0.25].copy().reset_index(drop=True)

KELLY_FRACTION = 0.25
KELLY_CAP      = 0.15
STARTING_BANKROLL = 100

def kelly_f(p, dec_odds):
    b = dec_odds - 1
    q = 1 - p
    f_full = (b * p - q) / b
    f_full = max(f_full, 0)
    return min(f_full * KELLY_FRACTION, KELLY_CAP)

bets["kelly_f"] = bets.apply(lambda r: kelly_f(r["model_prob"], r["decimal_odds"]), axis=1)

bankroll_kelly = [STARTING_BANKROLL]
kelly_used = []

for _, row in bets.iterrows():
    f = row["kelly_f"]
    kelly_used.append(f)
    stake_k = bankroll_kelly[-1] * f
    if row["won"]:
        bankroll_kelly.append(bankroll_kelly[-1] + stake_k * (row["decimal_odds"] - 1))
    else:
        bankroll_kelly.append(bankroll_kelly[-1] - stake_k)

dates     = [bets["event_date"].iloc[0]] + list(bets["event_date"])
kelly_arr = np.array(bankroll_kelly)
peak_k    = np.maximum.accumulate(kelly_arr)
dd_k      = (kelly_arr - peak_k) / peak_k * 100

n_bets   = len(bets)
win_rate = bets["won"].mean()
flat_roi = bets["pnl"].sum() / n_bets * 100
final_k  = bankroll_kelly[-1]

print(f"Quarter Kelly: ${final_k:,.0f}  ({(final_k/STARTING_BANKROLL-1)*100:+.0f}%)")
print(f"Avg Kelly fraction: {np.mean(kelly_used):.1%}  (max {max(kelly_used):.1%})")
print(f"Max drawdown: {dd_k.min():.1f}%")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 8),
                         gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})

ax = axes[0]

ax.fill_between(dates, STARTING_BANKROLL, kelly_arr,
                color="#22c55e", alpha=0.15, zorder=2)
ax.plot(dates, kelly_arr, color="#2563eb", linewidth=2.0, zorder=5)
ax.axhline(STARTING_BANKROLL, color="#9ca3af", linewidth=0.8, linestyle=":", zorder=1)

ax.annotate(f"  ${final_k:,.0f}",
            xy=(dates[-1], final_k),
            color="#2563eb", fontsize=11, fontweight="bold", va="center")

title_line1 = "UFC Model — Quarter Kelly (0.25x, 15% cap)"
title_line2 = (f"Feb 2024 – May 2026  ·  BestFightOdds Closing Lines  ·  $100 start"
               f"  |  Win rate {win_rate:.1%}  |  {n_bets} bets  |  {flat_roi:+.1f}% ROI per bet")
ax.set_title(f"{title_line1}\n{title_line2}", fontsize=11, pad=10)

ax.set_ylabel("Bankroll ($)", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.tick_params(axis="x", labelbottom=False)
ax.set_xlim(dates[0], dates[-1])
ax.yaxis.grid(True, color="#e5e7eb", linewidth=0.8)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

ax2 = axes[1]
ax2.fill_between(dates, dd_k, 0, color="#ef4444", alpha=0.4, zorder=2)
ax2.plot(dates, dd_k, color="#ef4444", linewidth=1.0, zorder=3)
ax2.axhline(0, color="#9ca3af", linewidth=0.8, zorder=1)
ax2.set_ylabel("Drawdown", fontsize=10)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax2.yaxis.grid(True, color="#e5e7eb", linewidth=0.8)
ax2.set_axisbelow(True)
ax2.spines[["top", "right"]].set_visible(False)
ax2.set_xlim(dates[0], dates[-1])
ax2.set_xlabel("Date", fontsize=11)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

plt.savefig("equity_curve.png", dpi=150, bbox_inches="tight")
print("Saved equity_curve.png")
