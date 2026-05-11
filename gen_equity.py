import sqlite3, pickle, numpy as np, pandas as pd, sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

bankroll_flat  = [STARTING_BANKROLL]
bankroll_kelly = [STARTING_BANKROLL]
kelly_used = []

for _, row in bets.iterrows():
    # flat 5%
    stake_f = bankroll_flat[-1] * 0.05
    if row["won"]:
        bankroll_flat.append(bankroll_flat[-1] + stake_f * (row["decimal_odds"] - 1))
    else:
        bankroll_flat.append(bankroll_flat[-1] - stake_f)

    # quarter kelly
    f = row["kelly_f"]
    kelly_used.append(f)
    stake_k = bankroll_kelly[-1] * f
    if row["won"]:
        bankroll_kelly.append(bankroll_kelly[-1] + stake_k * (row["decimal_odds"] - 1))
    else:
        bankroll_kelly.append(bankroll_kelly[-1] - stake_k)

dates = [bets["event_date"].iloc[0]] + list(bets["event_date"])
kelly_arr = np.array(bankroll_kelly)
flat_arr  = np.array(bankroll_flat)
peak_k    = np.maximum.accumulate(kelly_arr)
dd_k      = (kelly_arr - peak_k) / peak_k * 100
peak_f    = np.maximum.accumulate(flat_arr)
dd_f      = (flat_arr - peak_f) / peak_f * 100

print("=== Quarter Kelly vs 5% Flat ===")
print(f"Quarter Kelly (cap {KELLY_CAP:.0%}): ${bankroll_kelly[-1]:,.0f}  ({(bankroll_kelly[-1]/STARTING_BANKROLL-1)*100:+.0f}%)")
print(f"5% flat:                            ${bankroll_flat[-1]:,.0f}  ({(bankroll_flat[-1]/STARTING_BANKROLL-1)*100:+.0f}%)")
print(f"Avg Kelly fraction: {np.mean(kelly_used):.1%}  (max {max(kelly_used):.1%})")
print(f"Max drawdown (Kelly): {dd_k.min():.1f}%  |  (flat): {dd_f.min():.1f}%")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 9),
                         gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})

ax = axes[0]
ax.set_facecolor("#0f1117")
fig.patch.set_facecolor("#0f1117")

for i in range(len(bets)):
    color = "#26a641" if bets["won"].iloc[i] else "#da3633"
    ax.axvline(x=bets["event_date"].iloc[i], color=color, alpha=0.13, linewidth=0.8)

ax.plot(dates, bankroll_kelly, color="#58a6ff", linewidth=2.2, zorder=5,
        label="Quarter Kelly (cap 15%)")
ax.plot(dates, bankroll_flat,  color="#8b949e", linewidth=1.2, zorder=4,
        linestyle="--", alpha=0.7, label="5% flat (reference)")
ax.fill_between(dates, kelly_arr, peak_k, where=(kelly_arr < peak_k),
                color="#da3633", alpha=0.15, zorder=3)
ax.axhline(STARTING_BANKROLL, color="#ffffff", alpha=0.12, linewidth=0.8, linestyle=":")

final_k = bankroll_kelly[-1]
ax.scatter([dates[-1]], [final_k], color="#3fb950", s=80, zorder=10)
ax.annotate(f"  ${final_k:,.0f}\n  ({(final_k/STARTING_BANKROLL-1)*100:+.0f}%)",
            xy=(dates[-1], final_k), color="#3fb950",
            fontsize=10, fontweight="bold", va="center")

ax.set_title("UFC Model — Betting Equity Curve (Feb 2024 – May 2026)",
             color="#c9d1d9", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Bankroll ($)", color="#c9d1d9", fontsize=11)
ax.tick_params(colors="#c9d1d9", labelsize=9)
ax.spines[["top","right","bottom","left"]].set_color("#30363d")
ax.yaxis.grid(True, color="#21262d", linewidth=0.8)
ax.set_axisbelow(True)
ax.tick_params(axis="x", labelbottom=False)
ax.set_xlim(dates[0], dates[-1])

stats = (
    f"Edge >= 15% | Market Prob >= 25% | Quarter Kelly sizing (cap 15%)\n"
    f"174 bets  |  73.6% win rate  |  +44.7% flat ROI  |  "
    f"Avg stake: {np.mean(kelly_used):.1%}  |  Max drawdown: {dd_k.min():.1f}%"
)
ax.text(0.01, 0.04, stats, transform=ax.transAxes,
        color="#8b949e", fontsize=8.5, va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#161b22",
                  edgecolor="#30363d", alpha=0.9))
ax.legend(loc="upper left", facecolor="#161b22", edgecolor="#30363d",
          labelcolor="#c9d1d9", fontsize=9)

ax2 = axes[1]
ax2.set_facecolor("#0f1117")
ax2.fill_between(dates, dd_k, 0, color="#da3633", alpha=0.55)
ax2.plot(dates, dd_k, color="#da3633", linewidth=1.0)
ax2.plot(dates, dd_f, color="#8b949e", linewidth=0.8, linestyle="--", alpha=0.6)
ax2.axhline(0, color="#30363d", linewidth=0.8)
ax2.set_ylabel("Drawdown (%)", color="#c9d1d9", fontsize=10)
ax2.tick_params(colors="#c9d1d9", labelsize=9)
ax2.spines[["top","right","bottom","left"]].set_color("#30363d")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax2.yaxis.grid(True, color="#21262d", linewidth=0.8)
ax2.set_axisbelow(True)
ax2.set_xlim(dates[0], dates[-1])
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

plt.savefig("equity_curve.png", dpi=150, bbox_inches="tight",
            facecolor="#0f1117", edgecolor="none")
print("Saved equity_curve.png")
