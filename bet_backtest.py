"""
Betting strategy backtest — flat betting, 15%+ edge threshold.

Edge = model_prob - no_vig_closing_prob  (model vs fair market value)
Payout = based on actual closing moneyline (with vig, avg across books)

Usage:
    python bet_backtest.py                          # default: 15% edge, market >= 25%
    python bet_backtest.py --threshold 0.10         # override edge threshold
    python bet_backtest.py --min-market-prob 0.30   # only bet when market >= 30%
    python bet_backtest.py --min-market-prob 0.0    # disable market filter
"""

import argparse, sqlite3, pickle
import numpy as np
import pandas as pd
from pathlib import Path


EDGE_THRESHOLD_DEFAULT   = 0.15
MIN_MARKET_PROB_DEFAULT  = 0.25


def american_to_decimal(ml: float) -> float:
    """Convert American moneyline to decimal odds."""
    if ml >= 0:
        return 1 + ml / 100
    else:
        return 1 + 100 / abs(ml)


def run_backtest(edge_threshold: float = EDGE_THRESHOLD_DEFAULT,
                 min_market_prob: float = MIN_MARKET_PROB_DEFAULT):
    # ── Load model ────────────────────────────────────────────────────────────
    models = sorted(Path("models").glob("ufc_model_[0-9]*.pkl"))
    with open(models[-1], "rb") as f:
        artifact = pickle.load(f)

    pipeline     = artifact["pipeline"]
    feature_cols = artifact["features"]

    # ── Load features + raw odds ──────────────────────────────────────────────
    conn = sqlite3.connect("data/ufc.db")
    df = pd.read_sql("SELECT * FROM features ORDER BY event_date", conn)

    # Pull raw avg moneylines from odds table for payout calculation
    odds_raw = pd.read_sql(
        "SELECT db_f1_name, db_f2_name, f1_avg_odds, f2_avg_odds "
        "FROM odds WHERE f1_avg_odds IS NOT NULL",
        conn,
    )
    conn.close()

    # ── Holdout split (identical to backtest.py) ──────────────────────────────
    df = df.sort_values("event_date").dropna(subset=["label"])
    train_frac = artifact.get("train_frac", 0.80)
    calib_frac = artifact.get("calib_frac", 0.00)
    holdout_start = int(len(df) * (train_frac + calib_frac))
    holdout = df.iloc[holdout_start:].copy()

    # ── Model predictions ─────────────────────────────────────────────────────
    X = holdout[[c for c in feature_cols if c in holdout.columns]].fillna(0)
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_cols]

    proba = pipeline.predict_proba(X)
    holdout["prob_f1"] = proba[:, 1]
    holdout["prob_f2"] = 1.0 - holdout["prob_f1"]

    # ── Filter to fights with closing odds ────────────────────────────────────
    bettable = holdout[holdout["diff_closing_prob"].notna()].copy()

    # ── Attach raw moneylines ─────────────────────────────────────────────────
    # Build lookup: frozenset(name_a, name_b) → (f1_ml, f2_ml, bfo_f1, bfo_f2)
    ml_lookup: dict = {}
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

    # ── Edge calculation ──────────────────────────────────────────────────────
    bettable["edge_f1"] = bettable["prob_f1"] - bettable["f1_closing_prob"]
    bettable["edge_f2"] = bettable["prob_f2"] - bettable["f2_closing_prob"]

    # ── Determine bet side ────────────────────────────────────────────────────
    def pick_bet(row):
        if row["edge_f1"] >= edge_threshold and row["edge_f1"] >= row["edge_f2"]:
            return "f1"
        if row["edge_f2"] >= edge_threshold and row["edge_f2"] > row["edge_f1"]:
            return "f2"
        if row["edge_f1"] >= edge_threshold:
            return "f1"
        return None

    bettable["bet_side"] = bettable.apply(pick_bet, axis=1)
    bets = bettable[bettable["bet_side"].notna()].copy()

    # ── Market probability filter ─────────────────────────────────────────────
    if min_market_prob > 0:
        def _bet_market_prob(row):
            if row["bet_side"] == "f1":
                return row["f1_closing_prob"]
            return row["f2_closing_prob"]
        bets["_bet_market_prob"] = bets.apply(_bet_market_prob, axis=1)
        bets = bets[bets["_bet_market_prob"] >= min_market_prob].copy()

    # ── P&L ───────────────────────────────────────────────────────────────────
    def calc_pnl(row):
        if row["bet_side"] == "f1":
            won  = int(row["label"]) == 1
            dec  = american_to_decimal(row["ml_f1"])
            edge = row["edge_f1"]
            model_prob = row["prob_f1"]
            market_prob = row["f1_closing_prob"]
        else:
            won  = int(row["label"]) == 0
            dec  = american_to_decimal(row["ml_f2"])
            edge = row["edge_f2"]
            model_prob = row["prob_f2"]
            market_prob = row["f2_closing_prob"]

        pnl = (dec - 1) if won else -1.0
        return pd.Series({
            "won": won, "pnl": pnl,
            "decimal_odds": dec, "edge": edge,
            "model_prob": model_prob, "market_prob": market_prob,
        })

    bets = bets.join(bets.apply(calc_pnl, axis=1))
    bets["cumulative_pnl"] = bets["pnl"].cumsum()

    # ── Max drawdown ──────────────────────────────────────────────────────────
    peak = bets["cumulative_pnl"].cummax()
    drawdown = bets["cumulative_pnl"] - peak
    max_dd = drawdown.min()

    # ── Print results ─────────────────────────────────────────────────────────
    n      = len(bets)
    won    = bets["won"].sum()
    staked = float(n)
    profit = bets["pnl"].sum()
    roi    = profit / staked * 100 if n else 0

    total_with_odds = len(bettable)
    total_holdout   = len(holdout)

    mkt_filter_str = f">= {min_market_prob:.0%}" if min_market_prob > 0 else "none"
    print("=" * 65)
    print(f"BETTING BACKTEST  --  {models[-1].name}")
    print(f"Edge threshold    : {edge_threshold:.0%}")
    print(f"Market prob filter: {mkt_filter_str}")
    print("=" * 65)
    print(f"Holdout fights    : {total_holdout}")
    print(f"Fights with odds  : {total_with_odds}  ({total_with_odds/total_holdout:.0%} of holdout)")
    print(f"Bets placed       : {n}  ({n/total_with_odds:.0%} of odds-covered fights)")
    print(f"Date range        : {str(bets['event_date'].min())[:10]}  to  {str(bets['event_date'].max())[:10]}")
    print()
    print(f"Win rate          : {won}/{n}  ({won/n:.1%})")
    print(f"Units profit      : {profit:+.2f}u  (staked {staked:.0f}u)")
    print(f"ROI               : {roi:+.1f}%")
    print(f"Max drawdown      : {max_dd:.2f}u")
    print(f"Avg edge taken    : {bets['edge'].mean():.1%}")
    print(f"Avg decimal odds  : {bets['decimal_odds'].mean():.2f}x")
    print()
    print("Note: payouts use actual closing moneylines (with book vig).")
    print("      Edge calculated vs no-vig fair probability.")
    print()

    # ── By edge bucket ────────────────────────────────────────────────────────
    print("Results by edge bucket:")
    print(f"  {'Edge':<12} {'Bets':>5} {'Win%':>7} {'Avg Odds':>10} {'ROI':>8}")
    raw_bins = [0.05, 0.10, 0.15, 0.20, 1.0]
    bins = sorted(set([edge_threshold] + [b for b in raw_bins if b > edge_threshold]))
    bins = bins + [1.0] if bins[-1] != 1.0 else bins
    labels = [f"{int(bins[i]*100)}-{int(bins[i+1]*100)}%" for i in range(len(bins)-1)]
    labels[-1] = f"{int(bins[-2]*100)}%+"
    bets["edge_bucket"] = pd.cut(bets["edge"], bins=bins, labels=labels)
    for lab in labels:
        sub = bets[bets["edge_bucket"] == lab]
        if len(sub) == 0:
            continue
        sub_roi = sub["pnl"].sum() / len(sub) * 100
        print(f"  {lab:<12} {len(sub):>5} {sub['won'].mean():>7.1%} "
              f"{sub['decimal_odds'].mean():>10.2f}x {sub_roi:>+7.1f}%")

    print()

    # ── Favourites vs underdogs ───────────────────────────────────────────────
    favs = bets[bets["market_prob"] > 0.5]
    dogs = bets[bets["market_prob"] <= 0.5]
    print("Favourites vs underdogs:")
    for label, sub in [("Favourites", favs), ("Underdogs ", dogs)]:
        if len(sub) == 0:
            continue
        sub_roi = sub["pnl"].sum() / len(sub) * 100
        print(f"  {label}: {len(sub):>4} bets | win {sub['won'].mean():.1%} | "
              f"ROI {sub_roi:+.1f}%  avg odds {sub['decimal_odds'].mean():.2f}x")

    print()

    # ── Last 20 bets ──────────────────────────────────────────────────────────
    print("Last 20 bets (most recent first):")
    print(f"  {'Date':<12} {'Fighter':<25} {'Edge':>6} {'Odds':>6}  {'Result':>6}  {'P&L':>7}")
    for _, r in bets.tail(20).iloc[::-1].iterrows():
        if r["bet_side"] == "f1":
            name = str(r["fighter_1"])[:24]
        else:
            name = str(r["fighter_2"])[:24]
        ok  = "WIN " if r["won"] else "LOSS"
        date = str(r["event_date"])[:10]
        print(f"  {date:<12} {name:<25} {r['edge']:>5.1%} {r['decimal_odds']:>6.2f}x  "
              f"{ok:>6}  {r['pnl']:>+6.2f}u")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=EDGE_THRESHOLD_DEFAULT,
                        help="Minimum edge to place a bet (default 15%%)")
    parser.add_argument("--min-market-prob", type=float, default=MIN_MARKET_PROB_DEFAULT,
                        help="Minimum market implied probability for the bet side (default 0.25; set 0 to disable)")
    args = parser.parse_args()
    run_backtest(edge_threshold=args.threshold, min_market_prob=args.min_market_prob)
