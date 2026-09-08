"""
ytd_backtest.py — year-to-date betting backtest on a fixed bankroll.

Same rules as gen_equity.py (the production strategy):
  * edge = model_prob - no-vig closing prob;  bet the side with edge >= 15%
  * only bet when the market implied prob for that side is >= 25%
  * Quarter-Kelly stake (0.25x), capped at 15% of bankroll per bet
  * payouts use the actual average closing moneyline (with book vig)
  * bets settled in chronological order, bankroll compounds

Restricted to a date window (default: 2026-01-01 .. today). The window must start
after the model's holdout_start_date or the run aborts — otherwise it would be
scoring fights the model trained on.

Usage:
    python ytd_backtest.py
    python ytd_backtest.py --start 2026-01-01 --end 2026-12-31
    python ytd_backtest.py --bankroll 100
"""

import argparse, sqlite3, pickle, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

EDGE            = 0.15
MIN_MARKET_PROB = 0.25
KELLY_FRACTION  = 0.25
KELLY_CAP       = 0.15


def american_to_decimal(ml: float) -> float:
    return 1 + ml / 100 if ml >= 0 else 1 + 100 / abs(ml)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    ap.add_argument("--bankroll", type=float, default=100.0)
    ap.add_argument("--model", default=None, help="path to a specific model pkl")
    args = ap.parse_args()

    model_path = (Path(args.model) if args.model
                  else sorted(Path("models").glob("ufc_model_[0-9]*.pkl"))[-1])
    artifact = pickle.load(open(model_path, "rb"))
    pipeline     = artifact["pipeline"]
    feature_cols = artifact["features"]
    holdout_start = pd.to_datetime(artifact.get("holdout_start_date"))

    if pd.to_datetime(args.start) < holdout_start:
        sys.exit(f"ABORT: window start {args.start} precedes model holdout_start_date "
                 f"{holdout_start.date()} — would be in-sample.")

    conn = sqlite3.connect("data/ufc.db")
    df = pd.read_sql("SELECT * FROM features ORDER BY event_date", conn)
    odds_raw = pd.read_sql(
        "SELECT db_f1_name, db_f2_name, f1_avg_odds, f2_avg_odds "
        "FROM odds WHERE f1_avg_odds IS NOT NULL", conn)
    conn.close()

    df = df.sort_values("event_date").dropna(subset=["label"])
    df["event_date"] = pd.to_datetime(df["event_date"])
    win = df[(df["event_date"] >= args.start) & (df["event_date"] <= args.end)].copy()
    if win.empty:
        sys.exit("No fights in window.")

    # ── model predictions ────────────────────────────────────────────────────
    X = win.reindex(columns=feature_cols).fillna(0)
    proba = pipeline.predict_proba(X)
    win["prob_f1"] = proba[:, 1]
    win["prob_f2"] = 1.0 - win["prob_f1"]

    bettable = win[win["diff_closing_prob"].notna()].copy()

    # ── attach raw moneylines (orientation-aware) ────────────────────────────
    ml_lookup = {}
    for _, r in odds_raw.iterrows():
        ml_lookup[frozenset([r["db_f1_name"], r["db_f2_name"]])] = (
            r["db_f1_name"], r["f1_avg_odds"], r["f2_avg_odds"])

    def get_ml(row):
        hit = ml_lookup.get(frozenset([row["fighter_1"], row["fighter_2"]]))
        if not hit:
            return pd.Series({"ml_f1": np.nan, "ml_f2": np.nan})
        bfo_f1, ml1, ml2 = hit
        return pd.Series({"ml_f1": ml1, "ml_f2": ml2} if bfo_f1 == row["fighter_1"]
                         else {"ml_f1": ml2, "ml_f2": ml1})

    bettable = bettable.join(bettable.apply(get_ml, axis=1)).dropna(subset=["ml_f1", "ml_f2"])

    bettable["edge_f1"] = bettable["prob_f1"] - bettable["f1_closing_prob"]
    bettable["edge_f2"] = bettable["prob_f2"] - bettable["f2_closing_prob"]

    def pick(row):
        if row["edge_f1"] >= EDGE and row["edge_f1"] >= row["edge_f2"]:
            return "f1"
        if row["edge_f2"] >= EDGE and row["edge_f2"] > row["edge_f1"]:
            return "f2"
        if row["edge_f1"] >= EDGE:
            return "f1"
        return None

    bettable["side"] = bettable.apply(pick, axis=1)
    b = bettable[bettable["side"].notna()].copy()

    rows = []
    for _, r in b.iterrows():
        if r["side"] == "f1":
            won, dec, mkt, p, edge = int(r["label"]) == 1, american_to_decimal(r["ml_f1"]), r["f1_closing_prob"], r["prob_f1"], r["edge_f1"]
            name = r["fighter_1"]
        else:
            won, dec, mkt, p, edge = int(r["label"]) == 0, american_to_decimal(r["ml_f2"]), r["f2_closing_prob"], r["prob_f2"], r["edge_f2"]
            name = r["fighter_2"]
        rows.append({"event_date": r["event_date"], "fighter": name, "opp_side": r["side"],
                     "won": won, "decimal_odds": dec, "market_prob": mkt,
                     "model_prob": p, "edge": edge})
    bets = pd.DataFrame(rows)
    bets = bets[bets["market_prob"] >= MIN_MARKET_PROB].sort_values("event_date").reset_index(drop=True)
    if bets.empty:
        sys.exit("No bets cleared the filters in this window.")

    # ── Quarter-Kelly compounding ───────────────────────────────────────────
    def kelly_f(p, dec):
        b_ = dec - 1
        f = (b_ * p - (1 - p)) / b_
        return min(max(f, 0.0) * KELLY_FRACTION, KELLY_CAP)

    bank = [args.bankroll]
    stakes, fracs = [], []
    for _, r in bets.iterrows():
        f = kelly_f(r["model_prob"], r["decimal_odds"])
        stake = bank[-1] * f
        fracs.append(f); stakes.append(stake)
        bank.append(bank[-1] + stake * (r["decimal_odds"] - 1) if r["won"]
                    else bank[-1] - stake)
    bets["kelly_frac"] = fracs
    bets["stake"] = stakes
    bets["bankroll_after"] = bank[1:]
    bets["flat_pnl_u"] = np.where(bets["won"], bets["decimal_odds"] - 1, -1.0)

    arr  = np.array(bank)
    peak = np.maximum.accumulate(arr)
    dd   = (arr - peak) / peak * 100
    n    = len(bets)
    wr   = bets["won"].mean()
    final = bank[-1]
    flat_roi = bets["flat_pnl_u"].sum() / n * 100
    span_days = (bets["event_date"].max() - bets["event_date"].min()).days or 1
    lr = np.diff(np.log(arr))
    sharpe = (lr.mean() / lr.std()) * np.sqrt(n / span_days * 365.25) if lr.std() > 0 else float("nan")

    # ── report ──────────────────────────────────────────────────────────────
    print("=" * 68)
    print(f"YTD BETTING BACKTEST — {model_path.name}")
    print(f"Window            : {args.start} .. {args.end}")
    print(f"Rules             : edge>={EDGE:.0%}, market>={MIN_MARKET_PROB:.0%}, "
          f"{KELLY_FRACTION:g}x Kelly, {KELLY_CAP:.0%} cap")
    print("=" * 68)
    print(f"Starting bankroll : ${args.bankroll:,.2f}")
    print(f"Ending bankroll   : ${final:,.2f}   ({(final/args.bankroll-1)*100:+.1f}%)")
    print(f"Bets              : {n}   (won {bets['won'].sum()}, lost {n-bets['won'].sum()})")
    print(f"Win rate          : {wr:.1%}")
    print(f"Avg stake         : {bets['kelly_frac'].mean():.1%} of bankroll  (max {bets['kelly_frac'].max():.1%})")
    print(f"Max drawdown      : {dd.min():.1f}%")
    print(f"Flat-stake ROI    : {flat_roi:+.1f}%  ({bets['flat_pnl_u'].sum():+.2f}u / {n}u)")
    print(f"Sharpe (annualz.) : {sharpe:.2f}")
    print(f"Avg decimal odds  : {bets['decimal_odds'].mean():.2f}x   avg edge {bets['edge'].mean():.1%}")

    print("\nMonthly:")
    print(f"  {'Month':<9}{'Bets':>5}{'W-L':>8}{'Flat u':>9}{'Bankroll $':>13}")
    bets["month"] = bets["event_date"].dt.strftime("%Y-%m")
    for m, g in bets.groupby("month"):
        w, l = int(g["won"].sum()), int((~g["won"]).sum())
        print(f"  {m:<9}{len(g):>5}{f'{w}-{l}':>8}{g['flat_pnl_u'].sum():>9.2f}{g['bankroll_after'].iloc[-1]:>13,.2f}")

    print("\nEvery bet:")
    print(f"  {'Date':<11}{'Pick':<22}{'Edge':>6}{'Odds':>7}{'Stake$':>9}{'Result':>8}{'Bankroll$':>12}")
    for _, r in bets.iterrows():
        print(f"  {r['event_date'].strftime('%Y-%m-%d'):<11}{str(r['fighter'])[:21]:<22}"
              f"{r['edge']:>5.0%}{r['decimal_odds']:>7.2f}{r['stake']:>9.2f}"
              f"{'WIN' if r['won'] else 'LOSS':>8}{r['bankroll_after']:>12,.2f}")

    # ── plot ────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        matplotlib.rcParams["text.parse_math"] = False  # treat '$' literally
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import matplotlib.ticker as mticker

        d = [bets["event_date"].iloc[0]] + list(bets["event_date"])
        fig, (axb, axd) = plt.subplots(2, 1, figsize=(12, 8),
                                       gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
        axb.fill_between(d, args.bankroll, arr, color="#22c55e", alpha=0.15)
        axb.plot(d, arr, color="#2563eb", lw=2)
        axb.axhline(args.bankroll, color="#9ca3af", lw=0.8, ls=":")
        axb.annotate(f"  ${final:,.0f}", xy=(d[-1], final), color="#2563eb",
                     fontsize=11, fontweight="bold", va="center")
        axb.set_title("UFC Model — YTD 2026 — Quarter Kelly (0.25x, 15% cap)\n"
                      f"{args.start} .. {args.end}  |  {args.bankroll:.0f} start  |  "
                      f"{n} bets  |  win {wr:.0%}  |  end {final:,.0f} "
                      f"({(final/args.bankroll-1)*100:+.0f}%)",
                      fontsize=11, pad=10)
        axb.set_ylabel("Bankroll ($)")
        axb.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        axb.tick_params(axis="x", labelbottom=False)
        axb.set_xlim(d[0], d[-1]); axb.yaxis.grid(True, color="#e5e7eb", lw=0.8)
        axb.set_axisbelow(True); axb.spines[["top", "right"]].set_visible(False)

        axd.fill_between(d, dd, 0, color="#ef4444", alpha=0.4)
        axd.plot(d, dd, color="#ef4444", lw=1)
        axd.axhline(0, color="#9ca3af", lw=0.8)
        axd.set_ylabel("Drawdown")
        axd.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        axd.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        axd.xaxis.set_major_locator(mdates.MonthLocator())
        axd.set_xlim(d[0], d[-1]); axd.yaxis.grid(True, color="#e5e7eb", lw=0.8)
        axd.set_axisbelow(True); axd.spines[["top", "right"]].set_visible(False)
        plt.setp(axd.xaxis.get_majorticklabels(), rotation=30, ha="right")
        plt.savefig("ytd_equity_curve.png", dpi=150, bbox_inches="tight")
        print("\nSaved ytd_equity_curve.png")
    except Exception as e:
        print(f"\n(plot skipped: {e})")


if __name__ == "__main__":
    main()
