"""
ufc_predictor/elo.py

Computes per-fighter Elo ratings chronologically from fight history.
Ratings are maintained per weight class and updated after each fight.

Key design choices:
  - Ratings are computed BEFORE each fight (no leakage)
  - Finish bonus: KO/TKO and submissions transfer extra points
  - Inactivity decay: ratings drift toward baseline after long layoffs
  - Separate Elo pools per weight class
  - Pre-fight Elo diff is written into the features table in ufc.db

Usage:
    python -m ufc_predictor.elo            # compute and store Elo
    python -m ufc_predictor.elo --plot     # also plot rating distributions
"""

import sqlite3
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import contextmanager

log = logging.getLogger(__name__)

DATA_DIR = Path("data")
DB_PATH  = DATA_DIR / "ufc.db"

# ─────────────────────────────────────────────
# ELO HYPERPARAMETERS  (tune these)
# ─────────────────────────────────────────────

BASE_RATING      = 1500   # starting Elo for every fighter
K_BASE           = 28     # base points transferred (split decision — barely a win)
K_DECISION_BONUS = 2      # extra K for unanimous/majority decision (clear win)
K_FINISH_BONUS   = 4      # extra K for KO/TKO or submission (dominant win)
SCALE            = 400    # standard Elo scale factor
DECAY_RATE       = 0.01   # fraction to decay toward base per year of inactivity
DECAY_THRESHOLD  = 365    # days of inactivity before decay kicks in

# Method → K bonus mapping (above K_BASE)
# KO/TKO, SUB       → K_BASE + K_FINISH_BONUS   = 32  (most decisive)
# U-DEC, M-DEC      → K_BASE + K_DECISION_BONUS = 30  (clear judge win)
# S-DEC             → K_BASE                     = 28  (barely a win)
# DQ, Overturned, CNC, Other → no rating update
FINISH_METHODS   = {"KO/TKO", "SUB"}
DECISION_METHODS = {"U-DEC", "M-DEC"}
SPLIT_METHODS    = {"S-DEC"}
NO_UPDATE_METHODS = {"DQ", "Overturned", "CNC", "Other"}

# Canonical weight class names from ufcstats.com
WEIGHT_CLASSES = [
    "Strawweight", "Flyweight", "Bantamweight", "Featherweight",
    "Lightweight", "Welterweight", "Middleweight", "Light Heavyweight",
    "Heavyweight", "Women's Strawweight", "Women's Flyweight",
    "Women's Bantamweight", "Women's Featherweight",
]


# ─────────────────────────────────────────────
# DB HELPERS
# ─────────────────────────────────────────────

@contextmanager
def get_db(db_path: Path = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def ensure_elo_columns(db_path: Path = DB_PATH):
    """Add Elo columns to features table if they don't already exist."""
    cols_to_add = [
        ("f1_elo",      "REAL"),
        ("f2_elo",      "REAL"),
        ("diff_elo",    "REAL"),
        ("f1_elo_peak", "REAL"),
        ("f2_elo_peak", "REAL"),
    ]
    with get_db(db_path) as conn:
        existing = {row[1] for row in conn.execute("PRAGMA table_info(features)").fetchall()}
        for col, dtype in cols_to_add:
            if col not in existing:
                conn.execute(f"ALTER TABLE features ADD COLUMN {col} {dtype}")
                log.info(f"Added column {col} to features table")


# ─────────────────────────────────────────────
# CORE ELO LOGIC
# ─────────────────────────────────────────────

def expected_score(rating_a: float, rating_b: float) -> float:
    """Probability that fighter A beats fighter B given their ratings."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / SCALE))


def k_factor(method: str) -> float:
    """
    K-factor scales with how decisive the victory was:
      KO/TKO, SUB  → highest (finish = strongest signal)
      U-DEC, M-DEC → medium  (all/most judges agree)
      S-DEC        → lowest  (one judge disagreed on the winner)
    """
    m = str(method).strip()
    if m in FINISH_METHODS:
        return K_BASE + K_FINISH_BONUS
    if m in DECISION_METHODS:
        return K_BASE + K_DECISION_BONUS
    return K_BASE


def apply_decay(rating: float, last_fight_date: datetime, current_date: datetime) -> float:
    """
    Decay a fighter's rating toward BASE_RATING during long inactivity.
    Represents uncertainty about a fighter's current level after a layoff.
    """
    days_inactive = (current_date - last_fight_date).days
    if days_inactive <= DECAY_THRESHOLD:
        return rating
    if rating <= BASE_RATING:
        return rating
    years_inactive = (days_inactive - DECAY_THRESHOLD) / 365.0
    decayed = rating + (BASE_RATING - rating) * (1 - (1 - DECAY_RATE) ** years_inactive)
    return decayed


def normalize_weight_class(wc: str) -> str:
    """Map raw weight class strings to canonical names."""
    if not isinstance(wc, str):
        return "Unknown"
    wc = wc.strip()
    for canonical in WEIGHT_CLASSES:
        if canonical.lower() in wc.lower():
            return canonical
    # Catch-all for title fights e.g. "UFC Lightweight Championship"
    for canonical in WEIGHT_CLASSES:
        word = canonical.split()[-1].lower()  # e.g. "lightweight"
        if word in wc.lower():
            return canonical
    return "Unknown"


# ─────────────────────────────────────────────
# MAIN COMPUTATION
# ─────────────────────────────────────────────

def compute_elo(db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Replay all fights chronologically and compute pre-fight Elo for each fighter.

    Returns a DataFrame with columns:
        fight_url, f1_elo, f2_elo, diff_elo, f1_elo_peak, f2_elo_peak
    """
    with get_db(db_path) as conn:
        fights = pd.read_sql(
            "SELECT fight_url, fighter_1, fighter_2, outcome, method, "
            "       event_date, weight_class "
            "FROM fights "
            "ORDER BY event_date ASC",
            conn
        )

    log.info(f"Computing Elo across {len(fights)} fights...")

    # ratings[weight_class][fighter_name] = current Elo
    ratings: dict[str, dict[str, float]] = {}
    # peak ratings
    peaks:   dict[str, dict[str, float]] = {}
    # last fight date per fighter (for decay)
    last_fight: dict[str, datetime] = {}

    elo_rows = []

    for _, row in fights.iterrows():
        f1      = row["fighter_1"]
        f2      = row["fighter_2"]
        outcome = row["outcome"]    # "win" means f1 won
        method  = row["method"]
        wc      = normalize_weight_class(row["weight_class"])
        
        try:
            fight_date = datetime.strptime(str(row["event_date"]), "%Y-%m-%d")
        except Exception:
            fight_date = datetime.now() # SHOULD NOT HAVE EXCEPTION TO BE DATETIME OF NOW

        # Init weight class pool if needed
        if wc not in ratings:
            ratings[wc] = {}
            peaks[wc]   = {}

        # Init fighters if first appearance
        for name in [f1, f2]:
            if name not in ratings[wc]:
                ratings[wc][name] = BASE_RATING
                peaks[wc][name]   = BASE_RATING

        # Apply inactivity decay before the fight
        for name in [f1, f2]:
            if name in last_fight:
                ratings[wc][name] = apply_decay(
                    ratings[wc][name], last_fight[name], fight_date
                )

        # Snapshot PRE-FIGHT ratings (what the model will see)
        r1_pre = ratings[wc][f1]
        r2_pre = ratings[wc][f2]

        elo_rows.append({
            "fight_url":    row["fight_url"],
            "f1_elo":       round(r1_pre, 2),
            "f2_elo":       round(r2_pre, 2),
            "diff_elo":     round(r1_pre - r2_pre, 2),
            "f1_elo_peak":  round(peaks[wc][f1], 2),
            "f2_elo_peak":  round(peaks[wc][f2], 2),
        })

        # Update ratings POST-fight
        # Skip non-competitive outcomes (DQ, overturned, NC, draw)
        method_str = str(method).strip()
        if outcome not in ("win", "loss") or method_str in NO_UPDATE_METHODS:
            last_fight[f1] = fight_date
            last_fight[f2] = fight_date
            continue

        winner = f1 if outcome == "win" else f2
        loser  = f2 if outcome == "win" else f1
        K      = k_factor(method)

        exp_winner = expected_score(ratings[wc][winner], ratings[wc][loser])
        delta      = K * (1.0 - exp_winner)

        ratings[wc][winner] += delta
        ratings[wc][loser]  -= delta

        # Update peaks
        for name in [f1, f2]:
            if ratings[wc][name] > peaks[wc][name]:
                peaks[wc][name] = ratings[wc][name]

        last_fight[f1] = fight_date
        last_fight[f2] = fight_date

    elo_df = pd.DataFrame(elo_rows)
    log.info(f"Elo computed for {len(elo_df)} fights")
    return elo_df


def write_elo_to_db(elo_df: pd.DataFrame, db_path: Path = DB_PATH):
    """Merge Elo values into the features table."""
    ensure_elo_columns(db_path)
    with get_db(db_path) as conn:
        for _, row in elo_df.iterrows():
            conn.execute("""
                UPDATE features
                SET f1_elo      = ?,
                    f2_elo      = ?,
                    diff_elo    = ?,
                    f1_elo_peak = ?,
                    f2_elo_peak = ?
                WHERE fight_url = ?
            """, (
                row["f1_elo"], row["f2_elo"], row["diff_elo"],
                row["f1_elo_peak"], row["f2_elo_peak"],
                row["fight_url"],
            ))
    log.info(f"Elo written to features table in {db_path}")


def run_elo_pipeline(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Full pipeline: compute Elo and write to DB."""
    elo_df = compute_elo(db_path)
    write_elo_to_db(elo_df, db_path)
    return elo_df 


# ───────────────────────────────────────────── 
# DIAGNOSTICS
# ─────────────────────────────────────────────

def get_top_rated(db_path: Path = DB_PATH, weight_class: str = None, n: int = 20) -> pd.DataFrame:
    """
    Return current top-rated fighters. Useful sanity check — 
    if Khabib and Jon Jones aren't near the top, something is wrong.
    """
    with get_db(db_path) as conn:
        fights = pd.read_sql(
            "SELECT fighter_1, fighter_2, outcome, method, event_date, weight_class "
            "FROM fights ORDER BY event_date ASC",
            conn
        )

    ratings: dict[str, dict[str, float]] = {}
    last_fight: dict[str, datetime] = {}

    for _, row in fights.iterrows():
        f1  = row["fighter_1"]
        f2  = row["fighter_2"]
        wc  = normalize_weight_class(row["weight_class"])
        outcome = row["outcome"]
        method  = row["method"]

        try:
            fight_date = datetime.strptime(str(row["event_date"]), "%Y-%m-%d")
        except Exception:
            fight_date = datetime.now()

        if wc not in ratings:
            ratings[wc] = {}
        for name in [f1, f2]:
            if name not in ratings[wc]:
                ratings[wc][name] = BASE_RATING
            if name in last_fight:
                ratings[wc][name] = apply_decay(ratings[wc][name], last_fight[name], fight_date)

        method_str = str(method).strip()
        if outcome in ("win", "loss") and method_str not in NO_UPDATE_METHODS:
            winner = f1 if outcome == "win" else f2
            loser  = f2 if outcome == "win" else f1
            K      = k_factor(method)
            exp    = expected_score(ratings[wc][winner], ratings[wc][loser])
            delta  = K * (1.0 - exp)
            ratings[wc][winner] += delta
            ratings[wc][loser]  -= delta

        last_fight[f1] = fight_date
        last_fight[f2] = fight_date

    rows = []
    for wc, fighters in ratings.items():
        for name, rating in fighters.items():
            rows.append({"weight_class": wc, "fighter": name, "elo": round(rating, 1)})

    df = pd.DataFrame(rows)
    if weight_class:
        df = df[df["weight_class"].str.lower() == weight_class.lower()]
    return df.sort_values("elo", ascending=False).groupby("weight_class").head(n).reset_index(drop=True)


def plot_rating_history(fighter_name: str, db_path: Path = DB_PATH):
    """Plot a fighter's Elo trajectory over their career."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.error("pip install matplotlib")
        return

    with get_db(db_path) as conn:
        fights = pd.read_sql(
            "SELECT fighter_1, fighter_2, outcome, method, event_date, weight_class "
            "FROM fights ORDER BY event_date ASC",
            conn
        )

    ratings: dict[str, dict[str, float]] = {}
    last_fight: dict[str, datetime] = {}
    history = []  # (date, elo) for the target fighter

    for _, row in fights.iterrows():
        f1, f2  = row["fighter_1"], row["fighter_2"]
        wc      = normalize_weight_class(row["weight_class"])
        outcome = row["outcome"]
        method  = row["method"]

        try:
            fight_date = datetime.strptime(str(row["event_date"]), "%Y-%m-%d")
        except Exception:
            continue

        if wc not in ratings:
            ratings[wc] = {}
        for name in [f1, f2]:
            if name not in ratings[wc]:
                ratings[wc][name] = BASE_RATING
            if name in last_fight:
                ratings[wc][name] = apply_decay(ratings[wc][name], last_fight[name], fight_date)

        # Capture pre-fight rating for the target fighter
        if fighter_name in (f1, f2):
            history.append((fight_date, ratings[wc][fighter_name]))

        method_str = str(method).strip()
        if outcome in ("win", "loss") and method_str not in NO_UPDATE_METHODS:
            winner = f1 if outcome == "win" else f2
            loser  = f2 if outcome == "win" else f1
            K      = k_factor(method)
            exp    = expected_score(ratings[wc][winner], ratings[wc][loser])
            delta  = K * (1.0 - exp)
            ratings[wc][winner] += delta
            ratings[wc][loser]  -= delta

        last_fight[f1] = fight_date
        last_fight[f2] = fight_date

    if not history:
        print(f"No fights found for '{fighter_name}'")
        return

    dates, elos = zip(*history)
    plt.figure(figsize=(12, 5))
    plt.plot(dates, elos, marker="o", linewidth=2, color="#e05c00")
    plt.axhline(BASE_RATING, linestyle="--", color="gray", alpha=0.5, label="Base rating")
    plt.title(f"Elo Rating History — {fighter_name}", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Elo Rating")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"data/{fighter_name.replace(' ', '_')}_elo.png", dpi=150)
    plt.show()
    print(f"Saved to data/{fighter_name.replace(' ', '_')}_elo.png")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    elo_df = run_elo_pipeline()

    # Sanity check — print top 10 per weight class
    print("\n-- Top rated fighters (sample) --")
    top = get_top_rated(n=5)
    print(top.to_string(index=False))

    if "--plot" in sys.argv:
        # Example: plot Khabib's career arc
        plot_rating_history("Khabib Nurmagomedov")
