"""
ufc_predictor/features.py
Builds ML-ready feature vectors from raw scraped fight/fighter data.

Each row = one fight, from the perspective of fighter_1.
Label: 1 if fighter_1 won, 0 if fighter_2 won.

Features use ONLY stats from BEFORE the fight (rolling historical averages)
to prevent data leakage.
"""

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
import logging

log = logging.getLogger(__name__)
DATA_DIR = Path("data")
DB_PATH  = DATA_DIR / "ufc.db"


# ─────────────────────────────────────────────
# PARSING HELPERS
# ─────────────────────────────────────────────

def parse_fraction(s: str) -> tuple[float, float]:
    """'45 of 102' → (45, 102). Returns (nan, nan) on failure."""
    try:
        parts = str(s).split(" of ")
        return float(parts[0]), float(parts[1])
    except Exception:
        return np.nan, np.nan


def parse_pct(s: str) -> float:
    """'67%' → 0.67"""
    try:
        return float(str(s).replace("%", "")) / 100
    except Exception:
        return np.nan


def parse_ctrl(s: str) -> float:
    """'4:32' → 272 (seconds)"""
    try:
        m, sec = str(s).split(":")
        return float(m) * 60 + float(sec)
    except Exception:
        return np.nan


def parse_height(s: str) -> float:
    """'6' 1"' → 73 (inches)"""
    try:
        s = str(s).replace('"', '').strip()
        ft, inch = s.split("' ")
        return float(ft) * 12 + float(inch)
    except Exception:
        return np.nan


def parse_reach(s: str) -> float:
    """'74.0"' → 74.0"""
    try:
        return float(str(s).replace('"', '').strip())
    except Exception:
        return np.nan


# ─────────────────────────────────────────────
# FIGHT-LEVEL FEATURE EXTRACTION
# ─────────────────────────────────────────────

def extract_fight_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Parse raw string columns into numeric fight stats."""
    out = df.copy()

    # Significant strikes
    for fighter in ["f1", "f2"]:
        col = f"tot_sig_str_{fighter}"
        if col in out.columns:
            out[[f"sig_str_landed_{fighter}", f"sig_str_att_{fighter}"]] = out[col].apply(
                lambda x: pd.Series(parse_fraction(x))
            )

    # Takedowns
    for fighter in ["f1", "f2"]:
        col = f"tot_td_{fighter}"
        if col in out.columns:
            out[[f"td_landed_{fighter}", f"td_att_{fighter}"]] = out[col].apply(
                lambda x: pd.Series(parse_fraction(x))
            )

    # Control time
    for fighter in ["f1", "f2"]:
        col = f"tot_ctrl_{fighter}"
        if col in out.columns:
            out[f"ctrl_secs_{fighter}"] = out[col].apply(parse_ctrl)

    # Knockdowns
    for fighter in ["f1", "f2"]:
        col = f"tot_kd_{fighter}"
        if col in out.columns:
            out[f"kd_{fighter}"] = pd.to_numeric(out[col], errors="coerce")

    # Submission attempts
    for fighter in ["f1", "f2"]:
        col = f"tot_sub_att_{fighter}"
        if col in out.columns:
            out[f"sub_att_{fighter}"] = pd.to_numeric(out[col], errors="coerce")

    # Significant strike targets (head, body, leg) and positions (distance, clinch, ground)
    for fighter in ["f1", "f2"]:
        for category in ["head", "body", "leg", "distance", "clinch", "ground"]:
            col = f"sig_{category}_{fighter}"
            if col in out.columns:
                out[[f"{category}_landed_{fighter}", f"{category}_att_{fighter}"]] = out[col].apply(
                    lambda x: pd.Series(parse_fraction(x))
                )

    return out


# ─────────────────────────────────────────────
# PER-FIGHTER ROLLING STATS (prevents leakage)
# ─────────────────────────────────────────────

STAT_COLS = [
    "sig_str_landed", "sig_str_att", "td_landed", "td_att",
    "ctrl_secs", "kd", "sub_att",
    "head_landed", "head_att", "body_landed", "body_att",
    "leg_landed", "leg_att", "distance_landed", "distance_att",
    "clinch_landed", "clinch_att", "ground_landed", "ground_att",
]


def compute_fighter_history(fights: pd.DataFrame) -> dict:
    """
    Returns a dict: fighter_name → list of per-fight stats in chronological order.
    Used to compute rolling averages BEFORE each fight.
    """
    history = {}

    # We need long format: one row per fighter per fight
    for _, row in fights.iterrows():
        for side, opp in [("f1", "f2"), ("f2", "f1")]:
            name = row.get(f"fighter_{side[-1]}")  # fighter_1 or fighter_2
            if pd.isna(name) or name == "":
                continue

            entry = {
                "date":    row.get("event_date", ""),
                "won":     1 if (side == "f1" and row.get("outcome") == "win") or
                               (side == "f2" and row.get("outcome") == "loss") else 0,
                "method":  row.get("method", ""),
                "round":   row.get("round", np.nan),
            }
            for stat in STAT_COLS:
                entry[stat] = row.get(f"{stat}_{side}", np.nan)

            history.setdefault(name, []).append(entry)

    # Sort each fighter's fights by date
    for name in history:
        history[name].sort(key=lambda x: x["date"])

    return history


def _round_fraction(entry: dict) -> float:
    """Normalize round ended to fraction of scheduled rounds.

    Uses the actual scheduled_rounds from the fight page when available.
    Fallback: infer from round number and method.
    """
    try:
        r = float(entry["round"])
    except (ValueError, TypeError):
        return np.nan

    # Use scraped scheduled_rounds if available
    sched = entry.get("scheduled_rounds")
    try:
        sched = float(sched)
        if sched > 0 and not np.isnan(sched):
            return r / sched
    except (ValueError, TypeError):
        pass

    # Fallback inference for older data without scheduled_rounds
    method = str(entry.get("method", ""))
    if r > 3:
        scheduled = 5          # round 4 or 5 → must be a 5-round fight
    elif "Decision" in method:
        scheduled = r          # decision always ends in the last round
    else:
        scheduled = 3          # finish in rounds 1-3 → assume 3-rounder
    return r / scheduled


def rolling_avg(history_list: list[dict], n_fights: int = 7) -> dict:
    """Compute rolling average of stats over last n_fights."""
    if not history_list:
        return {}
    recent = history_list[-n_fights:]
    result = {
        "win_rate":    np.mean([h["won"] for h in recent]),
        "fights_count": len(history_list),
        "finish_rate": np.mean([1 if h["method"] in ("KO/TKO", "Submission") else 0
                                for h in recent]),
        "avg_round_ended": np.nanmean([_round_fraction(h) for h in recent]),
    }
    for stat in STAT_COLS:
        vals = [h.get(stat, np.nan) for h in recent]
        result[f"avg_{stat}"] = np.nanmean(vals)

    # Accuracy rates
    if result.get("avg_sig_str_att", 0) > 0:
        result["sig_str_acc"] = result["avg_sig_str_landed"] / result["avg_sig_str_att"]
    if result.get("avg_td_att", 0) > 0:
        result["td_acc"] = result["avg_td_landed"] / result["avg_td_att"]

    # Target accuracy (head, body, leg)
    for target in ["head", "body", "leg"]:
        att = result.get(f"avg_{target}_att", 0)
        if att > 0:
            result[f"{target}_acc"] = result[f"avg_{target}_landed"] / att

    # Position distribution (fraction of landed sig strikes at distance/clinch/ground)
    total_landed = result.get("avg_sig_str_landed", 0)
    if total_landed > 0:
        for pos in ["distance", "clinch", "ground"]:
            result[f"{pos}_pct"] = result.get(f"avg_{pos}_landed", 0) / total_landed
    
    # Control time percentage (ctrl_secs / estimated total fight time)
    # Each round is 5 min (300s). We round up by assuming every round was full.
    ctrl_pcts = []
    for h in recent:
        ctrl = h.get("ctrl_secs", np.nan)
        try:
            rnd = float(h.get("round", np.nan))
            if not np.isnan(ctrl) and not np.isnan(rnd) and rnd > 0:
                ctrl_pcts.append(ctrl / (rnd * 300))
        except (ValueError, TypeError):
            pass
    if ctrl_pcts:
        result["ctrl_pct"] = np.nanmean(ctrl_pcts)

    # 5-round fight experience (title fights / main events)
    result["five_round_exp"] = np.mean([
        1 if float(h.get("scheduled_rounds") or 3) == 5 else 0
        for h in recent
    ])

    return result


# ─────────────────────────────────────────────
# BUILD ML DATAFRAME
# ─────────────────────────────────────────────

def build_feature_dataframe(
    db_path: Path = DB_PATH,
    n_fights: int = 7,
    write_db: bool = True,
) -> pd.DataFrame:
    """
    Main function. Reads from ufc.db and returns a DataFrame where each row is a fight with:
    - Pre-fight rolling stats for both fighters (differenced: f1 - f2)
    - Physical attributes (height, reach, stance)
    - Target label: 1 = fighter_1 wins

    Fighter order is randomly swapped (50/50) to balance labels.
    ufcstats.com always lists the winner first, so without this the dataset
    would be ~98% label=1, making the model useless.
    """
    conn = sqlite3.connect(db_path)
    fights_raw  = pd.read_sql("SELECT * FROM fights ORDER BY event_date", conn)
    fighters_raw = pd.read_sql("SELECT * FROM fighters", conn)
    conn.close()

    log.info(f"Building features from {len(fights_raw)} fights and {len(fighters_raw)} fighters")

    fights = extract_fight_stats(fights_raw)
    fights["event_date"] = pd.to_datetime(fights["event_date"], errors="coerce")
    fights = fights.sort_values("event_date").reset_index(drop=True)

    # Reproducible random swap to balance labels
    rng = np.random.RandomState(42)

    history = {}
    last_fight_date: dict = {}
    feature_rows = []

    # DOB lookup for age feature
    dob_lookup: dict = {}
    for _, fr in fighters_raw.iterrows():
        fname = (str(fr.get("first_name") or "") + " " + str(fr.get("last_name") or "")).strip()
        dob = pd.to_datetime(fr.get("dob", ""), errors="coerce")
        if fname and pd.notna(dob):
            dob_lookup[fname] = dob

    for idx, row in fights.iterrows():
        f1 = row.get("fighter_1", "")
        f2 = row.get("fighter_2", "")
        outcome = row.get("outcome", "")

        if not f1 or not f2 or pd.isna(f1) or pd.isna(f2):
            continue

        fight_dt = row["event_date"]

        lfd_f1 = last_fight_date.get(f1)
        layoff_f1 = (fight_dt - lfd_f1).days if lfd_f1 is not None and pd.notna(fight_dt) else np.nan
        lfd_f2 = last_fight_date.get(f2)
        layoff_f2 = (fight_dt - lfd_f2).days if lfd_f2 is not None and pd.notna(fight_dt) else np.nan

        dob_f1 = dob_lookup.get(f1)
        age_f1 = (fight_dt - dob_f1).days / 365.25 if dob_f1 is not None and pd.notna(fight_dt) else np.nan
        dob_f2 = dob_lookup.get(f2)
        age_f2 = (fight_dt - dob_f2).days / 365.25 if dob_f2 is not None and pd.notna(fight_dt) else np.nan

        f1_stats = rolling_avg(history.get(f1, []), n_fights)
        f2_stats = rolling_avg(history.get(f2, []), n_fights)

        if f1_stats and f2_stats:
            # Random swap: 50% chance to flip fighter order
            swap = rng.random() < 0.5
            if swap:
                feat_f1, feat_f2 = f2, f1
                feat_f1_stats, feat_f2_stats = f2_stats, f1_stats
                label = 0 if outcome == "win" else 1
            else:
                feat_f1, feat_f2 = f1, f2
                feat_f1_stats, feat_f2_stats = f1_stats, f2_stats
                label = 1 if outcome == "win" else 0

            feature_row = {
                "fight_url":    row.get("fight_url", ""),
                "event_date":   str(row.get("event_date", "")),
                "fighter_1":    feat_f1,
                "fighter_2":    feat_f2,
                "weight_class": row.get("weight_class", ""), 
                "label":        label,
            }

            all_stat_keys = set(list(feat_f1_stats.keys()) + list(feat_f2_stats.keys()))
            for key in all_stat_keys:
                v1 = feat_f1_stats.get(key, np.nan)
                v2 = feat_f2_stats.get(key, np.nan)
                feature_row[f"f1_{key}"]   = v1
                feature_row[f"f2_{key}"]   = v2
                feature_row[f"diff_{key}"] = (v1 - v2) if (not np.isnan(v1) and not np.isnan(v2)) else np.nan

            # 5-round fight flag (known pre-fight from the schedule)
            try:
                is_5r = 1 if float(row.get("scheduled_rounds") or 3) == 5 else 0
            except (ValueError, TypeError):
                is_5r = 0
            feature_row["is_5round_fight"] = is_5r

            # Layoff and age (swap-aware)
            lo1 = layoff_f2 if swap else layoff_f1
            lo2 = layoff_f1 if swap else layoff_f2
            ag1 = age_f2 if swap else age_f1
            ag2 = age_f1 if swap else age_f2
            feature_row["f1_days_since_last_fight"] = lo1
            feature_row["f2_days_since_last_fight"] = lo2
            feature_row["diff_days_since_last_fight"] = (lo1 - lo2) if (not np.isnan(lo1) and not np.isnan(lo2)) else np.nan
            feature_row["f1_age"] = ag1
            feature_row["f2_age"] = ag2
            feature_row["diff_age"] = (ag1 - ag2) if (not np.isnan(ag1) and not np.isnan(ag2)) else np.nan

            feature_rows.append(feature_row)

        # Update history AFTER building features for this fight (always use original order)
        for side, won in [("f1", outcome == "win"), ("f2", outcome == "loss")]:
            name = f1 if side == "f1" else f2
            entry = {
                "date":   str(row.get("event_date", "")),
                "won":    int(won),
                "method": row.get("method", ""),
                "round":  row.get("round", np.nan),
                "scheduled_rounds": row.get("scheduled_rounds", np.nan),
            }
            for stat in STAT_COLS:
                suffix = "f1" if side == "f1" else "f2"
                entry[stat] = row.get(f"{stat}_{suffix}", np.nan)
            history.setdefault(name, []).append(entry)

        if pd.notna(fight_dt):
            last_fight_date[f1] = fight_dt
            last_fight_date[f2] = fight_dt

    df = pd.DataFrame(feature_rows)
    log.info(f"Feature dataframe shape: {df.shape}")

    # ── Merge physical attributes ──
    fighters_raw["full_name"] = (
        fighters_raw["first_name"].fillna("") + " " + fighters_raw["last_name"].fillna("")
    ).str.strip()
    fighters_raw["height_in"] = fighters_raw["height"].apply(parse_height)
    fighters_raw["reach_in"]  = fighters_raw["reach"].apply(parse_reach)

    phys = fighters_raw[["full_name", "height_in", "reach_in", "stance"]].drop_duplicates("full_name")

    df = df.merge(phys.rename(columns={
        "full_name": "fighter_1", "height_in": "f1_height",
        "reach_in": "f1_reach",  "stance": "f1_stance"
    }), on="fighter_1", how="left")

    df = df.merge(phys.rename(columns={
        "full_name": "fighter_2", "height_in": "f2_height",
        "reach_in": "f2_reach",  "stance": "f2_stance"
    }), on="fighter_2", how="left")

    df["diff_height"] = df["f1_height"] - df["f2_height"]
    df["diff_reach"]  = df["f1_reach"]  - df["f2_reach"]

    for stance in ["Orthodox", "Southpaw", "Switch"]:
        df[f"f1_stance_{stance}"] = (df["f1_stance"] == stance).astype(int)
        df[f"f2_stance_{stance}"] = (df["f2_stance"] == stance).astype(int)

    # ── Write features back to DB ──
    if write_db:
        conn = sqlite3.connect(db_path)
        df.to_sql("features", conn, if_exists="replace", index=False)
        conn.close()
        log.info(f"Features saved to {db_path} (features table)")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = build_feature_dataframe()
    print(df.head())
    print(df.describe())
