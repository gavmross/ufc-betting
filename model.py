"""
ufc_predictor/model.py
Trains a gradient boosted classifier on UFC fight features.
Includes walk-forward cross-validation (no leakage), hyperparameter tuning
via grid search, probability calibration, and prediction interface.
"""

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
import pickle
import logging
from datetime import datetime

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.base import clone
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

log = logging.getLogger(__name__)
DATA_DIR = Path("data")
DB_PATH  = DATA_DIR / "ufc.db"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# HYPERPARAMETER GRID
# ─────────────────────────────────────────────
# 54 combos × 7 folds = 378 fits (~5-10 min with n_jobs=-1)
PARAM_GRID = {
    "clf__n_estimators":  [100, 200, 300],
    "clf__max_depth":     [2, 3, 4],
    "clf__learning_rate": [0.01, 0.05, 0.1],
    "clf__subsample":     [0.8, 1.0],
}


# ─────────────────────────────────────────────
# FEATURE SELECTION
# ─────────────────────────────────────────────

# Differential features (f1 - f2) are most informative — avoids scale bias
DIFF_FEATURES = [
    "diff_win_rate",
    "diff_finish_rate",
    "diff_avg_sig_str_landed",
    "diff_avg_sig_str_att",
    "diff_avg_td_landed",
    "diff_avg_td_att",
    "diff_avg_ctrl_secs",
    "diff_avg_kd",
    "diff_avg_sub_att",
    "diff_sig_str_acc",
    "diff_td_acc",
    "diff_avg_round_ended",
    "diff_fights_count",
    "diff_height",
    "diff_reach",
    # Elo — encodes opponent quality / strength of schedule
    "diff_elo",
    # Strike target accuracy (head, body, leg)
    "diff_head_acc",
    "diff_body_acc",
    "diff_leg_acc",
    # Strike position distribution (where they fight from)
    "diff_distance_pct",
    "diff_clinch_pct",
    "diff_ground_pct",
    # Strike target volume
    "diff_avg_head_landed",
    "diff_avg_body_landed",
    "diff_avg_leg_landed",
    # Strike position volume
    "diff_avg_distance_landed",
    "diff_avg_clinch_landed",
    "diff_avg_ground_landed",
    # Control time as fraction of fight time
    "diff_ctrl_pct",
    # Layoff and age
    "diff_days_since_last_fight",
    "diff_age",
    # 5-round experience
    "diff_five_round_exp",
    # Closing-line market consensus (NaN → 0 = "50/50, no info" for older fights)
    "diff_closing_prob",
]

STANCE_FEATURES = [
    "f1_stance_Orthodox", "f1_stance_Southpaw", "f1_stance_Switch",
    "f2_stance_Orthodox", "f2_stance_Southpaw", "f2_stance_Switch",
]

ELO_FEATURES = [
    "f1_elo", "f2_elo",
    "f1_elo_peak", "f2_elo_peak",
]

CONTEXT_FEATURES = [
    "f1_age", "f2_age",
    "f1_days_since_last_fight", "f2_days_since_last_fight",
    "is_5round_fight",
]

ALL_FEATURES = DIFF_FEATURES + STANCE_FEATURES + ELO_FEATURES + CONTEXT_FEATURES


def get_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    available = [c for c in ALL_FEATURES if c in df.columns]
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        log.warning(f"Missing feature columns (will be zeroed): {missing}")
    X = df[available].fillna(0)
    y = df["label"]
    return X, y


# ─────────────────────────────────────────────
# WALK-FORWARD CROSS VALIDATION
# ─────────────────────────────────────────────

def walk_forward_cv(df: pd.DataFrame, n_splits: int = 7) -> dict:
    """
    Temporal cross-validation with hyperparameter tuning and calibration.

    1. Build temporal fold indices (train on past, test on future).
    2. Run GridSearchCV over PARAM_GRID using those folds to find best params.
    3. Re-evaluate per fold with best params, reporting both raw and
       calibrated (Platt scaling) log loss.

    Never use random train/test split — it causes leakage.
    """
    df = df.sort_values("event_date").dropna(subset=["label"])
    X, y = get_feature_matrix(df)

    n = len(df)
    fold_size = n // (n_splits + 1)

    # Build temporal fold indices for GridSearchCV
    folds = []
    for i in range(1, n_splits + 1):
        train_end = fold_size * i
        test_end = min(train_end + fold_size, n)
        train_idx = list(range(train_end))
        test_idx = list(range(train_end, test_end))
        folds.append((train_idx, test_idx))

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(random_state=42)),
    ])

    # ── Grid search across temporal folds ──
    log.info(f"Running grid search: {len(PARAM_GRID['clf__n_estimators'])*len(PARAM_GRID['clf__max_depth'])*len(PARAM_GRID['clf__learning_rate'])*len(PARAM_GRID['clf__subsample'])} param combos × {n_splits} folds")
    gs = GridSearchCV(
        pipeline, PARAM_GRID,
        cv=folds,
        scoring="neg_log_loss",
        refit=False,
        n_jobs=-1,
    )
    gs.fit(X, y)

    best_params = gs.best_params_
    log.info(f"Best params: {best_params}")
    log.info(f"Best CV neg_log_loss: {gs.best_score_:.4f}")

    # ── Per-fold evaluation with calibration ──
    scores = []
    for i, (train_idx, test_idx) in enumerate(folds):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        if len(y_train.unique()) < 2 or len(y_test) == 0:
            continue

        # Split training data temporally: first 80% train, last 20% calibration
        calib_start = int(len(train_idx) * 0.8)
        X_train_proper = X_train.iloc[:calib_start]
        y_train_proper = y_train.iloc[:calib_start]
        X_calib = X_train.iloc[calib_start:]
        y_calib = y_train.iloc[calib_start:]

        # Train pipeline with best params
        pipe = clone(pipeline).set_params(**best_params)
        pipe.fit(X_train_proper, y_train_proper)

        # Raw probabilities
        raw_probs = pipe.predict_proba(X_test)[:, 1]

        # Calibrated probabilities (Platt scaling on temporal holdout)
        calibrated = CalibratedClassifierCV(FrozenEstimator(pipe), method="sigmoid")
        calibrated.fit(X_calib, y_calib)
        cal_probs = calibrated.predict_proba(X_test)[:, 1]

        ll_raw = log_loss(y_test, raw_probs)
        ll_cal = log_loss(y_test, cal_probs)
        fold_score = {
            "fold": i + 1,
            "train_size": calib_start,
            "calib_size": len(y_calib),
            "test_size": len(y_test),
            "accuracy": accuracy_score(y_test, (cal_probs > 0.5).astype(int)),
            "auc": roc_auc_score(y_test, cal_probs),
            "log_loss_raw": ll_raw,
            "log_loss_calibrated": ll_cal,
            "prob_acc_raw": np.exp(-ll_raw),
            "prob_acc_calibrated": np.exp(-ll_cal),
        }
        scores.append(fold_score)
        log.info(
            f"Fold {i+1}: acc={fold_score['accuracy']:.3f}  "
            f"auc={fold_score['auc']:.3f}  "
            f"prob_acc_raw={fold_score['prob_acc_raw']:.1%}  "
            f"prob_acc_cal={fold_score['prob_acc_calibrated']:.1%}"
        )

    mean_ll_raw = np.mean([s["log_loss_raw"] for s in scores])
    mean_ll_cal = np.mean([s["log_loss_calibrated"] for s in scores])
    summary = {
        "folds": scores,
        "best_params": best_params,
        "mean_accuracy": np.mean([s["accuracy"] for s in scores]),
        "mean_auc": np.mean([s["auc"] for s in scores]),
        "mean_log_loss_raw": mean_ll_raw,
        "mean_log_loss_calibrated": mean_ll_cal,
        "mean_prob_acc_raw": np.exp(-mean_ll_raw),
        "mean_prob_acc_calibrated": np.exp(-mean_ll_cal),
    }
    log.info(
        f"CV Summary: acc={summary['mean_accuracy']:.3f}  "
        f"auc={summary['mean_auc']:.3f}  "
        f"prob_acc_raw={summary['mean_prob_acc_raw']:.1%}  "
        f"prob_acc_cal={summary['mean_prob_acc_calibrated']:.1%}"
    )
    return summary


# ─────────────────────────────────────────────
# TRAIN FINAL MODEL
# ─────────────────────────────────────────────

def train(db_path: Path = DB_PATH) -> object:
    """
    Train final model with tuned hyperparameters and probability calibration.

    1. Load features, run walk-forward CV (includes grid search).
    2. Build pipeline with best params from CV.
    3. Temporal split: first 80% for training, last 20% for calibration.
    4. Wrap trained pipeline in CalibratedClassifierCV (Platt scaling).
    5. Save calibrated model artifact.
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM features ORDER BY event_date", conn)
    conn.close()
    df = df.sort_values("event_date").dropna(subset=["label"])
    X, y = get_feature_matrix(df)

    log.info(f"Training on {len(df)} fights, {X.shape[1]} features")

    # Run walk-forward CV with grid search
    cv_results = walk_forward_cv(df)
    best_params = cv_results["best_params"]
    log.info(f"Using best params from CV: {best_params}")

    # Build pipeline with tuned params
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(random_state=42)),
    ])
    pipeline.set_params(**best_params)

    # Temporal split: 80% train, 20% calibration
    split = int(len(df) * 0.8)
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_calib, y_calib = X.iloc[split:], y.iloc[split:]
    log.info(f"Train split: {len(y_train)} fights, Calibration split: {len(y_calib)} fights")

    pipeline.fit(X_train, y_train)

    # Calibrate on temporal holdout (Platt scaling)
    calibrated = CalibratedClassifierCV(FrozenEstimator(pipeline), method="sigmoid")
    calibrated.fit(X_calib, y_calib)

    # Feature importance (from the inner pipeline's GBM)
    clf = pipeline.named_steps["clf"]
    feat_imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    log.info("Top 10 features:\n" + feat_imp.head(10).to_string())

    # Save model
    model_path = MODEL_DIR / f"ufc_model_{datetime.now().strftime('%Y%m%d')}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "pipeline": calibrated,
            "features": list(X.columns),
            "best_params": best_params,
            "cv_results": cv_results,
            "feat_importance": feat_imp.to_dict(),
            "calibration_method": "sigmoid",
            "trained_on": str(datetime.now()),
        }, f)
    log.info(f"Model saved to {model_path}")
    return calibrated, cv_results


# ─────────────────────────────────────────────
# PREDICTION INTERFACE
# ─────────────────────────────────────────────

def load_latest_model():
    """Load the most recently trained model."""
    models = sorted(MODEL_DIR.glob("ufc_model_*.pkl"))
    if not models:
        raise FileNotFoundError("No trained model found. Run train() first.")
    with open(models[-1], "rb") as f:
        return pickle.load(f)


def predict_fight(
    fighter_1: str,
    fighter_2: str,
    db_path: Path = DB_PATH,
    is_title_fight: bool = False,
    closing_odds_f1: float | None = None,
) -> dict:
    """
    Predict winner of an upcoming fight using each fighter's last known stats.
    """
    artifact = load_latest_model()
    pipeline = artifact["pipeline"]
    feature_cols = artifact["features"]

    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM features ORDER BY event_date", conn)
    fighters_df = pd.read_sql("SELECT first_name, last_name, dob FROM fighters", conn)
    conn.close()

    fighters_df["full_name"] = (
        fighters_df["first_name"].fillna("") + " " + fighters_df["last_name"].fillna("")
    ).str.strip()
    dob_map = fighters_df.set_index("full_name")["dob"].to_dict()

    def get_latest_stats(name: str, prefix: str) -> dict:
        """Get most recent pre-fight stats for a fighter."""
        as_f1 = df[df["fighter_1"] == name].sort_values("event_date").tail(1)
        as_f2 = df[df["fighter_2"] == name].sort_values("event_date").tail(1)
        row = as_f1.iloc[-1] if not as_f1.empty else (as_f2.iloc[-1] if not as_f2.empty else None)
        if row is None:
            return {}

        stat_keys = [
            "win_rate", "finish_rate", "avg_sig_str_landed", "avg_sig_str_att",
            "avg_td_landed", "avg_td_att", "avg_ctrl_secs", "avg_kd",
            "avg_sub_att", "sig_str_acc", "td_acc", "avg_round_ended", "fights_count",
            # strike targets
            "head_acc", "body_acc", "leg_acc",
            "avg_head_landed", "avg_body_landed", "avg_leg_landed",
            # strike positions
            "distance_pct", "clinch_pct", "ground_pct",
            "avg_distance_landed", "avg_clinch_landed", "avg_ground_landed",
            # control time fraction
            "ctrl_pct",
        ]
        source_prefix = "f1" if not as_f1.empty else "f2"
        stats = {}
        for k in stat_keys:
            col = f"{source_prefix}_{k}"
            stats[f"{prefix}_{k}"] = row.get(col, np.nan)

        # Physical
        for attr in ["height", "reach"]:
            stats[f"{prefix}_{attr}"] = row.get(f"{source_prefix}_{attr}", np.nan)

        for stance in ["Orthodox", "Southpaw", "Switch"]:
            stats[f"{prefix}_stance_{stance}"] = row.get(f"{source_prefix}_stance_{stance}", 0)

        return stats

    f1_stats = get_latest_stats(fighter_1, "f1")
    f2_stats = get_latest_stats(fighter_2, "f2")

    if not f1_stats:
        return {"error": f"No data found for {fighter_1}"}
    if not f2_stats:
        return {"error": f"No data found for {fighter_2}"}

    # Pull current Elo ratings from elo module
    try:
        from elo import compute_elo
        elo_df = compute_elo(db_path)
        # Get most recent Elo snapshot for each fighter
        def latest_elo(name, col):
            fights_as_f1 = elo_df[elo_df.get("fight_url", pd.Series()).isin(
                df[df["fighter_1"] == name]["fight_url"]
            )]
            fights_as_f2 = elo_df[elo_df.get("fight_url", pd.Series()).isin(
                df[df["fighter_2"] == name]["fight_url"]
            )]
            combined = pd.concat([fights_as_f1, fights_as_f2])
            return combined[col].iloc[-1] if not combined.empty else 1500.0

        f1_elo      = latest_elo(fighter_1, "f1_elo")
        f2_elo      = latest_elo(fighter_2, "f2_elo")
        f1_elo_peak = latest_elo(fighter_1, "f1_elo_peak")
        f2_elo_peak = latest_elo(fighter_2, "f2_elo_peak")
    except Exception as e:
        log.warning(f"Could not load Elo ratings: {e}. Defaulting to 1500.")
        f1_elo = f2_elo = f1_elo_peak = f2_elo_peak = 1500.0

    # Build differential features
    row = {}
    stat_keys = [
        "win_rate", "finish_rate", "avg_sig_str_landed", "avg_sig_str_att",
        "avg_td_landed", "avg_td_att", "avg_ctrl_secs", "avg_kd",
        "avg_sub_att", "sig_str_acc", "td_acc", "avg_round_ended", "fights_count",
        "height", "reach",
        # strike targets
        "head_acc", "body_acc", "leg_acc",
        "avg_head_landed", "avg_body_landed", "avg_leg_landed",
        # strike positions
        "distance_pct", "clinch_pct", "ground_pct",
        "avg_distance_landed", "avg_clinch_landed", "avg_ground_landed",
        # control time fraction
        "ctrl_pct",
    ]
    for k in stat_keys:
        v1 = f1_stats.get(f"f1_{k}", np.nan)
        v2 = f2_stats.get(f"f2_{k}", np.nan)
        row[f"diff_{k}"] = (v1 - v2) if (not np.isnan(v1) and not np.isnan(v2)) else 0

    # Elo features
    row["diff_elo"]    = f1_elo - f2_elo
    row["f1_elo"]      = f1_elo
    row["f2_elo"]      = f2_elo
    row["f1_elo_peak"] = f1_elo_peak
    row["f2_elo_peak"] = f2_elo_peak

    # Age and layoff (computed relative to today)
    today = datetime.now()

    def _current_age(name):
        dob_str = dob_map.get(name, "")
        dob = pd.to_datetime(dob_str, errors="coerce")
        if pd.isna(dob):
            return np.nan
        return (today - dob).days / 365.25

    def _current_layoff(name):
        as_f1 = df[df["fighter_1"] == name]["event_date"]
        as_f2 = df[df["fighter_2"] == name]["event_date"]
        all_dates = pd.concat([as_f1, as_f2])
        if all_dates.empty:
            return np.nan
        last_dt = pd.to_datetime(all_dates.max(), errors="coerce")
        if pd.isna(last_dt):
            return np.nan
        return (today - last_dt.to_pydatetime()).days

    f1_age_val    = _current_age(fighter_1)
    f2_age_val    = _current_age(fighter_2)
    f1_layoff_val = _current_layoff(fighter_1)
    f2_layoff_val = _current_layoff(fighter_2)

    row["f1_age"]   = f1_age_val
    row["f2_age"]   = f2_age_val
    row["diff_age"] = (f1_age_val - f2_age_val) if (not np.isnan(f1_age_val) and not np.isnan(f2_age_val)) else 0
    row["f1_days_since_last_fight"] = f1_layoff_val
    row["f2_days_since_last_fight"] = f2_layoff_val
    row["diff_days_since_last_fight"] = (f1_layoff_val - f2_layoff_val) if (not np.isnan(f1_layoff_val) and not np.isnan(f2_layoff_val)) else 0

    # 5-round fight flag
    row["is_5round_fight"] = 1 if is_title_fight else 0

    # Closing odds: caller can pass current BFO moneyline as fighter_1's no-vig prob
    # If not provided, defaults to 0 (= 50/50, no market information available)
    if closing_odds_f1 is not None:
        row["diff_closing_prob"] = round(closing_odds_f1 - (1.0 - closing_odds_f1), 4)
    else:
        row["diff_closing_prob"] = 0

    # 5-round experience differential (from latest feature row)
    def _5round_exp(name):
        as_f1 = df[df["fighter_1"] == name].sort_values("event_date")
        as_f2 = df[df["fighter_2"] == name].sort_values("event_date")
        row_src = as_f1.tail(1) if not as_f1.empty else as_f2.tail(1)
        if row_src.empty:
            return np.nan
        prefix = "f1" if not as_f1.empty else "f2"
        return row_src.iloc[0].get(f"{prefix}_five_round_exp", np.nan)

    f1_5r = _5round_exp(fighter_1)
    f2_5r = _5round_exp(fighter_2)
    row["diff_five_round_exp"] = (f1_5r - f2_5r) if (not np.isnan(f1_5r) and not np.isnan(f2_5r)) else 0

    for stance in ["Orthodox", "Southpaw", "Switch"]:
        row[f"f1_stance_{stance}"] = f1_stats.get(f"f1_stance_{stance}", 0)
        row[f"f2_stance_{stance}"] = f2_stats.get(f"f2_stance_{stance}", 0)

    X_pred = pd.DataFrame([row])
    for col in feature_cols:
        if col not in X_pred.columns:
            X_pred[col] = 0
    X_pred = X_pred[feature_cols]

    proba = pipeline.predict_proba(X_pred)[0]

    # Build human-readable stat profile per fighter (rates/pcts only)
    def _pct(val):
        """Format a 0-1 ratio as a rounded percentage, or None if missing."""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return round(float(val) * 100, 1)

    def _rnd(val, decimals=1):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return round(float(val), decimals)

    def build_profile(stats, prefix):
        return {
            "record": {
                "win_rate":    _pct(stats.get(f"{prefix}_win_rate")),
                "finish_rate": _pct(stats.get(f"{prefix}_finish_rate")),
                "fights":      _rnd(stats.get(f"{prefix}_fights_count"), 0),
            },
            "striking": {
                "sig_str_acc":  _pct(stats.get(f"{prefix}_sig_str_acc")),
                "head_acc":     _pct(stats.get(f"{prefix}_head_acc")),
                "body_acc":     _pct(stats.get(f"{prefix}_body_acc")),
                "leg_acc":      _pct(stats.get(f"{prefix}_leg_acc")),
            },
            "striking_position": {
                "distance_pct": _pct(stats.get(f"{prefix}_distance_pct")),
                "clinch_pct":   _pct(stats.get(f"{prefix}_clinch_pct")),
                "ground_pct":   _pct(stats.get(f"{prefix}_ground_pct")),
            },
            "grappling": {
                "td_acc":   _pct(stats.get(f"{prefix}_td_acc")),
                "ctrl_pct": _pct(stats.get(f"{prefix}_ctrl_pct")),
            },
            "physical": {
                "height_inches": _rnd(stats.get(f"{prefix}_height")),
                "reach_inches":  _rnd(stats.get(f"{prefix}_reach")),
            },
        }

    result = {
        "fighter_1": fighter_1,
        "fighter_2": fighter_2,
        "fighter_1_win_prob": round(float(proba[1]), 3),
        "fighter_2_win_prob": round(float(proba[0]), 3),
        "predicted_winner": fighter_1 if proba[1] > 0.5 else fighter_2,
        "confidence": round(float(max(proba)), 3),
        "elo": {
            fighter_1: {"current": round(f1_elo, 1), "peak": round(f1_elo_peak, 1)},
            fighter_2: {"current": round(f2_elo, 1), "peak": round(f2_elo_peak, 1)},
        },
        "f1_age":   round(f1_age_val, 1) if not np.isnan(f1_age_val) else None,
        "f2_age":   round(f2_age_val, 1) if not np.isnan(f2_age_val) else None,
        "f1_days_since_last_fight": int(f1_layoff_val) if not np.isnan(f1_layoff_val) else None,
        "f2_days_since_last_fight": int(f2_layoff_val) if not np.isnan(f2_layoff_val) else None,
        "f1_profile": build_profile(f1_stats, "f1"),
        "f2_profile": build_profile(f2_stats, "f2"),
    }
    if closing_odds_f1 is not None:
        result["market_consensus"] = {
            fighter_1: round(closing_odds_f1, 3),
            fighter_2: round(1.0 - closing_odds_f1, 3),
        }
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model, cv = train()

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 60)
    print(f"Best params: {cv['best_params']}")
    print()
    print(f"  Pick accuracy:        {cv['mean_accuracy']:.1%}")
    print(f"  AUC:                  {cv['mean_auc']:.3f}")
    print(f"  Prob accuracy (raw):  {cv['mean_prob_acc_raw']:.1%}")
    print(f"  Prob accuracy (cal):  {cv['mean_prob_acc_calibrated']:.1%}")
    print(f"  Log loss (raw):       {cv['mean_log_loss_raw']:.4f}")
    print(f"  Log loss (calibrated):{cv['mean_log_loss_calibrated']:.4f}")
    print()
    print("  (Prob accuracy = avg probability assigned to the correct outcome.")
    print("   Random guessing = 50%. Perfect model = 100%.)")
    print()
    print("Per-fold results:")
    for fold in cv["folds"]:
        print(
            f"  Fold {fold['fold']}: "
            f"acc={fold['accuracy']:.1%}  "
            f"auc={fold['auc']:.3f}  "
            f"prob_acc_raw={fold['prob_acc_raw']:.1%}  "
            f"prob_acc_cal={fold['prob_acc_calibrated']:.1%}"
        )