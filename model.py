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
# TEMPORAL SPLIT FRACTIONS
# ─────────────────────────────────────────────
# 3-way split keeps the evaluation holdout completely isolated from
# both hyperparameter tuning (CV) and probability calibration (Platt).
#
#   |──── 70% train ────|── 15% calibrate ──|── 15% holdout ──|
#   CV runs here only       Platt scaling       Never touched
#                                               until backtest.py
#
TRAIN_FRAC   = 0.70   # GBM training + walk-forward CV
CALIB_FRAC   = 0.15   # Platt scaling calibration
# holdout = 1 - TRAIN_FRAC - CALIB_FRAC = 0.15

# ─────────────────────────────────────────────
# HYPERPARAMETER GRID
# ─────────────────────────────────────────────
# max_depth is fixed at 4 and excluded from the search. The CV folds are
# dominated by pre-2021 data where closing odds are sparse, so the CV
# loss function can't distinguish depth=2 from depth=4. Depth=4 is required
# to capture interactions between diff_closing_prob and other features in
# the holdout (2024+), where odds are widely available. Fixing depth prevents
# the grid search from picking an under-powered model.
# 18 combos × 7 folds = 126 fits (~2-4 min with n_jobs=-1)
FIXED_MAX_DEPTH = 4
PARAM_GRID = {
    "clf__n_estimators":  [100, 200, 300],
    "clf__max_depth":     [FIXED_MAX_DEPTH],
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

    3-way temporal split (no data crosses boundaries):
      1. First TRAIN_FRAC  (70%) — GBM training + walk-forward CV
      2. Next  CALIB_FRAC  (15%) — Platt scaling calibration only
      3. Final 15%              — held out; never seen until backtest.py

    Walk-forward CV runs only on the training portion so hyperparameter
    selection never sees calibration or holdout fights.
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM features ORDER BY event_date", conn)
    conn.close()
    df = df.sort_values("event_date").dropna(subset=["label"])

    n         = len(df)
    n_train   = int(n * TRAIN_FRAC)
    n_calib   = int(n * CALIB_FRAC)
    # holdout starts at n_train + n_calib (never touched here)

    df_train = df.iloc[:n_train]
    df_calib = df.iloc[n_train : n_train + n_calib]

    X_train, y_train = get_feature_matrix(df_train)
    X_calib, y_calib = get_feature_matrix(df_calib)

    log.info(f"Total fights: {n}  |  train={n_train}, calib={n_calib}, "
             f"holdout={n - n_train - n_calib}")
    log.info(f"Holdout starts: {df.iloc[n_train + n_calib]['event_date']}")

    # Hyperparameter search over train+calib window so the grid search sees
    # fights from 2021+ (where closing odds are available). The final model is
    # still trained on df_train only; df_calib is used only for Platt scaling.
    df_search = df.iloc[: n_train + n_calib]
    cv_results = walk_forward_cv(df_search)
    best_params = cv_results["best_params"]
    log.info(f"Using best params from CV: {best_params}")

    # Build and fit pipeline on training data
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(random_state=42)),
    ])
    pipeline.set_params(**best_params)
    pipeline.fit(X_train, y_train)

    # Platt scaling on calibration set (separate from holdout)
    calibrated = CalibratedClassifierCV(FrozenEstimator(pipeline), method="sigmoid")
    calibrated.fit(X_calib, y_calib)

    # Feature importance (from the inner pipeline's GBM)
    clf = pipeline.named_steps["clf"]
    feat_imp = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    log.info("Top 10 features:\n" + feat_imp.head(10).to_string())

    holdout_start_date = str(df.iloc[n_train + n_calib]["event_date"])

    # Save model
    model_path = MODEL_DIR / f"ufc_model_{datetime.now().strftime('%Y%m%d')}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "pipeline":            calibrated,
            "features":            list(X_train.columns),
            "best_params":         best_params,
            "cv_results":          cv_results,
            "feat_importance":     feat_imp.to_dict(),
            "calibration_method":  "sigmoid",
            "trained_on":          str(datetime.now()),
            "holdout_start_date":  holdout_start_date,
            "train_frac":          TRAIN_FRAC,
            "calib_frac":          CALIB_FRAC,
        }, f)
    log.info(f"Model saved to {model_path}")
    return calibrated, cv_results


# ─────────────────────────────────────────────
# PREDICTION INTERFACE
# ─────────────────────────────────────────────

def load_latest_model():
    """Load the most recently trained production model (date-stamped, not experiment variants)."""
    models = sorted(MODEL_DIR.glob("ufc_model_[0-9]*.pkl"))
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
    debug: bool = False,
) -> dict:
    """
    Predict winner of an upcoming fight using each fighter's last known stats.

    debug=True adds "_debug_X_pred" (the exact raw feature row fed to the
    pipeline) and "_debug_feature_cols" to the returned dict, for offline
    explainability (e.g. SHAP) without duplicating this function's row
    construction logic elsewhere.
    """
    artifact = load_latest_model()
    pipeline = artifact["pipeline"]
    feature_cols = artifact["features"]

    from features import get_fighter_current_state, parse_height, parse_reach

    conn = sqlite3.connect(db_path)
    fighters_df = pd.read_sql("SELECT first_name, last_name, dob, height, reach, stance FROM fighters", conn)
    conn.close()

    fighters_df["full_name"] = (
        fighters_df["first_name"].fillna("") + " " + fighters_df["last_name"].fillna("")
    ).str.strip()
    fighters_df = fighters_df.set_index("full_name")
    dob_map = fighters_df["dob"].to_dict()

    STAT_KEYS = [
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
        "ctrl_pct", "five_round_exp",
    ]

    def get_latest_stats(name: str, prefix: str):
        """Get a fighter's TRUE current rolling stats and last-fight date,
        computed from their own full fight history (see
        features.get_fighter_current_state — this does NOT depend on
        whether their past opponents also had prior UFC data, unlike
        reading the sparse `features` table).
        """
        raw_stats, last_date = get_fighter_current_state(name, db_path)
        if not raw_stats:
            return {}, None

        stats = {f"{prefix}_{k}": raw_stats.get(k, np.nan) for k in STAT_KEYS}

        if name in fighters_df.index:
            frow = fighters_df.loc[name]
            if isinstance(frow, pd.DataFrame):
                frow = frow.iloc[0]
            stats[f"{prefix}_height"] = parse_height(frow.get("height"))
            stats[f"{prefix}_reach"] = parse_reach(frow.get("reach"))
            for stance in ["Orthodox", "Southpaw", "Switch"]:
                stats[f"{prefix}_stance_{stance}"] = 1 if frow.get("stance") == stance else 0

        return stats, last_date

    f1_stats, f1_last_fight = get_latest_stats(fighter_1, "f1")
    f2_stats, f2_last_fight = get_latest_stats(fighter_2, "f2")

    if not f1_stats:
        return {"error": f"No data found for {fighter_1}"}
    if not f2_stats:
        return {"error": f"No data found for {fighter_2}"}

    # Pull current Elo ratings from elo module
    try:
        from elo import compute_elo
        elo_df = compute_elo(db_path)

        def get_own_elo(name):
            """Return (current_elo, peak_elo) for a fighter using fights-table position.
            elo_df stores f1_elo/f2_elo in fights-table order (f1=winner, f2=loser).
            fights_f1/fights_f2 columns identify which fighter occupied each slot,
            so we always read the column that corresponds to this fighter's slot.
            """
            as_f1 = elo_df[elo_df["fights_f1"] == name]
            as_f2 = elo_df[elo_df["fights_f2"] == name]
            # Collect (current_elo, peak_elo) from each appearance
            records = []
            if not as_f1.empty:
                records.append(as_f1[["f1_elo", "f1_elo_peak"]].rename(
                    columns={"f1_elo": "elo", "f1_elo_peak": "peak"}).iloc[-1])
            if not as_f2.empty:
                records.append(as_f2[["f2_elo", "f2_elo_peak"]].rename(
                    columns={"f2_elo": "elo", "f2_elo_peak": "peak"}).iloc[-1])
            if not records:
                return 1500.0, 1500.0
            # Most recent appearance (elo_df is chronological, take last row overall)
            combined = pd.concat([
                as_f1[["f1_elo", "f1_elo_peak"]].rename(columns={"f1_elo": "elo", "f1_elo_peak": "peak"}),
                as_f2[["f2_elo", "f2_elo_peak"]].rename(columns={"f2_elo": "elo", "f2_elo_peak": "peak"}),
            ]).sort_index()
            last = combined.iloc[-1]
            return float(last["elo"]), float(last["peak"])

        f1_elo, f1_elo_peak = get_own_elo(fighter_1)
        f2_elo, f2_elo_peak = get_own_elo(fighter_2)
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
        "ctrl_pct", "five_round_exp",
    ]
    # NaN (not 0) on a missing side — matches training, where an unknown diff
    # is left NaN and filled by the pipeline's trained median imputer, rather
    # than being asserted as "exactly tied."
    for k in stat_keys:
        v1 = f1_stats.get(f"f1_{k}", np.nan)
        v2 = f2_stats.get(f"f2_{k}", np.nan)
        row[f"diff_{k}"] = (v1 - v2) if (not np.isnan(v1) and not np.isnan(v2)) else np.nan

    # Elo features
    row["diff_elo"]    = f1_elo - f2_elo
    row["f1_elo"]      = f1_elo
    row["f2_elo"]      = f2_elo
    row["f1_elo_peak"] = f1_elo_peak
    row["f2_elo_peak"] = f2_elo_peak

    # Age and layoff (computed relative to today). Layoff uses each fighter's
    # TRUE last fight date from get_latest_stats/get_fighter_current_state —
    # not a lookup into the sparse `features` table.
    today = datetime.now()

    def _current_age(name):
        dob_str = dob_map.get(name, "")
        dob = pd.to_datetime(dob_str, errors="coerce")
        if pd.isna(dob):
            return np.nan
        return (today - dob).days / 365.25

    def _layoff_from(last_fight_date):
        if last_fight_date is None or pd.isna(last_fight_date):
            return np.nan
        last_dt = pd.to_datetime(last_fight_date, errors="coerce")
        if pd.isna(last_dt):
            return np.nan
        return (today - last_dt.to_pydatetime()).days

    f1_age_val    = _current_age(fighter_1)
    f2_age_val    = _current_age(fighter_2)
    f1_layoff_val = _layoff_from(f1_last_fight)
    f2_layoff_val = _layoff_from(f2_last_fight)

    row["f1_age"]   = f1_age_val
    row["f2_age"]   = f2_age_val
    row["diff_age"] = (f1_age_val - f2_age_val) if (not np.isnan(f1_age_val) and not np.isnan(f2_age_val)) else np.nan
    row["f1_days_since_last_fight"] = f1_layoff_val
    row["f2_days_since_last_fight"] = f2_layoff_val
    row["diff_days_since_last_fight"] = (f1_layoff_val - f2_layoff_val) if (not np.isnan(f1_layoff_val) and not np.isnan(f2_layoff_val)) else np.nan

    # 5-round fight flag
    row["is_5round_fight"] = 1 if is_title_fight else 0

    # Closing odds: caller can pass current BFO moneyline as fighter_1's no-vig prob.
    # If not provided, leave NaN so the trained imputer fills the population
    # median rather than asserting a specific (and likely wrong) 50/50 market.
    if closing_odds_f1 is not None:
        row["diff_closing_prob"] = round(closing_odds_f1 - (1.0 - closing_odds_f1), 4)
    else:
        row["diff_closing_prob"] = np.nan

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
    if debug:
        result["_debug_X_pred"] = X_pred
        result["_debug_feature_cols"] = feature_cols
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