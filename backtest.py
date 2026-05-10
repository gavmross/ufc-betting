import sqlite3, pickle, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import log_loss, roc_auc_score

models = sorted(Path("models").glob("ufc_model_*.pkl"))
with open(models[-1], "rb") as f:
    artifact = pickle.load(f)

pipeline    = artifact["pipeline"]
feature_cols = artifact["features"]

conn = sqlite3.connect("data/ufc.db")
df = pd.read_sql("SELECT * FROM features ORDER BY event_date", conn)
conn.close()

df = df.sort_values("event_date").dropna(subset=["label"])

split   = int(len(df) * 0.8)
holdout = df.iloc[split:].copy()

X = holdout[[c for c in feature_cols if c in holdout.columns]].fillna(0)
for col in feature_cols:
    if col not in X.columns:
        X[col] = 0
X = X[feature_cols]

proba = pipeline.predict_proba(X)
holdout["prob_f1"]   = proba[:, 1]
holdout["pred_label"] = (holdout["prob_f1"] > 0.5).astype(int)
holdout["correct"]   = (holdout["pred_label"] == holdout["label"]).astype(int)
holdout["confidence"] = holdout["prob_f1"].apply(lambda p: max(p, 1 - p))

print("=" * 65)
print(f"BACKTEST  --  {models[-1].name}")
print("=" * 65)
print(f"Holdout fights : {len(holdout)}")
print(f"Date range     : {str(holdout['event_date'].min())[:10]}  to  {str(holdout['event_date'].max())[:10]}")
print(f"Pick accuracy  : {holdout['correct'].mean():.1%}")
print(f"AUC            : {roc_auc_score(holdout['label'], holdout['prob_f1']):.4f}")
print(f"Log loss       : {log_loss(holdout['label'], holdout['prob_f1']):.4f}")
print()

print("Accuracy by confidence bucket:")
print(f"  {'Bucket':<10} {'Fights':>7} {'Accuracy':>10} {'Avg Conf':>10}")
bins   = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 1.01]
labels = ["50-55%", "55-60%", "60-65%", "65-70%", "70-75%", "75%+"]
holdout["bucket"] = pd.cut(holdout["confidence"], bins=bins, labels=labels)
for b in labels:
    sub = holdout[holdout["bucket"] == b]
    if len(sub) == 0:
        continue
    print(f"  {b:<10} {len(sub):>7} {sub['correct'].mean():>9.1%} {sub['confidence'].mean():>9.1%}")

print()
print("Last 25 fights (most recent first):")
print(f"  {'Date':<12} {'Fighter 1':<22} {'Fighter 2':<22} {'Pred':>4} {'Conf':>6}  {'Result':>6}")
for _, r in holdout.tail(25).iloc[::-1].iterrows():
    pred  = "F1" if r["pred_label"] == 1 else "F2"
    ok    = "OK  " if r["correct"] else "MISS"
    date  = str(r["event_date"])[:10]
    f1    = str(r["fighter_1"])[:21]
    f2    = str(r["fighter_2"])[:21]
    print(f"  {date:<12} {f1:<22} {f2:<22} {pred:>4} {r['confidence']:>5.1%}  {ok:>6}")
