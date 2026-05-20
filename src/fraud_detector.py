"""fraud_detector.py — Graph-based Fraud, Waste & Abuse (FWA) Detection.

Trains and evaluates a gradient-boosted classifier on graph-derived features
to identify fraudulent healthcare claims. Combines:
  - Claim-level features (billed amounts, CPT count, bill/allow ratio)
  - Provider graph features (PageRank, betweenness, referral concentration)
  - Statistical anomaly signals (z-score vs provider baseline)

This mirrors real payer FWA analytics pipelines where graph features
significantly outperform claim-level features alone.

Usage:
    python src/fraud_detector.py
    python src/fraud_detector.py --features results/fraud_features.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("fraud-detector")

FEATURE_COLS = [
    "billed_amount",
    "allowed_amount",
    "paid_amount",
    "bill_allow_ratio",
    "is_denied",
    "n_cpt_codes",
    "bill_amount_zscore",
    "provider_pagerank",
    "provider_betweenness",
    "provider_in_degree",
    "provider_out_degree",
    "provider_ref_concentration",
    "provider_is_fraud_ring",
]


# ─────────────────────────────────────────────────────────────────────
#  Simple gradient boosting (pure NumPy — no sklearn dependency)
#  For production you'd use sklearn GradientBoostingClassifier or XGBoost.
# ─────────────────────────────────────────────────────────────────────

class DecisionStump:
    """Single-feature threshold classifier (weak learner for boosting)."""
    def __init__(self):
        self.feature_idx = 0
        self.threshold   = 0.0
        self.polarity    = 1
        self.alpha       = 0.0

    def fit(self, X: np.ndarray, residuals: np.ndarray, weights: np.ndarray):
        n_samples, n_features = X.shape
        best_err = float("inf")
        for j in range(n_features):
            thresholds = np.unique(X[:, j])
            for t in thresholds:
                for p in [1, -1]:
                    preds = np.where(X[:, j] >= t, p, -p).astype(float)
                    err = weights @ (preds != np.sign(residuals)).astype(float)
                    if err < best_err:
                        best_err = self.feature_idx = j
                        self.threshold = t
                        self.polarity  = p
                        best_err       = err
        self.alpha = 0.5 * np.log((1 - best_err + 1e-10) / (best_err + 1e-10))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(X[:, self.feature_idx] >= self.threshold,
                        self.polarity, -self.polarity).astype(float)


class GradientFraudDetector:
    """
    Lightweight gradient-boosted tree ensemble for binary FWA classification.
    Uses AdaBoost-style weak learners for interpretability.

    For production use: replace with sklearn.ensemble.GradientBoostingClassifier
    or XGBoost with the same feature set.
    """

    def __init__(self, n_estimators: int = 30, learning_rate: float = 0.5):
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.stumps: list[DecisionStump] = []
        self.feature_cols: list[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> None:
        self.feature_cols = feature_names
        n = len(y)
        weights = np.ones(n) / n
        y_signed = np.where(y == 1, 1.0, -1.0)

        for i in range(self.n_estimators):
            stump = DecisionStump()
            stump.fit(X, y_signed, weights)
            preds  = stump.predict(X)
            wrong  = (preds != y_signed).astype(float)
            weights *= np.exp(stump.alpha * wrong)
            weights /= weights.sum()
            self.stumps.append(stump)

        log.info("Trained %d stumps", self.n_estimators)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return fraud probability for each sample."""
        scores = np.zeros(len(X))
        for stump in self.stumps:
            scores += stump.alpha * stump.predict(X)
        # Sigmoid transform to [0,1]
        return 1 / (1 + np.exp(-scores))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def feature_importance(self) -> dict[str, float]:
        """Approximate feature importance via |alpha| × usage count."""
        importance: dict[str, float] = {f: 0.0 for f in self.feature_cols}
        for stump in self.stumps:
            if stump.feature_idx < len(self.feature_cols):
                importance[self.feature_cols[stump.feature_idx]] += abs(stump.alpha)
        total = sum(importance.values()) or 1.0
        return {k: round(v / total, 4) for k, v in
                sorted(importance.items(), key=lambda x: -x[1])}


# ─────────────────────────────────────────────────────────────────────
#  Evaluation utilities
# ─────────────────────────────────────────────────────────────────────

def _confusion(y_true: np.ndarray, y_pred: np.ndarray):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return tp, fp, fn, tn

def precision_recall_f1(y_true, y_pred):
    tp, fp, fn, _ = _confusion(y_true, y_pred)
    prec = tp / (tp + fp + 1e-10)
    rec  = tp / (tp + fn + 1e-10)
    f1   = 2 * prec * rec / (prec + rec + 1e-10)
    return round(prec, 4), round(rec, 4), round(f1, 4)

def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Approximate AUC via Mann-Whitney U."""
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    count = sum(p > n for p in pos for n in neg)
    count += 0.5 * sum(p == n for p in pos for n in neg)
    return round(count / (len(pos) * len(neg)), 4)


# ─────────────────────────────────────────────────────────────────────
#  Main pipeline
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("results/fraud_features.json"))
    parser.add_argument("--out-dir",  type=Path, default=Path("results"))
    parser.add_argument("--threshold",type=float, default=0.45)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.features.exists():
        log.error("%s not found. Run graph_analytics.py first.", args.features)
        return

    with open(args.features) as f:
        records = json.load(f)

    # ── Prepare data ──────────────────────────────────────────
    X_list, y_list = [], []
    for r in records:
        row = [float(r.get(col, 0) or 0) for col in FEATURE_COLS]
        X_list.append(row)
        y_list.append(int(r["is_fraud"]))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    # Normalise
    col_means = X.mean(axis=0)
    col_stds  = np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)
    X_norm    = (X - col_means) / col_stds

    # Train / test split (80/20, stratified approximation)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    np.random.seed(42)
    np.random.shuffle(pos_idx); np.random.shuffle(neg_idx)

    train_idx = np.concatenate([
        pos_idx[:int(0.8*len(pos_idx))],
        neg_idx[:int(0.8*len(neg_idx))],
    ])
    test_idx = np.concatenate([
        pos_idx[int(0.8*len(pos_idx)):],
        neg_idx[int(0.8*len(neg_idx)):],
    ])
    np.random.shuffle(train_idx); np.random.shuffle(test_idx)

    X_train, y_train = X_norm[train_idx], y[train_idx]
    X_test,  y_test  = X_norm[test_idx],  y[test_idx]

    log.info("Train: %d samples (%d fraud), Test: %d samples (%d fraud)",
             len(y_train), y_train.sum(), len(y_test), y_test.sum())

    # ── Train ─────────────────────────────────────────────────
    model = GradientFraudDetector(n_estimators=30, learning_rate=0.5)
    model.fit(X_train, y_train, FEATURE_COLS)

    # ── Evaluate ──────────────────────────────────────────────
    proba_test = model.predict_proba(X_test)
    pred_test  = model.predict(X_test, threshold=args.threshold)

    prec, rec, f1 = precision_recall_f1(y_test, pred_test)
    auc = roc_auc(y_test, proba_test)
    tp, fp, fn, tn = _confusion(y_test, pred_test)

    print("\n" + "═"*60)
    print("  Healthcare FWA Detector — Evaluation Results")
    print("═"*60)
    print(f"  Threshold : {args.threshold}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    print("\n  Feature Importance (top 10):")
    for feat, imp in list(model.feature_importance().items())[:10]:
        bar = "█" * int(imp * 50)
        print(f"    {feat:35s} {imp:.4f} {bar}")

    # ── Save outputs ──────────────────────────────────────────
    report = {
        "model":     "GradientFraudDetector (AdaBoost stumps)",
        "threshold": args.threshold,
        "metrics":   {"precision": prec, "recall": rec, "f1": f1, "auc_roc": auc},
        "confusion_matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "feature_importance": model.feature_importance(),
        "n_train": len(y_train),
        "n_test":  len(y_test),
        "fraud_rate_train": round(float(y_train.mean()), 4),
        "fraud_rate_test":  round(float(y_test.mean()), 4),
    }
    out_path = args.out_dir / "fraud_model_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Saved model report → %s", out_path)

    # Top flagged claims
    test_records = [records[i] for i in test_idx]
    flagged = sorted(
        [(test_records[i], float(proba_test[i]))
         for i in range(len(test_idx)) if pred_test[i] == 1],
        key=lambda x: -x[1]
    )[:20]
    flagged_out = [{"claim_id": r["claim_id"],
                    "fraud_score": round(s, 4),
                    "true_label": r["is_fraud"],
                    "fraud_type": r.get("fraud_type", "none")}
                   for r, s in flagged]
    with open(args.out_dir / "top_flagged_claims.json", "w") as f:
        json.dump(flagged_out, f, indent=2)
    log.info("Saved top %d flagged claims", len(flagged_out))
    print(f"\n  ✅ Full report saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
