"""
Stroke Optimizer  (put this file in  src/components/stroke_optimizer.py)

WHY THIS EXISTS
---------------
Stroke data has only ~5% positive cases. A model that says "no stroke" for
everybody gets ~95% accuracy but 0% recall (this is what Random Forest, KNN and
AdaBoost did in your results). Two things fix this properly:

  1. Imbalance handling INSIDE cross-validation (class weights / SMOTE).
  2. THRESHOLD TUNING: models output a probability. The default cut-off 0.5 is
     wrong for rare events. We pick the cut-off that maximises F2 (recall
     counts 2x more than precision, because missing a stroke is worse than a
     false alarm). The threshold is chosen on TRAIN out-of-fold predictions
     only, then the TEST set is used once for the final honest score.

Run from the project root:
    python -m src.components.stroke_optimizer
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, fbeta_score,
    f1_score, precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

RESULTS_DIR = os.path.join("results", "stroke")
MODEL_PATH = os.path.join("models", "stroke", "best_model.pkl")
BETA = 2.0          # F2: recall is weighted 2x more than precision
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Small wrapper classes (so the saved model works with predict_pipeline.py
# without changing it: it only calls .predict() and .predict_proba())
# ---------------------------------------------------------------------------
class ProbAverager:
    """Soft-voting: averages predict_proba of several fitted models."""
    def __init__(self, models):
        self.models = models

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)


class ThresholdClassifier:
    """Fitted model + tuned decision threshold. predict() uses the threshold."""
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = float(threshold)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)


# ---------------------------------------------------------------------------
def build_candidates(pos_weight):
    return {
        "LogReg (balanced)": LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000),
        "LogReg + SMOTE": ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", LogisticRegression(C=0.1, max_iter=3000)),
        ]),
        "RandomForest (balanced)": RandomForestClassifier(
            n_estimators=400, max_depth=6, min_samples_leaf=10,
            class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost (scale_pos_weight)": XGBClassifier(
            n_estimators=250, max_depth=3, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=3, scale_pos_weight=pos_weight,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1),
        "HistGradBoost (balanced)": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
            class_weight="balanced", random_state=RANDOM_STATE),
    }


def best_threshold(y_true, proba, beta=BETA):
    """Threshold that maximises F-beta on the given (out-of-fold) predictions."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    prec, rec = prec[:-1], rec[:-1]
    f = (1 + beta**2) * prec * rec / np.maximum(beta**2 * prec + rec, 1e-12)
    i = int(np.argmax(f))
    return float(thr[i])


def test_metrics(y_true, proba, thr):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": round(thr, 4),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "f2": fbeta_score(y_true, pred, beta=BETA, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
    }


def run():
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)

        train_path, test_path = DataIngestion("stroke").initiate_data_ingestion()
        X_train, y_train, X_test, y_test = DataTransformation("stroke").initiate_data_transformation(
            train_path, test_path)
        X_train = X_train.toarray() if hasattr(X_train, "toarray") else np.asarray(X_train)
        X_test = X_test.toarray() if hasattr(X_test, "toarray") else np.asarray(X_test)
        y_train, y_test = np.asarray(y_train), np.asarray(y_test)

        pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        print(f"Train: {len(y_train)} rows, {int(y_train.sum())} stroke cases "
              f"({y_train.mean()*100:.1f}%).  Test: {len(y_test)} rows, {int(y_test.sum())} stroke cases.")

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        candidates = build_candidates(pos_weight)

        # ---- 1) out-of-fold probabilities on TRAIN (no test leakage) ----
        oof = {}
        for name, model in candidates.items():
            oof[name] = cross_val_predict(clone(model), X_train, y_train, cv=cv,
                                          method="predict_proba")[:, 1]
            print(f"  OOF done: {name}")
        # soft-voting ensemble of the three most different model families
        ens_members = ["LogReg (balanced)", "RandomForest (balanced)", "XGBoost (scale_pos_weight)"]
        oof["Soft Voting (LR+RF+XGB)"] = np.mean([oof[m] for m in ens_members], axis=0)

        # ---- 2) tune threshold on OOF, pick best model by OOF F2 ----
        rows, tuned = [], {}
        for name, p in oof.items():
            thr = best_threshold(y_train, p)
            tuned[name] = thr
            oof_f2 = fbeta_score(y_train, (p >= thr).astype(int), beta=BETA)
            rows.append({"model": name, "threshold": thr, "oof_f2": oof_f2,
                         "oof_roc_auc": roc_auc_score(y_train, p),
                         "oof_pr_auc": average_precision_score(y_train, p)})
        sel = pd.DataFrame(rows).sort_values("oof_f2", ascending=False)
        best_name = sel.iloc[0]["model"]

        # ---- 3) fit on full train, score on TEST once ----
        fitted, test_proba = {}, {}
        for name, model in candidates.items():
            m = clone(model).fit(X_train, y_train)
            fitted[name] = m
            test_proba[name] = m.predict_proba(X_test)[:, 1]
        fitted["Soft Voting (LR+RF+XGB)"] = ProbAverager([fitted[m] for m in ens_members])
        test_proba["Soft Voting (LR+RF+XGB)"] = np.mean([test_proba[m] for m in ens_members], axis=0)

        out = []
        for name in oof:
            r = {"model": name, **test_metrics(y_test, test_proba[name], tuned[name])}
            r["selected_best"] = (name == best_name)
            out.append(r)
        res = pd.DataFrame(out).sort_values("f2", ascending=False)
        res.to_csv(os.path.join(RESULTS_DIR, "optimized_comparison.csv"), index=False)

        pd.set_option("display.width", 220)
        print("\n=== TEST SET RESULTS (threshold tuned on train OOF) ===")
        print(res.round(3).to_string(index=False))
        print(f"\nBest model chosen on TRAIN cross-validation (not on test): {best_name}")

        # ---- 4) save best model + threshold ----
        final = ThresholdClassifier(fitted[best_name], tuned[best_name])
        if os.path.exists(MODEL_PATH):
            os.replace(MODEL_PATH, MODEL_PATH.replace(".pkl", "_old.pkl"))
        save_object(MODEL_PATH, final)

        # ---- 5) charts ----
        p = test_proba[best_name]
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
        pr, rc, th = precision_recall_curve(y_test, p)
        ax[0].plot(rc, pr); ax[0].axhline(y_test.mean(), ls="--", c="gray")
        ax[0].set(title=f"Precision-Recall (AP={average_precision_score(y_test, p):.3f})",
                  xlabel="Recall", ylabel="Precision")
        fpr, tpr, _ = roc_curve(y_test, p)
        ax[1].plot(fpr, tpr); ax[1].plot([0, 1], [0, 1], "k--", alpha=.4)
        ax[1].set(title=f"ROC (AUC={roc_auc_score(y_test, p):.3f})", xlabel="FPR", ylabel="TPR")
        ax[2].plot(th, pr[:-1], label="precision"); ax[2].plot(th, rc[:-1], label="recall")
        ax[2].axvline(tuned[best_name], c="red", ls="--", label=f"chosen thr={tuned[best_name]:.2f}")
        ax[2].set(title="Threshold trade-off", xlabel="threshold"); ax[2].legend()
        fig.suptitle(f"Stroke: {best_name}"); plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "optimized_curves.png"), dpi=130); plt.close()
        logging.info("[stroke] optimizer finished")
        return best_name, res
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run()
