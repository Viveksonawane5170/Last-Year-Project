"""
Shared utility functions used by every component, across all 3 modules
(mental_health, stroke, diabetes). Keeping these generic means we don't
duplicate save/load/evaluate logic 3 times.
"""

import os
import sys
import dill  # handles saving sklearn Pipelines/vectorizers as well as pickle does
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    """Save any python object (model, vectorizer, preprocessor) to disk."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as f:
            dill.dump(obj, f)
        logging.info(f"Object saved at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """Load a previously saved object (model, vectorizer, preprocessor)."""
    try:
        with open(file_path, "rb") as f:
            return dill.load(f)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_classification_models(models: dict, X_train, y_train, X_test, y_test, average="binary"):
    """
    Trains each model in `models` dict, evaluates it on the test set,
    and returns a report dict: { model_name: {metric: value, ...}, ... }

    `average` should be "binary" for stroke/diabetes (2 classes) and
    "macro" or "weighted" for mental_health (multi-class).
    """
    try:
        report = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average=average, zero_division=0),
                "recall": recall_score(y_test, y_pred, average=average, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, average=average, zero_division=0),
            }

            # ROC-AUC only makes sense for binary classification with predict_proba
            if average == "binary" and hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X_test)[:, 1]
                    metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
                except Exception:
                    metrics["roc_auc"] = None
            else:
                metrics["roc_auc"] = None

            metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
            metrics["fitted_model"] = model

            report[name] = metrics
            logging.info(f"Trained {name}: accuracy={metrics['accuracy']:.4f}, f1={metrics['f1_score']:.4f}")

        return report

    except Exception as e:
        raise CustomException(e, sys)


def get_best_model(report: dict, primary_metric: str = "f1_score"):
    """
    Given the report dict from evaluate_classification_models, returns
    (best_model_name, best_model_object, best_score).
    """
    try:
        best_name = max(report, key=lambda name: report[name][primary_metric])
        best_score = report[best_name][primary_metric]
        best_model = report[best_name]["fitted_model"]
        return best_name, best_model, best_score
    except Exception as e:
        raise CustomException(e, sys)
