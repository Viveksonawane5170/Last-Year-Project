"""
Model Trainer Component

This file:
1. Creates candidate classification models
2. Performs 5-Fold Cross Validation
3. Performs hyperparameter tuning
4. Evaluates models on the untouched test set
5. Selects the best model using F1-score
6. Saves the best trained model

Works for:
    - diabetes
    - stroke
    - mental_health
"""

import os
import sys
import json
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    RandomizedSearchCV
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


# ============================================================
# MODULE CONFIGURATION
# ============================================================

MODEL_TRAINER_CONFIG = {

    "mental_health": {
        "model_path": os.path.join(
            "models",
            "mental_health",
            "best_model.pkl"
        ),
        "average": "macro",
        "primary_metric": "f1_score",
        "use_multinomial_nb": True,
    },

    "stroke": {
        "model_path": os.path.join(
            "models",
            "stroke",
            "best_model.pkl"
        ),
        "average": "binary",
        "primary_metric": "f1_score",
        "use_multinomial_nb": False,
    },

    "diabetes": {
        "model_path": os.path.join(
            "models",
            "diabetes",
            "best_model.pkl"
        ),
        "average": "binary",
        "primary_metric": "f1_score",
        "use_multinomial_nb": False,
    },
}


# ============================================================
# MODEL CREATION
# ============================================================

def get_model_dict(module_name: str, config: dict):

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        ),

        "SVM": SVC(
            probability=True,
            random_state=42,
            kernel="linear" if module_name == "mental_health" else "rbf"
        ),

        "KNN": KNeighborsClassifier(),

        "Naive Bayes":
            MultinomialNB()
            if config["use_multinomial_nb"]
            else GaussianNB(),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        ),

        "XGBoost": XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1
        ),
    }

    return models


# ============================================================
# HYPERPARAMETER SEARCH SPACE
# ============================================================

def get_param_distributions(module_name: str):

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    logistic_params = {
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ["liblinear", "lbfgs"],
        "class_weight": [None, "balanced"]
    }

    # --------------------------------------------------------
    # Decision Tree
    # --------------------------------------------------------

    decision_tree_params = {
        "max_depth": [None, 3, 5, 7, 10, 15],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "class_weight": [None, "balanced"]
    }

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    random_forest_params = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
        "class_weight": [
            None,
            "balanced",
            "balanced_subsample"
        ]
    }

    # --------------------------------------------------------
    # SVM
    # --------------------------------------------------------

    svm_params = {
        "C": [0.01, 0.1, 1, 10, 100],
        "gamma": ["scale", "auto"],
        "class_weight": [None, "balanced"]
    }

    # --------------------------------------------------------
    # KNN
    # --------------------------------------------------------

    knn_params = {
        "n_neighbors": [3, 5, 7, 9, 11, 15],
        "weights": ["uniform", "distance"],
        "p": [1, 2]
    }

    # --------------------------------------------------------
    # Gradient Boosting
    # --------------------------------------------------------

    gradient_boosting_params = {
        "n_estimators": [100, 150, 200, 300],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "max_depth": [2, 3, 4, 5],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "subsample": [0.8, 1.0]
    }

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    xgboost_params = {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [2, 3, 4, 5],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.2]
    }

    return {
        "Logistic Regression": logistic_params,
        "Decision Tree": decision_tree_params,
        "Random Forest": random_forest_params,
        "SVM": svm_params,
        "KNN": knn_params,
        "Gradient Boosting": gradient_boosting_params,
        "XGBoost": xgboost_params
    }


# ============================================================
# DATA CONVERSION
# ============================================================

def convert_to_dense(X):

    """
    Some models require dense numpy arrays.

    If X is a sparse matrix, convert it to dense.
    """

    if hasattr(X, "toarray"):
        return X.toarray()

    return np.asarray(X)


# ============================================================
# CROSS VALIDATION
# ============================================================

def perform_cross_validation(
    model,
    X_train,
    y_train,
    average="binary"
):

    """
    Perform 5-Fold Stratified Cross Validation.

    We calculate:
        Accuracy
        Precision
        Recall
        F1
        ROC-AUC
    """

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        error_score="raise"
    )

    cv_results = {
        "cv_accuracy": float(
            np.mean(results["test_accuracy"])
        ),

        "cv_precision": float(
            np.mean(results["test_precision"])
        ),

        "cv_recall": float(
            np.mean(results["test_recall"])
        ),

        "cv_f1": float(
            np.mean(results["test_f1"])
        ),

        "cv_roc_auc": float(
            np.mean(results["test_roc_auc"])
        )
    }

    return cv_results


# ============================================================
# TEST SET EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    average="binary"
):

    """
    Evaluate final model on untouched test data.
    """

    predictions = model.predict(X_test)

    # --------------------------------------------------------
    # Probability / ROC-AUC
    # --------------------------------------------------------

    roc_auc = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X_test)

        if probabilities.shape[1] == 2:
            roc_auc = roc_auc_score(
                y_test,
                probabilities[:, 1]
            )

    elif hasattr(model, "decision_function"):

        decision_scores = model.decision_function(X_test)

        roc_auc = roc_auc_score(
            y_test,
            decision_scores
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average=average,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average=average,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average=average,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "confusion_matrix": cm.tolist(),
        "fitted_model": model
    }


# ============================================================
# HYPERPARAMETER TUNING
# ============================================================

def tune_model(
    model_name,
    model,
    X_train,
    y_train,
    param_distributions
):

    """
    Randomized hyperparameter search using 5-fold CV.

    F1-score is used because this is a medical risk prediction
    project and accuracy alone is not enough.
    """

    if model_name not in param_distributions:

        logging.info(
            f"[{model_name}] No hyperparameter search configured."
        )

        model.fit(
            X_train,
            y_train
        )

        return model, None

    logging.info(
        f"[{model_name}] Starting hyperparameter tuning..."
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions[model_name],
        n_iter=20,
        scoring="f1",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=0,
        refit=True
    )

    search.fit(
        X_train,
        y_train
    )

    logging.info(
        f"[{model_name}] Best CV F1 = "
        f"{search.best_score_:.4f}"
    )

    logging.info(
        f"[{model_name}] Best Parameters = "
        f"{search.best_params_}"
    )

    return (
        search.best_estimator_,
        float(search.best_score_)
    )


# ============================================================
# MAIN MODEL TRAINING CLASS
# ============================================================

class ModelTrainer:

    def __init__(self, module_name: str):

        if module_name not in MODEL_TRAINER_CONFIG:

            raise ValueError(
                f"Unknown module_name '{module_name}'"
            )

        self.module_name = module_name

        self.config = MODEL_TRAINER_CONFIG[
            module_name
        ]

    # --------------------------------------------------------
    # TRAINING ENTRYPOINT
    # --------------------------------------------------------

    def initiate_model_training(
        self,
        X_train,
        y_train,
        X_test,
        y_test
    ):

        try:

            logging.info(
                f"================================================"
            )

            logging.info(
                f"[{self.module_name}] MODEL TRAINING STARTED"
            )

            logging.info(
                f"================================================"
            )

            # ------------------------------------------------
            # Convert sparse matrix to dense
            # ------------------------------------------------

            if self.module_name != "mental_health":

                X_train = convert_to_dense(X_train)
                X_test = convert_to_dense(X_test)

            # ------------------------------------------------
            # Get models
            # ------------------------------------------------

            models = get_model_dict(
                self.module_name,
                self.config
            )

            param_distributions = get_param_distributions(
                self.module_name
            )

            all_results = {}

            best_model = None
            best_model_name = None
            best_score = -np.inf

            # ------------------------------------------------
            # Train every model
            # ------------------------------------------------

            for model_name, model in models.items():

                logging.info(
                    f"------------------------------------------------"
                )

                logging.info(
                    f"[{self.module_name}] Processing: "
                    f"{model_name}"
                )

                # ==================================================
                # STEP 1 — BASELINE CROSS VALIDATION
                # ==================================================

                try:

                    cv_result = perform_cross_validation(
                        model,
                        X_train,
                        y_train,
                        average=self.config["average"]
                    )

                    logging.info(
                        f"[{model_name}] "
                        f"Baseline CV F1 = "
                        f"{cv_result['cv_f1']:.4f}"
                    )

                except Exception as e:

                    logging.warning(
                        f"[{model_name}] "
                        f"Baseline CV failed: {str(e)}"
                    )

                    cv_result = {
                        "cv_accuracy": None,
                        "cv_precision": None,
                        "cv_recall": None,
                        "cv_f1": None,
                        "cv_roc_auc": None
                    }

                # ==================================================
                # STEP 2 — HYPERPARAMETER TUNING
                # ==================================================

                try:

                    tuned_model, tuned_cv_f1 = tune_model(
                        model_name,
                        model,
                        X_train,
                        y_train,
                        param_distributions
                    )

                except Exception as e:

                    logging.warning(
                        f"[{model_name}] "
                        f"Hyperparameter tuning failed."
                    )

                    logging.warning(
                        str(e)
                    )

                    # Fallback to normal model

                    tuned_model = model

                    tuned_model.fit(
                        X_train,
                        y_train
                    )

                    tuned_cv_f1 = None

                # ==================================================
                # STEP 3 — TEST SET EVALUATION
                # ==================================================

                test_result = evaluate_model(
                    tuned_model,
                    X_test,
                    y_test,
                    average=self.config["average"]
                )

                # ------------------------------------------------
                # Combine results
                # ------------------------------------------------

                result = {

                    "cv_accuracy": cv_result[
                        "cv_accuracy"
                    ],

                    "cv_precision": cv_result[
                        "cv_precision"
                    ],

                    "cv_recall": cv_result[
                        "cv_recall"
                    ],

                    "cv_f1": cv_result[
                        "cv_f1"
                    ],

                    "cv_roc_auc": cv_result[
                        "cv_roc_auc"
                    ],

                    "tuned_cv_f1": tuned_cv_f1,

                    "accuracy": test_result[
                        "accuracy"
                    ],

                    "precision": test_result[
                        "precision"
                    ],

                    "recall": test_result[
                        "recall"
                    ],

                    "f1_score": test_result[
                        "f1_score"
                    ],

                    "roc_auc": test_result[
                        "roc_auc"
                    ],

                    "confusion_matrix":
                        test_result[
                            "confusion_matrix"
                        ],

                    "fitted_model":
                        test_result[
                            "fitted_model"
                        ]
                }

                all_results[
                    model_name
                ] = result

                # ------------------------------------------------
                # Print results
                # ------------------------------------------------

                logging.info(
                    f"[{model_name}] "
                    f"Test Accuracy = "
                    f"{result['accuracy']:.4f}"
                )

                logging.info(
                    f"[{model_name}] "
                    f"Test Precision = "
                    f"{result['precision']:.4f}"
                )

                logging.info(
                    f"[{model_name}] "
                    f"Test Recall = "
                    f"{result['recall']:.4f}"
                )

                logging.info(
                    f"[{model_name}] "
                    f"Test F1 = "
                    f"{result['f1_score']:.4f}"
                )

                if result["roc_auc"] is not None:

                    logging.info(
                        f"[{model_name}] "
                        f"Test ROC-AUC = "
                        f"{result['roc_auc']:.4f}"
                    )

                # ==================================================
                # BEST MODEL SELECTION
                # ==================================================

                current_score = result[
                    self.config["primary_metric"]
                ]

                if current_score > best_score:

                    best_score = current_score

                    best_model = tuned_model

                    best_model_name = model_name

            # ====================================================
            # SAVE BEST MODEL
            # ====================================================

            logging.info(
                f"================================================"
            )

            logging.info(
                f"[{self.module_name}] "
                f"BEST MODEL = {best_model_name}"
            )

            logging.info(
                f"[{self.module_name}] "
                f"BEST F1 = {best_score:.4f}"
            )

            logging.info(
                f"================================================"
            )

            save_object(
                self.config["model_path"],
                best_model
            )

            logging.info(
                f"[{self.module_name}] "
                f"Best model saved at: "
                f"{self.config['model_path']}"
            )

            # ====================================================
            # CLEAN REPORT
            # ====================================================

            clean_report = {}

            for model_name, metrics in all_results.items():

                clean_report[
                    model_name
                ] = {

                    "cv_accuracy":
                        metrics["cv_accuracy"],

                    "cv_precision":
                        metrics["cv_precision"],

                    "cv_recall":
                        metrics["cv_recall"],

                    "cv_f1":
                        metrics["cv_f1"],

                    "cv_roc_auc":
                        metrics["cv_roc_auc"],

                    "tuned_cv_f1":
                        metrics["tuned_cv_f1"],

                    "accuracy":
                        metrics["accuracy"],

                    "precision":
                        metrics["precision"],

                    "recall":
                        metrics["recall"],

                    "f1_score":
                        metrics["f1_score"],

                    "roc_auc":
                        metrics["roc_auc"],

                    "confusion_matrix":
                        metrics["confusion_matrix"]
                }

            # ====================================================
            # SAVE DETAILED JSON REPORT
            # ====================================================

            report_directory = os.path.join(
                "results",
                self.module_name
            )

            os.makedirs(
                report_directory,
                exist_ok=True
            )

            report_path = os.path.join(
                report_directory,
                "detailed_model_report.json"
            )

            with open(
                report_path,
                "w"
            ) as f:

                json.dump(
                    clean_report,
                    f,
                    indent=4
                )

            logging.info(
                f"[{self.module_name}] "
                f"Detailed report saved at: "
                f"{report_path}"
            )

            # ====================================================
            # RETURN VALUES
            # ====================================================

            return (
                best_model_name,
                best_score,
                clean_report
            )

        except Exception as e:

            logging.error(
                f"[{self.module_name}] "
                f"Model training failed."
            )

            raise CustomException(
                e,
                sys
            )