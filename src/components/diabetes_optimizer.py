"""
Diabetes Model Optimizer

Includes:
1. Individual Models
2. Hard Voting Ensemble
3. Soft Voting Ensemble
4. Threshold Tuning
5. 5-Fold Cross Validation
6. Final Test Evaluation
7. Automatic Best Model Selection
8. Model + Threshold saved to best_model.pkl
9. Complete comparison saved to optimized_comparison.csv
"""

import os
import sys
import pickle
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging


warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

TRAIN_PATH = os.path.join(
    "data",
    "processed",
    "diabetes",
    "train.csv"
)

TEST_PATH = os.path.join(
    "data",
    "processed",
    "diabetes",
    "test.csv"
)

PREPROCESSOR_PATH = os.path.join(
    "models",
    "diabetes",
    "preprocessor.pkl"
)

MODEL_PATH = os.path.join(
    "models",
    "diabetes",
    "best_model.pkl"
)

RESULT_PATH = os.path.join(
    "results",
    "diabetes",
    "optimized_comparison.csv"
)


# ============================================================
# CONFIG
# ============================================================

TARGET_COLUMN = "Outcome"

RANDOM_STATE = 42

N_SPLITS = 5

# Threshold range
THRESHOLDS = np.arange(
    0.30,
    0.71,
    0.01
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    logging.info(
        "Loading diabetes train and test data"
    )

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    test_df = pd.read_csv(
        TEST_PATH
    )

    X_train = train_df.drop(
        columns=[TARGET_COLUMN]
    )

    y_train = train_df[TARGET_COLUMN]

    X_test = test_df.drop(
        columns=[TARGET_COLUMN]
    )

    y_test = test_df[TARGET_COLUMN]

    return (
        X_train,
        y_train,
        X_test,
        y_test
    )


# ============================================================
# LOAD PREPROCESSOR
# ============================================================

def load_preprocessor():

    logging.info(
        "Loading diabetes preprocessor"
    )

    with open(
        PREPROCESSOR_PATH,
        "rb"
    ) as file:

        preprocessor = pickle.load(
            file
        )

    return preprocessor


# ============================================================
# CREATE INDIVIDUAL MODELS
# ============================================================

def create_individual_models():

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE
        ),

        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            random_state=RANDOM_STATE
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE
        )
    }

    return models


# ============================================================
# CREATE VOTING MODELS
# ============================================================

def create_voting_models():

    # Fresh models for Hard Voting
    hard_rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    hard_gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE
    )

    hard_xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE
    )

    hard_voting = VotingClassifier(
        estimators=[
            ("rf", hard_rf),
            ("gb", hard_gb),
            ("xgb", hard_xgb)
        ],
        voting="hard"
    )

    # Fresh models for Soft Voting
    soft_rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    soft_gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE
    )

    soft_xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE
    )

    soft_voting = VotingClassifier(
        estimators=[
            ("rf", soft_rf),
            ("gb", soft_gb),
            ("xgb", soft_xgb)
        ],
        voting="soft"
    )

    return (
        hard_voting,
        soft_voting
    )


# ============================================================
# FIND BEST THRESHOLD
# ============================================================

def find_best_threshold(
    y_true,
    probabilities
):

    best_threshold = 0.50

    best_score = -1

    for threshold in THRESHOLDS:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0
        )

        # Health project:
        # Give importance to both F1 and Recall
        score = (
            f1 +
            (0.05 * recall)
        )

        if score > best_score:

            best_score = score

            best_threshold = threshold

    return best_threshold


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    y_true,
    predictions,
    probabilities=None
):

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    if probabilities is not None:

        roc_auc = roc_auc_score(
            y_true,
            probabilities
        )

        pr_auc = average_precision_score(
            y_true,
            probabilities
        )

    else:

        roc_auc = np.nan

        pr_auc = np.nan

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions
    ).ravel()

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn
    }


# ============================================================
# OPTIMIZE ONE MODEL
# ============================================================

def optimize_model(
    model_name,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    cv
):

    print("\n")
    print("-" * 70)
    print(
        f"Optimizing: {model_name}"
    )

    # --------------------------------------------------------
    # HARD VOTING
    # --------------------------------------------------------

    if model_name == "Hard Voting":

        oof_predictions = cross_val_predict(
            model,
            X_train,
            y_train,
            cv=cv,
            method="predict"
        )

        cv_f1 = f1_score(
            y_train,
            oof_predictions,
            zero_division=0
        )

        cv_recall = recall_score(
            y_train,
            oof_predictions,
            zero_division=0
        )

        threshold = 0.50

        final_model = clone(
            model
        )

        final_model.fit(
            X_train,
            y_train
        )

        test_predictions = (
            final_model.predict(
                X_test
            )
        )

        metrics = calculate_metrics(
            y_test,
            test_predictions
        )

    # --------------------------------------------------------
    # ALL PROBABILITY MODELS
    # --------------------------------------------------------

    else:

        # OOF probabilities
        oof_probabilities = cross_val_predict(
            model,
            X_train,
            y_train,
            cv=cv,
            method="predict_proba"
        )[:, 1]

        # Threshold is learned ONLY from training OOF
        threshold = find_best_threshold(
            y_train,
            oof_probabilities
        )

        oof_predictions = (
            oof_probabilities >= threshold
        ).astype(int)

        cv_f1 = f1_score(
            y_train,
            oof_predictions,
            zero_division=0
        )

        cv_recall = recall_score(
            y_train,
            oof_predictions,
            zero_division=0
        )

        # Train final model
        final_model = clone(
            model
        )

        final_model.fit(
            X_train,
            y_train
        )

        # Test probabilities
        test_probabilities = (
            final_model.predict_proba(
                X_test
            )[:, 1]
        )

        # Apply tuned threshold
        test_predictions = (
            test_probabilities >= threshold
        ).astype(int)

        metrics = calculate_metrics(
            y_test,
            test_predictions,
            test_probabilities
        )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {
        "Model": model_name,
        "Threshold": threshold,

        "Accuracy": metrics["Accuracy"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "F1": metrics["F1"],
        "ROC_AUC": metrics["ROC_AUC"],
        "PR_AUC": metrics["PR_AUC"],

        "TP": metrics["TP"],
        "TN": metrics["TN"],
        "FP": metrics["FP"],
        "FN": metrics["FN"],

        "CV_F1": cv_f1,
        "CV_Recall": cv_recall
    }

    print(
        f"Threshold : {threshold:.3f}"
    )

    print(
        f"Accuracy  : {metrics['Accuracy']:.3f}"
    )

    print(
        f"Precision : {metrics['Precision']:.3f}"
    )

    print(
        f"Recall    : {metrics['Recall']:.3f}"
    )

    print(
        f"F1        : {metrics['F1']:.3f}"
    )

    if not np.isnan(
        metrics["ROC_AUC"]
    ):

        print(
            f"ROC-AUC   : {metrics['ROC_AUC']:.3f}"
        )

    if not np.isnan(
        metrics["PR_AUC"]
    ):

        print(
            f"PR-AUC    : {metrics['PR_AUC']:.3f}"
        )

    print(
        f"TP={metrics['TP']} "
        f"FN={metrics['FN']} "
        f"FP={metrics['FP']} "
        f"TN={metrics['TN']}"
    )

    print(
        f"CV F1     : {cv_f1:.3f}"
    )

    print(
        f"CV Recall : {cv_recall:.3f}"
    )

    return (
        result,
        final_model
    )


# ============================================================
# MAIN OPTIMIZER
# ============================================================

def optimize_diabetes():

    try:

        print("\n")

        print("=" * 70)

        print(
            "          DIABETES COMPLETE MODEL OPTIMIZER"
        )

        print("=" * 70)

        # ====================================================
        # 1. Load data
        # ====================================================

        (
            X_train,
            y_train,
            X_test,
            y_test
        ) = load_data()

        print(
            f"\nTrain shape: {X_train.shape}"
        )

        print(
            f"Test shape : {X_test.shape}"
        )

        print(
            "\nTraining class distribution:"
        )

        print(
            y_train.value_counts()
        )

        # ====================================================
        # 2. Load preprocessor
        # ====================================================

        preprocessor = load_preprocessor()

        # ====================================================
        # 3. Transform data
        # ====================================================

        X_train_transformed = (
            preprocessor.transform(
                X_train
            )
        )

        X_test_transformed = (
            preprocessor.transform(
                X_test
            )
        )

        print(
            "\nPreprocessing completed."
        )

        # ====================================================
        # 4. Create models
        # ====================================================

        individual_models = (
            create_individual_models()
        )

        (
            hard_voting,
            soft_voting
        ) = create_voting_models()

        all_models = {}

        # Add individual models
        for name, model in individual_models.items():

            all_models[name] = model

        # Add ensemble models
        all_models[
            "Hard Voting"
        ] = hard_voting

        all_models[
            "Soft Voting"
        ] = soft_voting

        print("\nModels to evaluate:")

        for name in all_models:

            print(
                f"  - {name}"
            )

        # ====================================================
        # 5. Cross-validation
        # ====================================================

        cv = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE
        )

        results = []

        trained_models = {}

        # ====================================================
        # 6. Optimize every model
        # ====================================================

        for model_name, model in all_models.items():

            (
                result,
                trained_model
            ) = optimize_model(
                model_name,
                model,
                X_train_transformed,
                y_train,
                X_test_transformed,
                y_test,
                cv
            )

            results.append(
                result
            )

            trained_models[
                model_name
            ] = trained_model

        # ====================================================
        # 7. Create results DataFrame
        # ====================================================

        results_df = pd.DataFrame(
            results
        )

        # ====================================================
        # 8. Model selection
        # ====================================================
        #
        # IMPORTANT:
        # We use CV results for selection.
        # Test data is NOT used to select the model.
        #
        # Primary:
        #       CV F1
        #
        # Secondary:
        #       CV Recall
        #
        # ====================================================

        results_df["SelectionScore"] = (
            results_df["CV_F1"]
            +
            (
                0.05 *
                results_df["CV_Recall"]
            )
        )

        results_df = results_df.sort_values(
            by="SelectionScore",
            ascending=False
        ).reset_index(
            drop=True
        )

        best_model_name = (
            results_df.iloc[0]["Model"]
        )

        best_threshold = (
            results_df.iloc[0]["Threshold"]
        )

        best_model = trained_models[
            best_model_name
        ]

        # ====================================================
        # 9. Save comparison
        # ====================================================

        os.makedirs(
            os.path.dirname(
                RESULT_PATH
            ),
            exist_ok=True
        )

        results_df.to_csv(
            RESULT_PATH,
            index=False
        )

        # ====================================================
        # 10. Save final model
        # ====================================================

        model_package = {

            "model": best_model,

            "threshold": float(
                best_threshold
            ),

            "model_name":
                best_model_name
        }

        os.makedirs(
            os.path.dirname(
                MODEL_PATH
            ),
            exist_ok=True
        )

        with open(
            MODEL_PATH,
            "wb"
        ) as file:

            pickle.dump(
                model_package,
                file
            )

        # ====================================================
        # 11. Final results
        # ====================================================

        print("\n")

        print("=" * 70)

        print(
            "             FINAL DIABETES RESULTS"
        )

        print("=" * 70)

        display_columns = [
            "Model",
            "Threshold",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC_AUC",
            "PR_AUC",
            "TP",
            "TN",
            "FP",
            "FN",
            "CV_F1",
            "CV_Recall",
            "SelectionScore"
        ]

        print(
            results_df[
                display_columns
            ].to_string(
                index=False
            )
        )

        print("\n")

        print("=" * 70)

        print(
            f"FINAL MODEL: {best_model_name}"
        )

        print(
            f"FINAL THRESHOLD: "
            f"{best_threshold:.3f}"
        )

        print("=" * 70)

        print("\nFiles created/updated:")

        print(
            f"1. {RESULT_PATH}"
        )

        print(
            f"2. {MODEL_PATH}"
        )

        print(
            f"3. {PREPROCESSOR_PATH}"
        )

        print("\nDiabetes optimization completed.")

        print("=" * 70)

        return results_df

    except Exception as e:

        raise CustomException(
            e,
            sys
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    optimize_diabetes()