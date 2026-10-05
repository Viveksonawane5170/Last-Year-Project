import os
import sys
import csv

from sklearn.ensemble import (
    VotingClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.tree import DecisionTreeClassifier

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


class EnsembleTrainer:

    def __init__(self, module_name):

        self.module_name = module_name

        self.config = {

            "diabetes": {
                "model_path": "models/diabetes/best_model.pkl",
                "result_path": "results/diabetes/ensemble_comparison.csv"
            },

            "stroke": {
                "model_path": "models/stroke/best_model.pkl",
                "result_path": "results/stroke/ensemble_comparison.csv"
            },

            "mental_health": {
                "model_path": "models/mental_health/best_model.pkl",
                "result_path": "results/mental_health/ensemble_comparison.csv"
            }
        }

    # ---------------------------------------------------------
    # CREATE BASE MODELS
    # ---------------------------------------------------------

    def create_base_models(self):

        models = {

            "Decision Tree": DecisionTreeClassifier(
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),

            "Random Forest": RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),

            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),

            "XGBoost": XGBClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                min_child_weight=3,
                gamma=0,
                eval_metric="logloss",
                random_state=42
            )
        }

        return models

    # ---------------------------------------------------------
    # EVALUATE MODEL
    # ---------------------------------------------------------

    def evaluate_model(self, model, X_test, y_test):

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        roc_auc = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(X_test)[:, 1]

            roc_auc = roc_auc_score(
                y_test,
                probabilities
            )

        elif hasattr(model, "decision_function"):

            scores = model.decision_function(X_test)

            roc_auc = roc_auc_score(
                y_test,
                scores
            )

        cm = confusion_matrix(
            y_test,
            predictions
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm.tolist()
        }

    # ---------------------------------------------------------
    # ENSEMBLE TRAINING
    # ---------------------------------------------------------

    def initiate_ensemble_training(
        self,
        X_train,
        y_train,
        X_test,
        y_test
    ):

        try:

            logging.info(
                f"Starting ensemble training for {self.module_name}"
            )

            # Convert sparse matrix to dense
            if hasattr(X_train, "toarray"):
                X_train = X_train.toarray()

            if hasattr(X_test, "toarray"):
                X_test = X_test.toarray()

            base_models = self.create_base_models()

            results = {}

            # =================================================
            # 1. INDIVIDUAL MODELS
            # =================================================

            for name, model in base_models.items():

                print(f"\nTraining {name}...")

                model.fit(
                    X_train,
                    y_train
                )

                metrics = self.evaluate_model(
                    model,
                    X_test,
                    y_test
                )

                results[name] = metrics

                print(name)

                print(
                    f"Accuracy : {metrics['accuracy']:.4f}"
                )

                print(
                    f"Precision: {metrics['precision']:.4f}"
                )

                print(
                    f"Recall   : {metrics['recall']:.4f}"
                )

                print(
                    f"F1 Score : {metrics['f1_score']:.4f}"
                )

                if metrics["roc_auc"] is not None:

                    print(
                        f"ROC-AUC  : {metrics['roc_auc']:.4f}"
                    )

            # =================================================
            # 2. HARD VOTING
            # =================================================

            print("\nTraining Hard Voting...")

            hard_voting = VotingClassifier(

                estimators=[

                    (
                        "dt",
                        DecisionTreeClassifier(
                            max_depth=5,
                            min_samples_split=5,
                            min_samples_leaf=2,
                            random_state=42
                        )
                    ),

                    (
                        "rf",
                        RandomForestClassifier(
                            n_estimators=300,
                            max_depth=10,
                            min_samples_split=5,
                            min_samples_leaf=2,
                            random_state=42,
                            n_jobs=-1
                        )
                    ),

                    (
                        "gb",
                        GradientBoostingClassifier(
                            n_estimators=200,
                            learning_rate=0.05,
                            max_depth=3,
                            min_samples_split=5,
                            min_samples_leaf=2,
                            random_state=42
                        )
                    ),

                    (
                        "xgb",
                        XGBClassifier(
                            n_estimators=200,
                            max_depth=3,
                            learning_rate=0.05,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            min_child_weight=3,
                            gamma=0,
                            eval_metric="logloss",
                            random_state=42
                        )
                    )
                ],

                voting="hard"
            )

            hard_voting.fit(
                X_train,
                y_train
            )

            hard_metrics = self.evaluate_model(
                hard_voting,
                X_test,
                y_test
            )

            results["Hard Voting"] = hard_metrics

            print("Hard Voting")

            print(
                f"Accuracy : {hard_metrics['accuracy']:.4f}"
            )

            print(
                f"Precision: {hard_metrics['precision']:.4f}"
            )

            print(
                f"Recall   : {hard_metrics['recall']:.4f}"
            )

            print(
                f"F1 Score : {hard_metrics['f1_score']:.4f}"
            )

            # =================================================
            # 3. SOFT VOTING
            # =================================================

            print("\nTraining Soft Voting...")

            soft_voting = VotingClassifier(

                estimators=[

                    (
                        "dt",
                        DecisionTreeClassifier(
                            max_depth=5,
                            min_samples_split=5,
                            min_samples_leaf=2,
                            random_state=42
                        )
                    ),

                    (
                        "rf",
                        RandomForestClassifier(
                            n_estimators=300,
                            max_depth=10,
                            min_samples_split=5,
                            min_samples_leaf=2,
                            random_state=42,
                            n_jobs=-1
                        )
                    ),

                    (
                        "gb",
                        GradientBoostingClassifier(
                            n_estimators=200,
                            learning_rate=0.05,
                            max_depth=3,
                            min_samples_split=5,
                            min_samples_leaf=2,
                            random_state=42
                        )
                    ),

                    (
                        "xgb",
                        XGBClassifier(
                            n_estimators=200,
                            max_depth=3,
                            learning_rate=0.05,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            min_child_weight=3,
                            gamma=0,
                            eval_metric="logloss",
                            random_state=42
                        )
                    )
                ],

                voting="soft"
            )

            soft_voting.fit(
                X_train,
                y_train
            )

            soft_metrics = self.evaluate_model(
                soft_voting,
                X_test,
                y_test
            )

            results["Soft Voting"] = soft_metrics

            print("Soft Voting")

            print(
                f"Accuracy : {soft_metrics['accuracy']:.4f}"
            )

            print(
                f"Precision: {soft_metrics['precision']:.4f}"
            )

            print(
                f"Recall   : {soft_metrics['recall']:.4f}"
            )

            print(
                f"F1 Score : {soft_metrics['f1_score']:.4f}"
            )

            if soft_metrics["roc_auc"] is not None:

                print(
                    f"ROC-AUC  : {soft_metrics['roc_auc']:.4f}"
                )

            # =================================================
            # 4. FIND BEST MODEL
            # =================================================

            best_name = max(
                results,
                key=lambda x: results[x]["f1_score"]
            )

            best_score = results[
                best_name
            ]["f1_score"]

            # =================================================
            # 5. GET BEST MODEL OBJECT
            # =================================================

            if best_name == "Hard Voting":

                best_model = hard_voting

            elif best_name == "Soft Voting":

                best_model = soft_voting

            else:

                best_model = base_models[
                    best_name
                ]

            # =================================================
            # 6. SAVE FINAL BEST MODEL
            # =================================================

            model_path = self.config[
                self.module_name
            ]["model_path"]

            os.makedirs(
                os.path.dirname(model_path),
                exist_ok=True
            )

            save_object(
                model_path,
                best_model
            )

            print(
                f"\nFinal model saved at: {model_path}"
            )

            # =================================================
            # 7. SAVE COMPARISON CSV
            # =================================================

            result_path = self.config[
                self.module_name
            ]["result_path"]

            os.makedirs(
                os.path.dirname(result_path),
                exist_ok=True
            )

            with open(
                result_path,
                "w",
                newline=""
            ) as file:

                writer = csv.writer(file)

                writer.writerow([
                    "model",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1_score",
                    "roc_auc"
                ])

                for name, metrics in results.items():

                    writer.writerow([
                        name,
                        metrics["accuracy"],
                        metrics["precision"],
                        metrics["recall"],
                        metrics["f1_score"],
                        metrics["roc_auc"]
                    ])

            # =================================================
            # 8. FINAL OUTPUT
            # =================================================

            print("\n" + "=" * 60)

            print(
                f"BEST ENSEMBLE MODEL: {best_name}"
            )

            print(
                f"BEST F1 SCORE: {best_score:.4f}"
            )

            print("=" * 60)

            logging.info(
                f"Best ensemble model: {best_name}"
            )

            return (
                best_name,
                best_score,
                results
            )

        except Exception as e:

            raise CustomException(
                e,
                sys
            )