"""
Train Pipeline: wires ingestion -> transformation -> training together.
Run this per module, or loop over all 3.

Usage:
    python -m src.pipeline.train_pipeline --module diabetes
    python -m src.pipeline.train_pipeline --module all
"""

import argparse
import csv
import os
import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

ALL_MODULES = ["diabetes", "stroke", "mental_health"]  # build order: easiest tabular -> hardest (text)


def run_pipeline_for_module(module_name: str):
    try:
        logging.info(f"===== Starting pipeline for module: {module_name} =====")

        # Stage 1: Ingestion
        ingestion = DataIngestion(module_name=module_name)
        train_path, test_path = ingestion.initiate_data_ingestion()

        # Stage 2: Transformation
        transformation = DataTransformation(module_name=module_name)
        X_train, y_train, X_test, y_test = transformation.initiate_data_transformation(train_path, test_path)

        # Stage 3: Model Training + Evaluation + Save Best Model
        trainer = ModelTrainer(module_name=module_name)
        best_name, best_score, report = trainer.initiate_model_training(X_train, y_train, X_test, y_test)

        print(f"\n[{module_name}] Best model: {best_name} (score={best_score:.4f})")

        # Save comparison report to results/<module>/model_comparison.csv
        results_path = os.path.join("results", module_name, "model_comparison.csv")
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "accuracy", "precision", "recall", "f1_score", "roc_auc"])
            for name, metrics in report.items():
                writer.writerow([
                    name, metrics["accuracy"], metrics["precision"],
                    metrics["recall"], metrics["f1_score"], metrics["roc_auc"],
                ])
        logging.info(f"[{module_name}] Comparison report saved to {results_path}")

        return best_name, best_score

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=str, default="diabetes",
                         choices=ALL_MODULES + ["all"],
                         help="Which module to train. Use 'all' to run all 3 in build order.")
    args = parser.parse_args()

    if args.module == "all":
        for m in ALL_MODULES:
            run_pipeline_for_module(m)
    else:
        run_pipeline_for_module(args.module)
