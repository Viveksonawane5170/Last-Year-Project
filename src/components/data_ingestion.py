"""
Data Ingestion component.
Handles all 3 modules (mental_health, stroke, diabetes) through one class,
driven by a per-module config, instead of 3 separate files.
"""

import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


# ------------------------------------------------------------------
# One config entry per module: where the dataset lives, and where
# train/test splits should be written to.
# ------------------------------------------------------------------
MODULE_CONFIG = {
    "mental_health": {
        "raw_data_path": os.path.join(
            "data", "raw", "mental_health", "mental_health_raw.csv"
        ),
        "train_path": os.path.join(
            "data", "processed", "mental_health", "train.csv"
        ),
        "test_path": os.path.join(
            "data", "processed", "mental_health", "test.csv"
        ),
        "target_column": "label",
    },

    "stroke": {
        "raw_data_path": os.path.join(
            "data", "raw", "stroke", "stroke_raw.csv"
        ),
        "train_path": os.path.join(
            "data", "processed", "stroke", "train.csv"
        ),
        "test_path": os.path.join(
            "data", "processed", "stroke", "test.csv"
        ),
        "target_column": "stroke",
    },

    "diabetes": {
        # IMPORTANT:
        # Use cleaned diabetes dataset
        "raw_data_path": os.path.join(
            "data", "raw", "diabetes", "diabetes_clean.csv"
        ),
        "train_path": os.path.join(
            "data", "processed", "diabetes", "train.csv"
        ),
        "test_path": os.path.join(
            "data", "processed", "diabetes", "test.csv"
        ),
        "target_column": "Outcome",
    },
}


@dataclass
class DataIngestionConfig:
    raw_data_path: str
    train_path: str
    test_path: str
    target_column: str


class DataIngestion:

    def __init__(self, module_name: str):

        if module_name not in MODULE_CONFIG:
            raise ValueError(
                f"Unknown module_name '{module_name}'. "
                f"Must be one of {list(MODULE_CONFIG)}"
            )

        self.module_name = module_name
        self.config = DataIngestionConfig(
            **MODULE_CONFIG[module_name]
        )

    def initiate_data_ingestion(self, test_size=0.2, random_state=42):
        """
        Reads the dataset for this module,
        performs a stratified train/test split,
        saves train and test datasets,
        and returns their paths.
        """

        logging.info(
            f"[{self.module_name}] Starting data ingestion"
        )

        try:

            # ------------------------------------------------------
            # 1. Read dataset
            # ------------------------------------------------------
            df = pd.read_csv(
                self.config.raw_data_path
            )

            logging.info(
                f"[{self.module_name}] "
                f"Loaded data with shape {df.shape}"
            )

            # ------------------------------------------------------
            # 2. Create processed directory
            # ------------------------------------------------------
            os.makedirs(
                os.path.dirname(self.config.train_path),
                exist_ok=True
            )

            # ------------------------------------------------------
            # 3. Train-test split
            # ------------------------------------------------------
            train_set, test_set = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state,
                stratify=(
                    df[self.config.target_column]
                    if self.config.target_column in df.columns
                    else None
                )
            )

            # ------------------------------------------------------
            # 4. Save train dataset
            # ------------------------------------------------------
            train_set.to_csv(
                self.config.train_path,
                index=False
            )

            # ------------------------------------------------------
            # 5. Save test dataset
            # ------------------------------------------------------
            test_set.to_csv(
                self.config.test_path,
                index=False
            )

            logging.info(
                f"[{self.module_name}] Train/test split completed: "
                f"train={train_set.shape}, "
                f"test={test_set.shape}"
            )

            return (
                self.config.train_path,
                self.config.test_path
            )

        except Exception as e:

            raise CustomException(e, sys)


# --------------------------------------------------------------
# Manual testing
# Run:
# python -m src.components.data_ingestion
# --------------------------------------------------------------
if __name__ == "__main__":

    for module in ["diabetes"]:

        ingestion = DataIngestion(
            module_name=module
        )

        train_path, test_path = (
            ingestion.initiate_data_ingestion()
        )

        print(
            f"{module}: "
            f"train -> {train_path}, "
            f"test -> {test_path}"
        )