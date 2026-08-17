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
# One config entry per module: where the raw file lives, and where
# train/test splits should be written to.
# TODO: update `raw_data_path` once you've placed the actual datasets
# in data/raw/<module>/
# ------------------------------------------------------------------
MODULE_CONFIG = {
    "mental_health": {
        "raw_data_path": os.path.join("data", "raw", "mental_health", "mental_health_raw.csv"),
        "train_path": os.path.join("data", "processed", "mental_health", "train.csv"),
        "test_path": os.path.join("data", "processed", "mental_health", "test.csv"),
        "target_column": "label",   # e.g. depression / anxiety / stress / normal
    },
    "stroke": {
        "raw_data_path": os.path.join("data", "raw", "stroke", "stroke_raw.csv"),
        "train_path": os.path.join("data", "processed", "stroke", "train.csv"),
        "test_path": os.path.join("data", "processed", "stroke", "test.csv"),
        "target_column": "stroke",
    },
    "diabetes": {
        "raw_data_path": os.path.join("data", "raw", "diabetes", "diabetes_raw.csv"),
        "train_path": os.path.join("data", "processed", "diabetes", "train.csv"),
        "test_path": os.path.join("data", "processed", "diabetes", "test.csv"),
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
            raise ValueError(f"Unknown module_name '{module_name}'. Must be one of {list(MODULE_CONFIG)}")
        self.module_name = module_name
        self.config = DataIngestionConfig(**MODULE_CONFIG[module_name])

    def initiate_data_ingestion(self, test_size=0.2, random_state=42):
        """
        Reads the raw CSV for this module, does a stratified train/test split,
        writes both splits to data/processed/<module>/, and returns their paths.
        """
        logging.info(f"[{self.module_name}] Starting data ingestion")
        try:
            df = pd.read_csv(self.config.raw_data_path)
            logging.info(f"[{self.module_name}] Loaded raw data with shape {df.shape}")

            os.makedirs(os.path.dirname(self.config.train_path), exist_ok=True)

            train_set, test_set = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state,
                stratify=df[self.config.target_column] if self.config.target_column in df.columns else None,
            )

            train_set.to_csv(self.config.train_path, index=False)
            test_set.to_csv(self.config.test_path, index=False)

            logging.info(f"[{self.module_name}] Train/test split done: "
                         f"train={train_set.shape}, test={test_set.shape}")

            return self.config.train_path, self.config.test_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Quick manual test: python -m src.components.data_ingestion
    for module in ["diabetes"]:   # start with diabetes first, per our build order
        ingestion = DataIngestion(module_name=module)
        train_path, test_path = ingestion.initiate_data_ingestion()
        print(f"{module}: train -> {train_path}, test -> {test_path}")
