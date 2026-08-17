"""
Data Transformation component.
Tabular modules (stroke, diabetes) get impute+encode+scale pipelines.
The text module (mental_health) gets a text-cleaning + TF-IDF pipeline.
One file, branching by module — instead of 3 separate files.
"""

import os
import re
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

# TODO: for real text cleaning you'll likely add nltk/spacy stopword removal
# and lemmatization here once the mental-health dataset is finalized.
BASIC_STOPWORDS = {"the", "a", "an", "is", "are", "and", "or", "to", "of", "in", "it", "i"}


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)                 # keep letters only
    tokens = [w for w in text.split() if w not in BASIC_STOPWORDS and len(w) > 1]
    return " ".join(tokens)


@dataclass
class TransformationConfig:
    preprocessor_path: str
    target_column: str


TRANSFORM_CONFIG = {
    "mental_health": {
        "preprocessor_path": os.path.join("models", "mental_health", "vectorizer.pkl"),
        "target_column": "label",
        "text_column": "text",
    },
    "stroke": {
        "preprocessor_path": os.path.join("models", "stroke", "preprocessor.pkl"),
        "target_column": "stroke",
        # TODO: confirm exact column names once dataset is finalized
        "numerical_columns": ["age", "avg_glucose_level", "bmi"],
        "categorical_columns": ["gender", "hypertension", "heart_disease",
                                 "ever_married", "work_type", "Residence_type", "smoking_status"],
    },
    "diabetes": {
        "preprocessor_path": os.path.join("models", "diabetes", "preprocessor.pkl"),
        "target_column": "Outcome",
        "numerical_columns": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                               "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
        "categorical_columns": [],
    },
}


class DataTransformation:
    def __init__(self, module_name: str):
        if module_name not in TRANSFORM_CONFIG:
            raise ValueError(f"Unknown module_name '{module_name}'")
        self.module_name = module_name
        self.config = TRANSFORM_CONFIG[module_name]

    # ---------------- Tabular path (stroke, diabetes) ----------------
    def _build_tabular_preprocessor(self):
        num_cols = self.config.get("numerical_columns", [])
        cat_cols = self.config.get("categorical_columns", [])

        num_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])

        transformers = []
        if num_cols:
            transformers.append(("num", num_pipeline, num_cols))
        if cat_cols:
            transformers.append(("cat", cat_pipeline, cat_cols))

        return ColumnTransformer(transformers=transformers)

    def _transform_tabular(self, train_path, test_path):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        target_col = self.config["target_column"]
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        preprocessor = self._build_tabular_preprocessor()
        X_train_arr = preprocessor.fit_transform(X_train)
        X_test_arr = preprocessor.transform(X_test)

        save_object(self.config["preprocessor_path"], preprocessor)
        logging.info(f"[{self.module_name}] Tabular preprocessor saved")

        return X_train_arr, y_train.values, X_test_arr, y_test.values

    # ---------------- Text path (mental_health) ----------------
    def _transform_text(self, train_path, test_path):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        text_col = self.config["text_column"]
        target_col = self.config["target_column"]

        train_df[text_col] = train_df[text_col].apply(clean_text)
        test_df[text_col] = test_df[text_col].apply(clean_text)

        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train_arr = vectorizer.fit_transform(train_df[text_col])
        X_test_arr = vectorizer.transform(test_df[text_col])

        save_object(self.config["preprocessor_path"], vectorizer)
        logging.info(f"[{self.module_name}] TF-IDF vectorizer saved (vocab size={len(vectorizer.vocabulary_)})")

        return X_train_arr, train_df[target_col].values, X_test_arr, test_df[target_col].values

    # ---------------- Public entrypoint ----------------
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info(f"[{self.module_name}] Starting data transformation")
            if self.module_name == "mental_health":
                return self._transform_text(train_path, test_path)
            else:
                return self._transform_tabular(train_path, test_path)
        except Exception as e:
            raise CustomException(e, sys)
