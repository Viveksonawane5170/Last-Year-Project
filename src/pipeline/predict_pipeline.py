"""
Predict Pipeline: the single seam where the app (Streamlit/FastAPI) meets
the 3 trained models. UI/API code should ONLY ever call functions in this
file — never load models directly elsewhere.
"""

import os
import sys

import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from src.components.data_transformation import clean_text

MODEL_PATHS = {
    "mental_health": {
        "model": os.path.join("models", "mental_health", "best_model.pkl"),
        "preprocessor": os.path.join("models", "mental_health", "vectorizer.pkl"),
    },
    "stroke": {
        "model": os.path.join("models", "stroke", "best_model.pkl"),
        "preprocessor": os.path.join("models", "stroke", "preprocessor.pkl"),
    },
    "diabetes": {
        "model": os.path.join("models", "diabetes", "best_model.pkl"),
        "preprocessor": os.path.join("models", "diabetes", "preprocessor.pkl"),
    },
}


class PredictPipeline:
    def __init__(self):
        # Lazy-loaded cache so we don't reload models from disk on every request
        self._cache = {}

    def _load(self, module_name):
        if module_name not in self._cache:
            paths = MODEL_PATHS[module_name]
            model = load_object(paths["model"])
            preprocessor = load_object(paths["preprocessor"])
            self._cache[module_name] = (model, preprocessor)
            logging.info(f"[{module_name}] Model + preprocessor loaded into cache")
        return self._cache[module_name]

    def predict_mental_health(self, text: str):
        try:
            model, vectorizer = self._load("mental_health")
            cleaned = clean_text(text)
            X = vectorizer.transform([cleaned])
            prediction = model.predict(X)[0]
            confidence = None
            if hasattr(model, "predict_proba"):
                confidence = float(max(model.predict_proba(X)[0]))
            return {"prediction": prediction, "confidence": confidence}
        except Exception as e:
            raise CustomException(e, sys)

    def predict_stroke(self, input_dict: dict):
        try:
            model, preprocessor = self._load("stroke")
            df = pd.DataFrame([input_dict])
            X = preprocessor.transform(df)
            if hasattr(X, "toarray"):
                X = X.toarray()
            prediction = int(model.predict(X)[0])
            confidence = None
            if hasattr(model, "predict_proba"):
                confidence = float(model.predict_proba(X)[0][prediction])
            return {"prediction": prediction, "confidence": confidence}
        except Exception as e:
            raise CustomException(e, sys)

    def predict_diabetes(self, input_dict: dict):
        try:
            model, preprocessor = self._load("diabetes")
            df = pd.DataFrame([input_dict])
            X = preprocessor.transform(df)
            if hasattr(X, "toarray"):
                X = X.toarray()
            prediction = int(model.predict(X)[0])
            confidence = None
            if hasattr(model, "predict_proba"):
                confidence = float(model.predict_proba(X)[0][prediction])
            return {"prediction": prediction, "confidence": confidence}
        except Exception as e:
            raise CustomException(e, sys)


# ------------------------------------------------------------------
# Example CustomData-style helper for stroke/diabetes forms (optional).
# The app layer can build this dict directly from form fields instead,
# this class is just for convenience/validation if you want it.
# ------------------------------------------------------------------
class CustomData:
    def __init__(self, **kwargs):
        self.data = kwargs

    def to_dict(self):
        return self.data
