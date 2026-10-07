"""
Prediction Pipeline

Single interface between Flask application and trained models.

Modules:
1. Mental Health
2. Stroke
3. Diabetes

The Flask application should call only this pipeline.
"""

import os
import sys

import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from src.components.data_transformation import clean_text

import os
import sys

import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from src.components.data_transformation import clean_text


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_PATHS = {

    "mental_health": {
        "model": os.path.join(
            "models",
            "mental_health",
            "best_model.pkl"
        ),

        "preprocessor": os.path.join(
            "models",
            "mental_health",
            "vectorizer.pkl"
        ),
    },

    "stroke": {
        "model": os.path.join(
            "models",
            "stroke",
            "best_model.pkl"
        ),

        "preprocessor": os.path.join(
            "models",
            "stroke",
            "preprocessor.pkl"
        ),
    },

    "diabetes": {
        "model": os.path.join(
            "models",
            "diabetes",
            "best_model.pkl"
        ),

        "preprocessor": os.path.join(
            "models",
            "diabetes",
            "preprocessor.pkl"
        ),
    },
}


# ============================================================
# PREDICT PIPELINE
# ============================================================

class PredictPipeline:

    def __init__(self):

        # Models are loaded only once
        # and then kept in memory.
        self._cache = {}


    # ========================================================
    # LOAD MODEL + PREPROCESSOR
    # ========================================================

    def _load(self, module_name):

        if module_name not in self._cache:

            paths = MODEL_PATHS[module_name]

            saved_model = load_object(
                paths["model"]
            )

            preprocessor = load_object(
                paths["preprocessor"]
            )

            # ------------------------------------------------
            # Diabetes optimizer saves:
            #
            # {
            #     "model": model,
            #     "threshold": threshold,
            #     "model_name": model_name
            # }
            #
            # Stroke may still have a normal model object.
            # ------------------------------------------------

            if (
                module_name == "diabetes"
                and isinstance(saved_model, dict)
                and "model" in saved_model
            ):

                model = saved_model["model"]

                threshold = saved_model.get(
                    "threshold",
                    0.50
                )

                model_name = saved_model.get(
                    "model_name",
                    "Unknown"
                )

            else:

                model = saved_model

                threshold = 0.50

                model_name = "Standard Model"

            self._cache[module_name] = (
                model,
                preprocessor,
                threshold,
                model_name
            )

            logging.info(
                f"[{module_name}] "
                f"Model + preprocessor loaded into cache"
            )

        return self._cache[module_name]


    # ========================================================
    # MENTAL HEALTH
    # ========================================================

    def predict_mental_health(
        self,
        text: str
    ):

        try:

            (
                model,
                vectorizer,
                threshold,
                model_name
            ) = self._load(
                "mental_health"
            )

            cleaned = clean_text(
                text
            )

            X = vectorizer.transform(
                [cleaned]
            )

            prediction = model.predict(
                X
            )[0]

            confidence = None

            if hasattr(
                model,
                "predict_proba"
            ):

                confidence = float(
                    max(
                        model.predict_proba(X)[0]
                    )
                )

            return {
                "prediction": prediction,
                "confidence": confidence
            }

        except Exception as e:

            raise CustomException(
                e,
                sys
            )


    # ========================================================
    # STROKE
    # ========================================================

    def predict_stroke(
        self,
        input_dict: dict
    ):

        try:

            (
                model,
                preprocessor,
                threshold,
                model_name
            ) = self._load(
                "stroke"
            )

            df = pd.DataFrame(
                [input_dict]
            )

            X = preprocessor.transform(
                df
            )

            if hasattr(
                X,
                "toarray"
            ):

                X = X.toarray()

            # ------------------------------------------------
            # Stroke prediction
            # ------------------------------------------------

            if hasattr(
                model,
                "predict_proba"
            ):

                probabilities = (
                    model.predict_proba(X)[0]
                )

                prediction = int(
                    probabilities[1] >= threshold
                )

                confidence = float(
                    probabilities[prediction]
                )

            else:

                prediction = int(
                    model.predict(X)[0]
                )

                confidence = None

            return {
                "prediction": prediction,
                "confidence": confidence
            }

        except Exception as e:

            raise CustomException(
                e,
                sys
            )


    # ========================================================
    # DIABETES
    # ========================================================

    def predict_diabetes(
        self,
        input_dict: dict
    ):

        try:

            (
                model,
                preprocessor,
                threshold,
                model_name
            ) = self._load(
                "diabetes"
            )

            # ------------------------------------------------
            # Convert Flask input into DataFrame
            # ------------------------------------------------

            df = pd.DataFrame(
                [input_dict]
            )

            # ------------------------------------------------
            # Apply same preprocessing used during training
            # ------------------------------------------------

            X = preprocessor.transform(
                df
            )

            if hasattr(
                X,
                "toarray"
            ):

                X = X.toarray()

            # ------------------------------------------------
            # Probability-based prediction
            # ------------------------------------------------

            if hasattr(
                model,
                "predict_proba"
            ):

                probabilities = (
                    model.predict_proba(X)[0]
                )

                positive_probability = float(
                    probabilities[1]
                )

                # --------------------------------------------
                # IMPORTANT:
                #
                # Use optimized threshold.
                #
                # Example:
                # threshold = 0.40
                #
                # probability >= 0.40
                #       ↓
                # Diabetes Risk
                #
                # probability < 0.40
                #       ↓
                # Low Risk
                # --------------------------------------------

                prediction = int(
                    positive_probability >= threshold
                )

                confidence = (
                    positive_probability
                    if prediction == 1
                    else float(
                        probabilities[0]
                    )
                )

            else:

                prediction = int(
                    model.predict(X)[0]
                )

                confidence = None

            # ------------------------------------------------
            # Return useful information to Flask
            # ------------------------------------------------

            return {

                "prediction": prediction,

                "confidence": confidence,

                "probability": (
                    positive_probability
                    if "positive_probability"
                    in locals()
                    else None
                ),

                "threshold": threshold,

                "model_name": model_name
            }

        except Exception as e:

            raise CustomException(
                e,
                sys
            )


# ============================================================
# CUSTOM DATA HELPER
# ============================================================

class CustomData:

    def __init__(
        self,
        **kwargs
    ):

        self.data = kwargs


    def to_dict(self):

        return self.data