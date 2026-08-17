"""
Flask app: single entrypoint, one route per module.
Run with: python app/app.py   (from project root)
Then open http://127.0.0.1:5000
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, render_template, request

from src.pipeline.predict_pipeline import PredictPipeline
from src.exception import CustomException

app = Flask(__name__)
pipeline = PredictPipeline()


@app.route("/")
def home():
    return render_template("index.html")


# ---------------- Mental Health ----------------
@app.route("/mental-health", methods=["GET", "POST"])
def mental_health():
    result = None
    error = None
    text_value = ""

    if request.method == "POST":
        text_value = request.form.get("text", "")
        if not text_value.strip():
            error = "Please enter some text."
        else:
            try:
                result = pipeline.predict_mental_health(text_value)
            except Exception as e:
                error = f"Model not available yet. Train the mental_health module first. ({e})"

    return render_template("mental_health.html", result=result, error=error, text_value=text_value)


# ---------------- Stroke ----------------
@app.route("/stroke", methods=["GET", "POST"])
def stroke():
    result = None
    error = None
    form_data = {}

    if request.method == "POST":
        form_data = request.form.to_dict()
        try:
            input_dict = {
                "gender": form_data.get("gender"),
                "age": float(form_data.get("age", 0)),
                "hypertension": int(form_data.get("hypertension", 0)),
                "heart_disease": int(form_data.get("heart_disease", 0)),
                "ever_married": form_data.get("ever_married"),
                "work_type": form_data.get("work_type"),
                "Residence_type": form_data.get("residence_type"),
                "avg_glucose_level": float(form_data.get("avg_glucose_level", 0)),
                "bmi": float(form_data.get("bmi", 0)),
                "smoking_status": form_data.get("smoking_status"),
            }
            raw_result = pipeline.predict_stroke(input_dict)
            result = {
                "label": "High risk of stroke" if raw_result["prediction"] == 1 else "Low risk of stroke",
                "confidence": raw_result["confidence"],
            }
        except Exception as e:
            error = f"Model not available yet. Train the stroke module first. ({e})"

    return render_template("stroke.html", result=result, error=error, form_data=form_data)


# ---------------- Diabetes ----------------
@app.route("/diabetes", methods=["GET", "POST"])
def diabetes():
    result = None
    error = None
    form_data = {}

    if request.method == "POST":
        form_data = request.form.to_dict()
        try:
            input_dict = {
                "Pregnancies": int(form_data.get("pregnancies", 0)),
                "Glucose": float(form_data.get("glucose", 0)),
                "BloodPressure": float(form_data.get("blood_pressure", 0)),
                "SkinThickness": float(form_data.get("skin_thickness", 0)),
                "Insulin": float(form_data.get("insulin", 0)),
                "BMI": float(form_data.get("bmi", 0)),
                "DiabetesPedigreeFunction": float(form_data.get("dpf", 0)),
                "Age": int(form_data.get("age", 0)),
            }
            raw_result = pipeline.predict_diabetes(input_dict)
            result = {
                "label": "Diabetic" if raw_result["prediction"] == 1 else "Not Diabetic",
                "confidence": raw_result["confidence"],
            }
        except Exception as e:
            error = f"Model not available yet. Train the diabetes module first. ({e})"

    return render_template("diabetes.html", result=result, error=error, form_data=form_data)


if __name__ == "__main__":
    app.run(debug=True)
