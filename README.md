# Detection & Prediction of Mental Health, Brain Stroke, and Diabetes

End-to-end ML project with 3 independent modules (mental health, stroke, diabetes)
sharing one pipeline pattern: Ingestion → Transformation → Training → Prediction → App.

## Setup
```bash
pip install -r requirements.txt
```

## 1. Add your datasets
Place raw CSVs here (names must match `src/components/data_ingestion.py`'s `MODULE_CONFIG`):
```
data/raw/mental_health/mental_health_raw.csv   (columns: text, label)
data/raw/stroke/stroke_raw.csv                 (standard Kaggle stroke schema)
data/raw/diabetes/diabetes_raw.csv             (Pima Indians Diabetes schema)
```
If your column names differ, update the config dicts at the top of:
- `src/components/data_ingestion.py`
- `src/components/data_transformation.py`

## 2. Train a module
```bash
python -m src.pipeline.train_pipeline --module diabetes     # start here
python -m src.pipeline.train_pipeline --module stroke
python -m src.pipeline.train_pipeline --module mental_health
python -m src.pipeline.train_pipeline --module all          # runs all 3 in order
```
This runs ingestion → transformation → trains all 6 models → saves the best one to
`models/<module>/best_model.pkl` and writes a comparison table to
`results/<module>/model_comparison.csv`.

## 3. Run the app
```bash
python app/app.py
```
Then open http://127.0.0.1:5000 in your browser. Routes: `/`, `/mental-health`, `/stroke`, `/diabetes`.

## Project structure
```
src/
  exception.py        custom exception with file+line info
  logger.py            timestamped logging setup
  utils.py             save/load objects, model evaluation helpers
  components/
    data_ingestion.py       reads raw CSV, stratified train/test split (all 3 modules)
    data_transformation.py  tabular preprocessing + text/TF-IDF preprocessing (all 3 modules)
    model_trainer.py        trains & compares the 6 models, saves best (all 3 modules)
  pipeline/
    train_pipeline.py       orchestrates ingestion->transformation->training
    predict_pipeline.py     loads saved models, exposes predict_*() for the app
app/
  app.py               Flask entrypoint, one route per module (GET form + POST predict)
  templates/           Jinja2 HTML templates (base.html + one per module)
  static/style.css     minimal styling
notebooks/             EDA notebooks (exploration only, not production code)
data/raw/, data/processed/, models/, results/   per-module subfolders
```

## Notes / TODOs before this runs end-to-end
- [ ] Drop the 3 real datasets into `data/raw/<module>/`
- [ ] Confirm/adjust column names in `MODULE_CONFIG` (data_ingestion.py) and
      `TRANSFORM_CONFIG` (data_transformation.py) to match your actual datasets
- [ ] Stroke data is usually heavily imbalanced — consider adding class_weight="balanced"
      to models, or SMOTE, inside `model_trainer.py` if F1/recall are low
- [ ] Text cleaning in `data_transformation.py::clean_text()` is intentionally basic —
      upgrade with nltk/spacy stopwords + lemmatization once dataset is finalized
