# Run this project on your computer

Requires Python 3.10-3.12. Run all commands from inside this folder.

## Windows (PowerShell)
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements-app.txt
    python scripts/train_delay_model.py
    streamlit run app.py

## Mac / Linux
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements-app.txt
    python scripts/train_delay_model.py
    streamlit run app.py

The browser opens at http://localhost:8501.

## Notes
- `train_delay_model.py` rebuilds artifacts/model_pipeline.joblib for YOUR
  scikit-learn version (prevents pickle version errors).
- The map tab needs data/raw/airports.csv and airlines.csv from the Kaggle
  "2015 Flight Delays and Cancellations" dataset. Without them the app still
  runs; only the map is skipped.
- scripts/compute_report_metrics.py needs the full flights_clean.csv (5.7M
  rows). To try it on the sample: edit its data path to
  data/processed/flights_sample.csv.
