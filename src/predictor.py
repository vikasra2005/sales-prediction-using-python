from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_sales_model.joblib"


def load_model(model_path: str | Path = MODEL_PATH):
    """Load the trained sales prediction model."""
    return joblib.load(model_path)


def predict_sales(tv: float, radio: float, newspaper: float, model_path: str | Path = MODEL_PATH) -> float:
    """Predict sales from TV, Radio, and Newspaper spend values."""
    model = load_model(model_path)
    features = pd.DataFrame([[tv, radio, newspaper]], columns=["TV", "Radio", "Newspaper"])
    prediction = model.predict(features)[0]
    return float(prediction)
