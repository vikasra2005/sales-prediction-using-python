from pathlib import Path

from flask import Flask, render_template_string, request

from src.data_loader import load_dataset
from src.model_pipeline import train_and_evaluate_model
from src.predictor import predict_sales

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "Advertising.csv"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

app = Flask(__name__)


def get_dataset_summary() -> dict:
    df = load_dataset(DATA_PATH)
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "avg_sales": round(float(df["Sales"].mean()), 2),
        "tv_corr": round(float(df["TV"].corr(df["Sales"])), 4),
        "radio_corr": round(float(df["Radio"].corr(df["Sales"])), 4),
        "newspaper_corr": round(float(df["Newspaper"].corr(df["Sales"])), 4),
    }


@app.route("/", methods=["GET", "POST"])
def home():
    summary = get_dataset_summary()
    prediction_text = None
    input_data = {"TV": 150, "Radio": 25, "Newspaper": 20}

    if request.method == "POST":
        input_data = {
            "TV": float(request.form.get("tv", 0) or 0),
            "Radio": float(request.form.get("radio", 0) or 0),
            "Newspaper": float(request.form.get("newspaper", 0) or 0),
        }
        prediction = predict_sales(
            input_data["TV"],
            input_data["Radio"],
            input_data["Newspaper"],
            MODEL_DIR / "best_sales_model.joblib",
        )
        prediction_text = f"Estimated Sales: {prediction:.2f} units"

    model_info = train_and_evaluate_model(load_dataset(DATA_PATH), MODEL_DIR)
    best_model = model_info["best_model"]
    metrics = model_info["metrics"]

    html = """
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Sales Prediction Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; background: #f4f7fb; color: #1f2937; margin: 0; padding: 30px; }
            .container { max-width: 900px; margin: auto; }
            .card { background: white; border-radius: 12px; box-shadow: 0 4px 18px rgba(0, 0, 0, 0.06); padding: 25px; margin-bottom: 20px; }
            h1, h2 { margin-top: 0; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; }
            .stat { background: #eef4ff; border-radius: 10px; padding: 16px; }
            .stat strong { display: block; font-size: 24px; margin-top: 8px; }
            form { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; }
            input, button { width: 100%; padding: 12px; border-radius: 8px; border: 1px solid #d1d5db; font-size: 16px; }
            button { background: #2563eb; color: white; border: none; cursor: pointer; }
            .result { background: #ecfdf5; border-left: 4px solid #10b981; padding: 16px; margin-top: 20px; border-radius: 8px; }
            .small { color: #4b5563; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card">
                <h1>Sales Prediction Dashboard</h1>
                <p class="small">Advertising dataset analysis for TV, Radio and Newspaper performance.</p>
                <div class="grid">
                    <div class="stat">Dataset rows<strong>{{ summary['rows'] }}</strong></div>
                    <div class="stat">Avg Sales<strong>{{ summary['avg_sales'] }}</strong></div>
                    <div class="stat">TV correlation<strong>{{ summary['tv_corr'] }}</strong></div>
                    <div class="stat">Radio correlation<strong>{{ summary['radio_corr'] }}</strong></div>
                    <div class="stat">Newspaper correlation<strong>{{ summary['newspaper_corr'] }}</strong></div>
                </div>
            </div>

            <div class="card">
                <h2>Predict Sales</h2>
                <form method="post">
                    <div>
                        <label>TV Spend</label>
                        <input type="number" step="0.1" name="tv" value="{{ input_data['TV'] }}">
                    </div>
                    <div>
                        <label>Radio Spend</label>
                        <input type="number" step="0.1" name="radio" value="{{ input_data['Radio'] }}">
                    </div>
                    <div>
                        <label>Newspaper Spend</label>
                        <input type="number" step="0.1" name="newspaper" value="{{ input_data['Newspaper'] }}">
                    </div>
                    <div>
                        <label>&nbsp;</label>
                        <button type="submit">Predict</button>
                    </div>
                </form>
                {% if prediction_text %}
                    <div class="result">
                        <strong>{{ prediction_text }}</strong>
                    </div>
                {% endif %}
            </div>

            <div class="card">
                <h2>Model Summary</h2>
                <p><strong>Best Model:</strong> {{ best_model }}</p>
                <p><strong>MAE:</strong> {{ '%.4f' % metrics['MAE'] }}</p>
                <p><strong>RMSE:</strong> {{ '%.4f' % metrics['RMSE'] }}</p>
                <p><strong>R² Score:</strong> {{ '%.4f' % metrics['R2_Score'] }}</p>
            </div>
        </div>
    </body>
    </html>
    """
    return render_template_string(html, summary=summary, best_model=best_model, metrics=metrics, input_data=input_data, prediction_text=prediction_text)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
