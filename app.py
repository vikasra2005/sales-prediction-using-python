from pathlib import Path

from flask import Flask, render_template_string, request
import pandas as pd
import joblib

from src.data_loader import load_dataset


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = PROJECT_ROOT / "data" / "Advertising.csv"
MODEL_DIR = PROJECT_ROOT / "models"

BEST_MODEL_PATH = MODEL_DIR / "best_sales_model.joblib"
MODEL_COMPARISON_PATH = MODEL_DIR / "model_comparison.csv"


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# LOAD DATASET SUMMARY
# ============================================================

def get_dataset_summary() -> dict:
    """
    Load the advertising dataset and calculate
    basic statistics used by the dashboard.
    """

    df = load_dataset(DATA_PATH)

    return {
        "rows": len(df),
        "columns": list(df.columns),
        "avg_sales": round(float(df["Sales"].mean()), 2),
        "tv_corr": round(float(df["TV"].corr(df["Sales"])), 4),
        "radio_corr": round(float(df["Radio"].corr(df["Sales"])), 4),
        "newspaper_corr": round(
            float(df["Newspaper"].corr(df["Sales"])), 4
        ),
    }


# ============================================================
# LOAD EXISTING MACHINE LEARNING MODEL
# ============================================================

def load_existing_model():
    """
    Load the already-trained machine learning model.

    The model was trained previously and saved as:
    models/best_sales_model.joblib

    We do NOT train the model when a user opens the website.
    """

    return joblib.load(BEST_MODEL_PATH)


# ============================================================
# LOAD EXISTING MODEL METRICS
# ============================================================

def get_model_information():
    """
    Load the previously calculated model comparison results.
    """

    results_df = pd.read_csv(MODEL_COMPARISON_PATH)

    # The CSV was already sorted by RMSE during training.
    best_model = results_df.iloc[0]["Model"]

    metrics = results_df.iloc[0].to_dict()

    return best_model, metrics


# ============================================================
# HOME ROUTE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    summary = get_dataset_summary()

    # --------------------------------------------------------
    # Default input values
    # --------------------------------------------------------

    input_data = {
        "TV": 150,
        "Radio": 25,
        "Newspaper": 20,
    }

    prediction_text = None

    # --------------------------------------------------------
    # Load existing model and model information
    # --------------------------------------------------------

    model = load_existing_model()

    best_model, metrics = get_model_information()

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    if request.method == "POST":

        try:

            input_data = {
                "TV": float(request.form.get("tv", 0) or 0),
                "Radio": float(request.form.get("radio", 0) or 0),
                "Newspaper": float(
                    request.form.get("newspaper", 0) or 0
                ),
            }

            # Prepare input for machine learning model
            prediction_input = pd.DataFrame(
                [[
                    input_data["TV"],
                    input_data["Radio"],
                    input_data["Newspaper"],
                ]],
                columns=["TV", "Radio", "Newspaper"],
            )

            # Make prediction
            prediction = model.predict(prediction_input)[0]

            prediction_text = (
                f"Estimated Sales: {prediction:.2f} units"
            )

        except Exception as error:

            prediction_text = (
                f"Prediction error: {str(error)}"
            )

    # ========================================================
    # HTML DASHBOARD
    # ========================================================

    html = """
    <!doctype html>

    <html lang="en">

    <head>

        <meta charset="utf-8">

        <meta name="viewport"
              content="width=device-width, initial-scale=1">

        <title>Sales Prediction Dashboard</title>

        <style>

            * {
                box-sizing: border-box;
            }

            body {
                font-family: Arial, sans-serif;
                background: #f4f7fb;
                color: #1f2937;
                margin: 0;
                padding: 30px;
            }

            .container {
                max-width: 900px;
                margin: auto;
            }

            .card {
                background: white;
                border-radius: 12px;
                box-shadow:
                    0 4px 18px rgba(0, 0, 0, 0.06);
                padding: 25px;
                margin-bottom: 20px;
            }

            h1,
            h2 {
                margin-top: 0;
            }

            h1 {
                color: #17324d;
            }

            .subtitle {
                color: #4b5563;
                margin-bottom: 25px;
            }

            .grid {
                display: grid;
                grid-template-columns:
                    repeat(auto-fit, minmax(180px, 1fr));
                gap: 14px;
            }

            .stat {
                background: #eef4ff;
                border-radius: 10px;
                padding: 16px;
            }

            .stat strong {
                display: block;
                font-size: 24px;
                margin-top: 8px;
                color: #17324d;
            }

            form {
                display: grid;
                grid-template-columns:
                    repeat(auto-fit, minmax(180px, 1fr));
                gap: 16px;
                align-items: end;
            }

            label {
                display: block;
                margin-bottom: 6px;
                font-weight: 600;
            }

            input,
            button {
                width: 100%;
                padding: 12px;
                border-radius: 8px;
                border: 1px solid #d1d5db;
                font-size: 16px;
            }

            input:focus {
                outline: none;
                border-color: #2563eb;
            }

            button {
                background: #2563eb;
                color: white;
                border: none;
                cursor: pointer;
                font-weight: 600;
            }

            button:hover {
                background: #1d4ed8;
            }

            .result {
                background: #ecfdf5;
                border-left: 4px solid #10b981;
                padding: 16px;
                margin-top: 20px;
                border-radius: 8px;
                color: #065f46;
            }

            .small {
                color: #4b5563;
            }

            .metric {
                padding: 10px 0;
                border-bottom: 1px solid #e5e7eb;
            }

            .metric:last-child {
                border-bottom: none;
            }

            .footer {
                text-align: center;
                color: #6b7280;
                font-size: 14px;
                margin-top: 30px;
            }

            @media (max-width: 600px) {

                body {
                    padding: 15px;
                }

                .card {
                    padding: 18px;
                }

                h1 {
                    font-size: 26px;
                }

            }

        </style>

    </head>


    <body>

        <div class="container">


            <!-- =================================================
                 HEADER / DATASET SUMMARY
            ================================================== -->

            <div class="card">

                <h1>
                    Sales Prediction Dashboard
                </h1>

                <p class="subtitle">
                    Advertising dataset analysis for
                    TV, Radio and Newspaper performance.
                </p>


                <div class="grid">

                    <div class="stat">

                        Dataset Rows

                        <strong>
                            {{ summary['rows'] }}
                        </strong>

                    </div>


                    <div class="stat">

                        Average Sales

                        <strong>
                            {{ summary['avg_sales'] }}
                        </strong>

                    </div>


                    <div class="stat">

                        TV Correlation

                        <strong>
                            {{ summary['tv_corr'] }}
                        </strong>

                    </div>


                    <div class="stat">

                        Radio Correlation

                        <strong>
                            {{ summary['radio_corr'] }}
                        </strong>

                    </div>


                    <div class="stat">

                        Newspaper Correlation

                        <strong>
                            {{ summary['newspaper_corr'] }}
                        </strong>

                    </div>

                </div>

            </div>


            <!-- =================================================
                 SALES PREDICTION
            ================================================== -->

            <div class="card">

                <h2>
                    Predict Sales
                </h2>

                <p class="small">
                    Enter advertising spending values
                    to estimate sales.
                </p>


                <form method="post">


                    <div>

                        <label for="tv">
                            TV Spend
                        </label>

                        <input
                            type="number"
                            step="0.1"
                            id="tv"
                            name="tv"
                            value="{{ input_data['TV'] }}"
                            required
                        >

                    </div>


                    <div>

                        <label for="radio">
                            Radio Spend
                        </label>

                        <input
                            type="number"
                            step="0.1"
                            id="radio"
                            name="radio"
                            value="{{ input_data['Radio'] }}"
                            required
                        >

                    </div>


                    <div>

                        <label for="newspaper">
                            Newspaper Spend
                        </label>

                        <input
                            type="number"
                            step="0.1"
                            id="newspaper"
                            name="newspaper"
                            value="{{ input_data['Newspaper'] }}"
                            required
                        >

                    </div>


                    <div>

                        <button type="submit">
                            Predict
                        </button>

                    </div>


                </form>


                {% if prediction_text %}

                    <div class="result">

                        <strong>
                            {{ prediction_text }}
                        </strong>

                    </div>

                {% endif %}

            </div>


            <!-- =================================================
                 MODEL SUMMARY
            ================================================== -->

            <div class="card">

                <h2>
                    Model Summary
                </h2>


                <div class="metric">

                    <strong>
                        Best Model:
                    </strong>

                    {{ best_model }}

                </div>


                <div class="metric">

                    <strong>
                        MAE:
                    </strong>

                    {{ '%.4f' % metrics['MAE'] }}

                </div>


                <div class="metric">

                    <strong>
                        RMSE:
                    </strong>

                    {{ '%.4f' % metrics['RMSE'] }}

                </div>


                <div class="metric">

                    <strong>
                        R² Score:
                    </strong>

                    {{ '%.4f' % metrics['R2_Score'] }}

                </div>

            </div>


            <!-- =================================================
                 FOOTER
            ================================================== -->

            <div class="footer">

                Machine Learning Sales Prediction Project

            </div>


        </div>

    </body>

    </html>
    """

    return render_template_string(
        html,
        summary=summary,
        best_model=best_model,
        metrics=metrics,
        input_data=input_data,
        prediction_text=prediction_text,
    )


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )