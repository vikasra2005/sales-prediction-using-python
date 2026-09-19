from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def train_and_evaluate_model(df: pd.DataFrame, model_dir: str | Path) -> dict:
    """Train and compare multiple regression models."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    X = df[["TV", "Radio", "Newspaper"]]
    y = df["Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=None,
        ),
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, predictions)

        result = {
            "Model": name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2_Score": r2,
        }
        results.append(result)

        joblib.dump(model, model_dir / f"{name.lower().replace(' ', '_')}.joblib")

    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    best_model_name = results_df.iloc[0]["Model"]
    best_model = models[best_model_name]
    joblib.dump(best_model, model_dir / "best_sales_model.joblib")

    results_df.to_csv(model_dir / "model_comparison.csv", index=False)

    report_lines = [
        "Sales Prediction Model Report",
        "============================",
        f"Best Model: {best_model_name}",
        "",
        results_df.to_string(index=False),
    ]
    (model_dir / "sales_model_report.txt").write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "best_model": best_model_name,
        "model_comparison": results_df,
        "metrics": results_df.iloc[0].to_dict(),
    }
