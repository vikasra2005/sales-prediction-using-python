from pathlib import Path

from src.data_loader import load_dataset
from src.exploratory_analysis import generate_eda_report
from src.model_pipeline import train_and_evaluate_model

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "Advertising.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"


def write_summary_report(df, results) -> None:
    """Write a business-friendly summary report to the outputs folder."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    correlation = df[["TV", "Radio", "Newspaper"]].corrwith(df["Sales"]).sort_values(ascending=False)

    report = [
        "Sales Analysis Report",
        "====================",
        f"Dataset rows: {len(df)}",
        f"Dataset columns: {list(df.columns)}",
        "",
        "Channel influence on sales:",
        correlation.to_string(),
        "",
        "Best predictive model:",
        f"- {results['best_model']}",
        "",
        "Model comparison:",
        results["model_comparison"].to_string(index=False),
    ]

    (OUTPUT_DIR / "sales_analysis_report.txt").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    print("Loading dataset...")
    df = load_dataset(DATA_PATH)
    print(f"Dataset shape: {df.shape}")

    print("Generating exploratory visualizations...")
    generate_eda_report(df, OUTPUT_DIR)

    print("Training and evaluating models...")
    results = train_and_evaluate_model(df, MODEL_DIR)
    write_summary_report(df, results)
    print("Best model:", results["best_model"])
    print(results["model_comparison"].to_string(index=False))

    print("\nProject completed successfully.")
    print(f"Check outputs in: {OUTPUT_DIR}")
    print(f"Check model files in: {MODEL_DIR}")


if __name__ == "__main__":
    main()
