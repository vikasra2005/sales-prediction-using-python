from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def generate_eda_report(df, output_dir: str | Path) -> None:
    """Generate plots for sales analysis and save them to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")

    plt.figure(figsize=(10, 6))
    sns.histplot(df["Sales"], kde=True, bins=20, color="steelblue")
    plt.title("Sales Distribution")
    plt.xlabel("Sales")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_dir / "sales_distribution.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 8))
    sns.pairplot(df, diag_kind="kde")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_relationships.png", dpi=300)
    plt.close()

    numeric_corr = df.corr(numeric_only=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=300)
    plt.close()

    print("Exploratory plots saved in outputs/")
