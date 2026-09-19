from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Advertising.csv"


def load_dataset(data_path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load and lightly clean the advertising dataset."""
    df = pd.read_csv(data_path)
    df.columns = [str(col).strip() for col in df.columns]

    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed|^$")]

    if "Sales" not in df.columns:
        raise ValueError("The dataset must include a Sales column.")

    df = df.dropna().reset_index(drop=True)
    return df
