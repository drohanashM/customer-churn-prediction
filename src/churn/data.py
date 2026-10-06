import pandas as pd

from .config import DATA_PATH, ID_COL, TARGET


def load_clean(path=DATA_PATH) -> pd.DataFrame:
    """Load the raw Telco CSV and fix its known quirks."""
    df = pd.read_csv(path)
    # TotalCharges has 11 blank strings (brand-new customers with tenure 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df.drop(columns=[ID_COL])


def split_xy(df: pd.DataFrame):
    return df.drop(columns=[TARGET]), df[TARGET]
