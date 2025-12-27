import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic data cleaning before feature engineering
    """
    df = df.copy()

    # Drop customer ID if present
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    # Convert TotalCharges safely
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Handle missing values
    df.fillna(0, inplace=True)

    return df
