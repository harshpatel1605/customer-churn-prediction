import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create final features for churn model training
    """
    df = df.copy()

    # -------------------------
    # Binary mappings
    # -------------------------
    binary_map = {"Yes": 1, "No": 0}

    binary_cols = [
        "Partner", "Dependents", "PhoneService",
        "PaperlessBilling"
    ]

    for col in binary_cols:
        df[col] = df[col].map(binary_map)

    # -------------------------
    # Internet services cleanup
    # -------------------------
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]

    df[service_cols] = df[service_cols].replace(
        {"No internet service": "No"}
    )

    for col in service_cols:
        df[col] = df[col].map(binary_map)

    # -------------------------
    # Derived Features
    # -------------------------
    df["early_customer"] = (df["tenure"] < 12).astype(int)

    df["service_count"] = df[service_cols].sum(axis=1)

    df["contract_risk"] = (df["Contract"] == "Month-to-month").astype(int)

    df["contract_tenure_risk"] = (
        (df["Contract"] == "Month-to-month") & (df["tenure"] < 12)
    ).astype(int)

    # -------------------------
    # One-Hot Encoding
    # -------------------------
    df = pd.get_dummies(
        df,
        columns=["PaymentMethod", "InternetService"],
        drop_first=False
    )

    # -------------------------
    # FINAL FEATURE LIST (ORDER FIXED)
    # -------------------------
    final_features = [
        "early_customer",
        "SeniorCitizen",
        "TechSupport",
        "DeviceProtection",
        "StreamingTV",
        "StreamingMovies",
        "tenure",
        "contract_tenure_risk",
        "MonthlyCharges",
        "Dependents",
        "OnlineBackup",
        "PaymentMethod_Credit card (automatic)",
        "InternetService_No",
        "PaperlessBilling",
        "contract_risk",
        "OnlineSecurity",
        "Partner",
        "MultipleLines",
        "InternetService_Fiber optic",
        "PhoneService",
        "PaymentMethod_Electronic check",
        "service_count",
        "PaymentMethod_Mailed check"
    ]

    return df[final_features]
