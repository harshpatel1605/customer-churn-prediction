import numpy as np

FEATURE_ORDER = [
    "early_customer",
    "SeniorCitizen",
    "TechSupport_Yes",
    "DeviceProtection_Yes",
    "StreamingTV_Yes",
    "StreamingMovies_Yes",
    "tenure",
    "contract_tenure_risk",
    "MonthlyCharges",
    "Dependents",
    "OnlineBackup_Yes",
    "PaymentMethod_Credit card (automatic)",
    "InternetService_No",
    "PaperlessBilling",
    "contract_risk",
    "OnlineSecurity_Yes",
    "Partner",
    "MultipleLines_Yes",
    "InternetService_Fiber optic",
    "PhoneService",
    "PaymentMethod_Electronic check",
    "service_count",
    "PaymentMethod_Mailed check"
]


def build_features(form):
    """
    Takes Flask request.form and returns
    numpy array in correct feature order
    """

    # ---- User Inputs ----
    tenure = int(form["tenure"])
    monthly_charges = float(form["MonthlyCharges"])

    senior = int(form["SeniorCitizen"])
    partner = int(form["Partner"])
    dependents = int(form["Dependents"])
    phone_service = int(form["PhoneService"])
    multiple_lines = int(form["MultipleLines"])
    paperless = int(form["PaperlessBilling"])

    contract = form["Contract"]
    internet = form["InternetService"]
    payment = form["PaymentMethod"]

    tech_support = int(form["TechSupport"])
    online_security = int(form["OnlineSecurity"])
    online_backup = int(form["OnlineBackup"])
    device_protection = int(form["DeviceProtection"])
    streaming_tv = int(form["StreamingTV"])
    streaming_movies = int(form["StreamingMovies"])

    # ---- Derived Features ----
    early_customer = 1 if tenure < 12 else 0

    service_count = sum([
        tech_support, online_security, online_backup,
        device_protection, streaming_tv, streaming_movies
    ])

    contract_risk = 1 if contract == "Month-to-month" else 0
    contract_tenure_risk = 1 if (contract == "Month-to-month" and tenure < 12) else 0

    # ---- Initialize All Features with 0 ----
    features = {f: 0 for f in FEATURE_ORDER}

    # ---- Fill Required Features ----
    features.update({
        "early_customer": early_customer,
        "SeniorCitizen": senior,
        "TechSupport_Yes": tech_support,
        "DeviceProtection_Yes": device_protection,
        "StreamingTV_Yes": streaming_tv,
        "StreamingMovies_Yes": streaming_movies,
        "tenure": tenure,
        "contract_tenure_risk": contract_tenure_risk,
        "MonthlyCharges": monthly_charges,
        "Dependents": dependents,
        "OnlineBackup_Yes": online_backup,
        "PaperlessBilling": paperless,
        "contract_risk": contract_risk,
        "OnlineSecurity_Yes": online_security,
        "Partner": partner,
        "MultipleLines_Yes": multiple_lines,
        "PhoneService": phone_service,
        "service_count": service_count
    })

    # ---- One-Hot Logic ----
    if internet == "No":
        features["InternetService_No"] = 1
    elif internet == "Fiber optic":
        features["InternetService_Fiber optic"] = 1

    if payment == "Electronic check":
        features["PaymentMethod_Electronic check"] = 1
    elif payment == "Credit card (automatic)":
        features["PaymentMethod_Credit card (automatic)"] = 1
    elif payment == "Mailed check":
        features["PaymentMethod_Mailed check"] = 1

    # ---- Final Ordered Array ----
    return np.array([features[f] for f in FEATURE_ORDER]).reshape(1, -1)
