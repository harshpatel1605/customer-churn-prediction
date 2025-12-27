from flask import Flask, render_template, request
import joblib
import numpy as np
import os
import sys

# -------------------------------------------------
# Path setup (so src/ is accessible)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

app = Flask(__name__)

# -------------------------------------------------
# Load trained model
# -------------------------------------------------
MODEL_PATH = os.path.join(BASE_DIR, "model", "churn_model.pkl")
model = joblib.load(MODEL_PATH)

# -------------------------------------------------
# FINAL FEATURE ORDER (MUST MATCH TRAINING)
# -------------------------------------------------
FEATURE_ORDER = [
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

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    form = request.form

    # -------------------------
    # User Inputs
    # -------------------------
    tenure = int(form["tenure"])
    monthly_charges = float(form["MonthlyCharges"])

    senior = int(form["SeniorCitizen"])
    partner = int(form["Partner"])
    dependents = int(form["Dependents"])
    phone_service = int(form["PhoneService"])
    multiple_lines = int(form["MultipleLines"])
    paperless = int(form["PaperlessBilling"])

    tech_support = int(form["TechSupport"])
    online_security = int(form["OnlineSecurity"])
    online_backup = int(form["OnlineBackup"])
    device_protection = int(form["DeviceProtection"])
    streaming_tv = int(form["StreamingTV"])
    streaming_movies = int(form["StreamingMovies"])

    contract = form["Contract"]              # Month-to-month / One year / Two year
    internet = form["InternetService"]        # DSL / Fiber optic / No
    payment = form["PaymentMethod"]           # Electronic / Credit card / Mailed

    # -------------------------
    # Derived Features
    # -------------------------
    early_customer = 1 if tenure < 12 else 0

    service_count = (
        tech_support +
        online_security +
        online_backup +
        device_protection +
        streaming_tv +
        streaming_movies
    )

    contract_risk = 1 if contract == "Month-to-month" else 0
    contract_tenure_risk = 1 if (contract == "Month-to-month" and tenure < 12) else 0

    # -------------------------
    # Initialize ALL features with 0
    # -------------------------
    features = {f: 0 for f in FEATURE_ORDER}

    # -------------------------
    # Fill known features
    # -------------------------
    features.update({
        "early_customer": early_customer,
        "SeniorCitizen": senior,
        "TechSupport": tech_support,
        "DeviceProtection": device_protection,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "tenure": tenure,
        "contract_tenure_risk": contract_tenure_risk,
        "MonthlyCharges": monthly_charges,
        "Dependents": dependents,
        "OnlineBackup": online_backup,
        "PaperlessBilling": paperless,
        "contract_risk": contract_risk,
        "OnlineSecurity": online_security,
        "Partner": partner,
        "MultipleLines": multiple_lines,
        "PhoneService": phone_service,
        "service_count": service_count
    })

    # -------------------------
    # One-Hot Encoded Fields
    # -------------------------
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

    # -------------------------
    # Final Model Input
    # -------------------------
    final_input = np.array(
        [features[f] for f in FEATURE_ORDER]
    ).reshape(1, -1)

    prediction = model.predict(final_input)[0]

    result = "Customer Will Churn ❌" if prediction == 1 else "Customer Will Stay ✅"

    return render_template("index.html", prediction=result)


# -------------------------------------------------
# Run App
# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
