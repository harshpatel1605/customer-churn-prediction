# 📉 Customer Churn Prediction: End-to-End ML System

An end-to-end **machine learning system** that predicts the likelihood of customer churn using real-world telecom data. Designed for scalability, interpretability, and real-time usage — suitable for both demos and production environments.

Built with **FastAPI, Docker, DVC, MLflow**, and deployed on **AWS EC2 with ECR**, following best practices in MLOps and ML engineering.

---

## 🚀 Project Highlights

- 🔍 **Churn Prediction**: Real-time risk prediction using customer account inputs.
- 🧠 **ML Models**: XGBoost (base + Optuna-tuned) with threshold optimization for recall.
- 🧪 **MLOps Integration**: Full lifecycle tracking via MLflow, hyperparameter tuning with Optuna.
- 🧱 **Modular ML Pipelines**: Clean, reusable code structure for ingestion, transformation, training, and evaluation.
- ⚡ **REST API**: FastAPI-powered backend for quick inference.
- 📦 **Dockerized Deployment**: Easy containerization for local or cloud use.
- 🔁 **CI/CD Pipeline**: Automated build and deployment via GitHub Actions + AWS ECR + EC2.
- 📊 **Data Versioning**: Full pipeline reproducibility using DVC.

---

## 📊 Dataset Overview

- **Source**: [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Target Variable**: `Churn` (binary: Yes or No)
- **Features Used**:
  - Demographics: gender, senior citizen, partner, dependents
  - Services: phone, internet, streaming, security, backup, tech support
  - Billing: contract type, payment method, monthly charges, total charges
  - Account: tenure, paperless billing

---

## 🔬 Data Preprocessing & Feature Engineering

- Binary encoding for Yes/No and Male/Female columns
- One-hot encoding for multi-category columns (contract, payment method, internet service, etc.)
- Collapsing redundant dummy columns (`No internet service`, `No phone service`) into single flags
- VIF-based multicollinearity check
- Column alignment at inference to match training feature set exactly

---

## 🧠 Machine Learning Models

| Model | Description |
|---|---|
| XGBoost Base | Default hyperparameters with class imbalance handling via `scale_pos_weight` |
| XGBoost Tuned | Optuna-optimized across 30 trials, maximizing recall at threshold 0.25 |

**Optimization**:
- Hyperparameter tuning with **Optuna** (30 trials, recall-focused objective)
- Custom decision threshold (0.25) to minimize false negatives
- Experiment tracking with **MLflow**
- Model versioning and artifact logging

---

## ⚙️ ML Pipeline Stages (DVC)

| Stage | Description |
|---|---|
| `data_ingestion` | Loads raw data from source into `artifacts/` |
| `data_transformation` | Encodes, cleans, and prepares features |
| `model_training` | Trains base and tuned XGBoost models |
| `model_evaluation` | Evaluates models, saves metrics and confusion matrices |

Run the full pipeline with:
```bash
dvc repro
```

---

## ⚙️ API Endpoints (FastAPI)

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the prediction form UI |
| `/api/predict` | POST | Returns churn prediction and probability from input |

---

## 🖥️ Frontend

- Single-page form for inputting customer details
- Real-time prediction result shown on the same page (no redirect)
- Displays churn label, probability percentage, and visual probability bar

---

## 🚢 Deployment

- **Containerized** using Docker
- **CI/CD** via GitHub Actions with three stages:
  - `Continuous Integration` — lint and test
  - `Continuous Delivery` — build and push Docker image to AWS ECR
  - `Continuous Deployment` — pull and run latest image on AWS EC2 (self-hosted runner)
- **Cloud**: AWS EC2 (self-hosted runner) + AWS ECR (container registry)

---

## 🛠 Tech Stack

| Layer | Tools |
|---|---|
| **ML** | Scikit-learn, XGBoost, Optuna |
| **Experiment Tracking** | MLflow |
| **Data Versioning** | DVC |
| **Backend** | Python, FastAPI, Uvicorn |
| **Frontend** | HTML, CSS, JavaScript |
| **Deployment** | Docker, AWS EC2, AWS ECR |
| **CI/CD** | GitHub Actions |

---

## 📁 Project Structure

```
customer-churn-prediction/
├── .github/
│   └── workflows/
│       └── main.yaml                  # CI/CD pipeline
├── artifacts/                         # DVC tracked outputs
│   ├── data_ingestion/
│   │   └── raw_data.csv
│   ├── data_transformation/
│   │   └── transformed_data.csv
│   ├── model_training/
│   │   ├── base_model.pkl
│   │   ├── best_params.json
│   │   └── tuned_model.pkl
│   └── model_evaluation/
│       ├── scores.json
│       ├── XGBoost.png
│       └── XGBoost_Tuned.png
├── config/
│   └── config.yaml                    # Central configuration
├── model/
│   └── tuned_model.pkl                # Model committed to GitHub
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_data_transformation.ipynb
│   └── 03_model_training.ipynb
├── src/
│   └── CustomerChurnPrediction/
│       ├── components/
│       │   ├── data_ingestion.py
│       │   ├── data_transformation.py
│       │   ├── model_training.py
│       │   └── model_evaluation.py
│       ├── pipelines/
│       │   ├── stage_01_data_ingestion.py
│       │   ├── stage_02_data_transformation.py
│       │   ├── stage_03_model_training.py
│       │   ├── stage_04_model_evaluation.py
│       │   └── prediction_pipeline.py
│       ├── config/
│       │   └── configuration.py
│       ├── entity/
│       │   └── config_entity.py
│       ├── constants/
│       │   └── __init__.py
│       └── utils/
│           ├── common.py
│           ├── logger.py
│           └── exception.py
├── templates/
│   └── index.html                     # Frontend UI
├── app.py                             # FastAPI application
├── main.py                            # Runs full training pipeline
├── dvc.yaml                           # DVC pipeline definition
├── dvc.lock                           # DVC lock file
├── Dockerfile                         # Container definition
├── requirements.txt
├── setup.py
└── README.md
```
---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve the project — new models, UI improvements, or backend optimization — feel free to open a pull request or create an issue.

---

## 📬 Contact

- **Email**: harshpatel16052005.email@gmail.com
- **LinkedIn**: [Harsh Patel](https://linkedin.com/in/harsh-patel-581352358)
- **GitHub**: [Harsh Patel](https://github.com/harshpatel1605)

---

## 📎 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.