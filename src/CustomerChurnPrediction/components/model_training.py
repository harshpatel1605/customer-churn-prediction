import os
import json
import sys
import time
import joblib
import optuna
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from CustomerChurnPrediction.utils.exception import CustomException
from CustomerChurnPrediction.utils.logger import logger
from CustomerChurnPrediction.entity.config_entity import ModelTrainingConfig


class ModelTraining:
    """Trains a base XGBoost model and a tuned XGBoost model (via Optuna), saving both to disk."""

    THRESHOLD = 0.25

    def __init__(self, config: ModelTrainingConfig, n_trials: int = 30):
        """Sets up config and trial count."""
        self.config = config
        self.n_trials = n_trials

    def get_training_data(self) -> pd.DataFrame:
        """Loads the transformed training CSV."""
        logger.info(f"Loading training data from: {self.config.training_data_path}")
        return pd.read_csv(self.config.training_data_path)

    def split_data(self, df: pd.DataFrame, target_col: str = "Churn", test_size: float = 0.2):
        """Splits the data into train and test sets."""
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        logger.info(f"Split data: {X_train.shape[0]} train rows, {X_test.shape[0]} test rows")
        return X_train, X_test, y_train, y_test

    def get_scale_pos_weight(self, y_train) -> float:
        """Calculates the class weight used to correct for churn-class imbalance."""
        return (y_train == 0).sum() / (y_train == 1).sum()

    @staticmethod
    def save_model(model: XGBClassifier, path: str) -> None:
        """Saves a model to the given path, creating the folder if needed."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)
        logger.info(f"Model saved to {path}")

    @staticmethod
    def save_model_for_github(model: XGBClassifier, filename: str = "tuned_model.pkl") -> None:
        """Saves a model into a small 'model/' folder meant to be committed to GitHub, since artifacts/ is not pushed."""
        github_dir = "model"
        os.makedirs(github_dir, exist_ok=True)
        path = os.path.join(github_dir, filename)
        joblib.dump(model, path)
        logger.info(f"Model saved for GitHub at {path}")
    
    @staticmethod
    def save_best_params(best_params:dict):
        with open("artifacts/model_training/best_params.json","w") as f:
            json.dump(best_params, f, indent=4)

    def train_base_model(self, X_train, y_train) -> XGBClassifier:
        """Trains the base XGBoost model and saves it to disk."""
        scale_pos_weight = self.get_scale_pos_weight(y_train)

        params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "logloss",
        }

        start_train = time.time()
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        logger.info(f"Base model trained in {time.time() - start_train:.2f} seconds")

        self.save_model(model, self.config.base_model_path)

        return model

    def tune_model(self, X_train, y_train, X_test, y_test) -> XGBClassifier:
        """Runs Optuna to find the best hyperparameters, then trains and saves the final tuned model."""
        scale_pos_weight = self.get_scale_pos_weight(y_train)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 300, 800),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
                "random_state": 42,
                "n_jobs": -1,
                "scale_pos_weight": scale_pos_weight,
                "eval_metric": "logloss",
            }

            model = XGBClassifier(**params)
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]
            y_pred = (proba >= self.THRESHOLD).astype(int)
            return recall_score(y_test, y_pred, pos_label=1)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)

        self.save_best_params(study.best_params)

        logger.info(f"Best trial: {study.best_trial.number} | Recall: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        best_params = {
            **study.best_params,
            "random_state": 42,
            "n_jobs": -1,
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "logloss",
        }

        start_train = time.time()
        best_model = XGBClassifier(**best_params)
        best_model.fit(X_train, y_train)
        logger.info(f"Tuned model trained in {time.time() - start_train:.2f} seconds")

        self.save_model(best_model, self.config.tuned_model_path)
        self.save_model_for_github(best_model)

        return best_model

    def run_training(self) -> None:
        """Runs the full training pipeline: load data, split, train base model, tune model."""
        df = self.get_training_data()
        X_train, X_test, y_train, y_test = self.split_data(df)

        self.train_base_model(X_train, y_train)
        self.tune_model(X_train, y_train, X_test, y_test)