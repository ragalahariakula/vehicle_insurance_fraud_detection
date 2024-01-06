import os
import pandas as pd
from sklearn.metrics import f1_score,accuracy_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np 
import joblib
from vifd.entity.config_entity import ModelEvaluationConfig
from vifd.utils.common import save_json
from pathlib import Path

class ModelEvaluation:
    def __init__(self, config:ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        f1 = f1_score(actual, pred)
        acc = accuracy_score(actual, pred)
        return f1, acc

    def log_into_mlflow(self):
        X_test = pd.read_csv(self.config.X_test_data_path)
        y_test = pd.read_csv(self.config.y_test_data_path)['fraud_reported_Y'].values
        model = joblib.load(self.config.model_path)

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            predicted_qualities = model.predict(X_test)

            f1, acc = self.eval_metrics(y_test, predicted_qualities)
            
            scores = {"f1": f1, "acc": acc}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_metric("f1", f1)
            mlflow.log_metric("acc", acc)

            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model, "model", registered_model_name="LogisticRegression")
            else:
                mlflow.sklearn.log_model(model, "model")
