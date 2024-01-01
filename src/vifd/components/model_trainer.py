import pandas as pd
import os
from vifd import logger
from sklearn.linear_model import LogisticRegression
import joblib
from vifd.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        X_train = pd.read_csv(self.config.X_train_data_path)
        y_train = pd.read_csv(self.config.y_train_data_path)['fraud_reported_Y'].values
        X_test = pd.read_csv(self.config.X_test_data_path)
        y_test = pd.read_csv(self.config.y_test_data_path)['fraud_reported_Y'].values

        log_reg = LogisticRegression(random_state=42)
        log_reg.fit(X_train, y_train)

        joblib.dump(log_reg, os.path.join(self.config.root_dir, self.config.model_name))
