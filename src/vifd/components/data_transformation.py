import os
from vifd import logger
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTENC
from vifd.entity.config_entity import DataTransformationConfig
import pickle

class DataTransformation:
    def __init__(self, config):
        self.config = config
        self.encoder_file = os.path.join(self.config.root_dir, 'encoder.pkl')
        self.scaler_file = os.path.join(self.config.root_dir, 'scaler.pkl')

    def load_data(self):
        data = pd.read_csv(self.config.data_path)
        return data
    
    def encoder_ip_data_columns(self,data):
        k=data.copy()
        m = ['policy_state', 'policy_csl', 'incident_type',
                                'incident_severity', 'authorities_contacted',
                                'incident_state', 'incident_city',
                                'police_report_available', 'auto_make', 'auto_model']
        enc1 = OneHotEncoder(handle_unknown='ignore', drop='first')
        cat_enc_ip_data = pd.DataFrame(enc1.fit_transform(k[m]).toarray())
        cat_enc_ip_data.columns = enc1.get_feature_names_out()
        
        with open(self.encoder_file, 'wb') as encoder_file:
            pickle.dump(enc1, encoder_file)

    def encode_categorical_columns(self, data):
        categorical_columns = ['policy_state', 'policy_csl', 'incident_type',
                                'incident_severity', 'authorities_contacted',
                                'incident_state', 'incident_city',
                                'police_report_available', 'auto_make', 'auto_model', 'fraud_reported']
        
        enc = OneHotEncoder(handle_unknown='ignore', drop='first')
        cat_enc_data = pd.DataFrame(enc.fit_transform(data[categorical_columns]).toarray())
        cat_enc_data.columns = enc.get_feature_names_out()
            
        return cat_enc_data

    def preprocess_data(self, data, cat_enc_data):
        numerical_columns = ['months_as_customer', 'policy_deductable', 'policy_annual_premium',
                              'umbrella_limit', 'number_of_vehicles_involved', 'bodily_injuries',
                              'witnesses', 'total_claim_amount', 'injury_claim', 'property_claim',
                              'vehicle_claim', 'auto_year']

        df = pd.concat([data[numerical_columns], cat_enc_data], axis=1)
        df.dropna(inplace=True)
        return df

    def scale_numerical_features(self, X_train, X_test, numerical_columns):
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train[numerical_columns]),
                                      columns=numerical_columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(scaler.fit_transform(X_test[numerical_columns]),
                                     columns=numerical_columns, index=X_test.index)
        
        with open(self.scaler_file, 'wb') as scaler_file:
            pickle.dump(scaler, scaler_file)

        for col in numerical_columns:
            X_train[col] = X_train_scaled[col]
            X_test[col] = X_test_scaled[col]

    def train_test_split(self, df):
        X = df.drop('fraud_reported_Y', axis=1)
        y = df['fraud_reported_Y']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        return X_train, X_test, y_train, y_test

    def save_data(self, X_train, X_test, y_train, y_test, part_name):
        X_train.to_csv(os.path.join(self.config.root_dir, f"{part_name}_X_train.csv"), index=False)
        X_test.to_csv(os.path.join(self.config.root_dir, f"{part_name}_X_test.csv"), index=False)
        y_train.to_csv(os.path.join(self.config.root_dir, f"{part_name}_y_train.csv"), index=False)
        y_test.to_csv(os.path.join(self.config.root_dir, f"{part_name}_y_test.csv"), index=False)

    def train_test_splitting(self):
        try:
            data = self.load_data()
            self.encoder_ip_data_columns(data)
            cat_enc_data = self.encode_categorical_columns(data)
            df = self.preprocess_data(data, cat_enc_data)

            numerical_columns = ['months_as_customer', 'policy_deductable', 'policy_annual_premium',
                                  'umbrella_limit', 'number_of_vehicles_involved', 'bodily_injuries',
                                  'witnesses', 'total_claim_amount', 'injury_claim', 'property_claim',
                                  'vehicle_claim', 'auto_year']

            X_train, X_test, y_train, y_test = self.train_test_split(df)

            self.scale_numerical_features(X_train, X_test, numerical_columns)

            self.save_data(X_train, X_test, y_train, y_test, "split")
            
            sm = SMOTENC(categorical_features=np.arange(80, 90), random_state=123, sampling_strategy=.6)

            X_train_re, y_train_re = sm.fit_resample(X_train, y_train)

            self.save_data(X_train_re, X_test, y_train_re, y_test, "balanced_split")

            logger.info("Data split and balanced successfully.")
        except Exception as e:
            raise e
