import os
import pandas as pd
from vifd import logger
from vifd.entity.config_entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self):
        try:
            data = pd.read_csv(self.config.unzip_data_dir)
            self.clean_data(data)
            validation_status = self.check_schema(data)
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}")
            return validation_status
        except Exception as e:
            raise e
        

    def clean_data(self, data):
        data.drop(['age', 'policy_number', 'policy_bind_date', 'insured_zip', 'insured_sex',
                   'insured_education_level', 'insured_occupation', 'insured_hobbies',
                   'insured_relationship', 'collision_type', 'property_damage', 'capital-gains',
                   'capital-loss', 'incident_date', 'incident_location', 'incident_hour_of_the_day',
                   '_c39'], axis=1, inplace=True)
        filtered_data = data[(data['police_report_available'] == '?') & (data['authorities_contacted'] != 'Other')]
        police_report_available_mode = data['police_report_available'].mode()[0]
        data.loc[data['police_report_available'] == '?', 'police_report_available'] = 'yes'
        data = data[data["umbrella_limit"] != -1000000]
        data[data["authorities_contacted"].isnull()]["fraud_reported"].value_counts()
        data = data.copy()  
        data.dropna(subset=['authorities_contacted'], inplace=True)
        data.to_csv(os.path.join(self.config.root_dir, "cleaned_data.csv"))
    def check_schema(self, data):
        all_cols = list(data.columns)
        all_schema = self.config.all_schema.keys()
        for col in all_cols:
            if col not in all_schema:
                return False
        return True

