from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
import pickle
from src.vifd.pipeline.prediction import PredictionPipeline

app = Flask(__name__)  # initializing a flask app

# Load the encoder
encoder_path = "artifacts/data_transformation/encoder.pkl"
with open(encoder_path, 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)

# Load the scaler
scaler_path = "artifacts/data_transformation/scaler.pkl"
with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define categorical and numerical columns
categorical_columns = ['policy_state', 'policy_csl', 'incident_type',
                        'incident_severity', 'authorities_contacted',
                        'incident_state', 'incident_city',
                        'police_report_available', 'auto_make', 'auto_model']

numerical_columns = ['months_as_customer', 'policy_deductable', 'policy_annual_premium',
                      'umbrella_limit', 'number_of_vehicles_involved', 'bodily_injuries',
                      'witnesses', 'total_claim_amount', 'injury_claim', 'property_claim',
                      'vehicle_claim', 'auto_year']

@app.route('/', methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")

@app.route('/train', methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!"

@app.route('/predict', methods=['POST', 'GET'])  # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            months_as_customer = int(request.form['months_as_customer'])
            policy_state = request.form['policy_state']
            policy_csl = request.form['policy_csl']
            policy_deductable = int(request.form['policy_deductable'])
            policy_annual_premium = float(request.form['policy_annual_premium'])
            umbrella_limit = int(request.form['umbrella_limit'])
            incident_type = request.form['incident_type']
            incident_severity = request.form['incident_severity']
            authorities_contacted = request.form['authorities_contacted']
            incident_state = request.form['incident_state']
            incident_city = request.form['incident_city']
            number_of_vehicles_involved = int(request.form['number_of_vehicles_involved'])
            bodily_injuries = int(request.form['bodily_injuries'])
            witnesses = int(request.form['witnesses'])
            police_report_available = request.form['police_report_available']
            total_claim_amount = int(request.form['total_claim_amount'])
            injury_claim = int(request.form['injury_claim'])
            property_claim = int(request.form['property_claim'])
            vehicle_claim = int(request.form['vehicle_claim'])
            auto_make = request.form['auto_make']
            auto_model = request.form['auto_model']
            auto_year = int(request.form['auto_year'])

            # Create a DataFrame with the input data
            input_data = pd.DataFrame({
                'months_as_customer': [months_as_customer],
                'policy_state': [policy_state],
                'policy_csl': [policy_csl],
                'policy_deductable': [policy_deductable],
                'policy_annual_premium': [policy_annual_premium],
                'umbrella_limit': [umbrella_limit],
                'incident_type': [incident_type],
                'incident_severity': [incident_severity],
                'authorities_contacted': [authorities_contacted],
                'incident_state': [incident_state],
                'incident_city': [incident_city],
                'number_of_vehicles_involved': [number_of_vehicles_involved],
                'bodily_injuries': [bodily_injuries],
                'witnesses': [witnesses],
                'police_report_available': [police_report_available],
                'total_claim_amount': [total_claim_amount],
                'injury_claim': [injury_claim],
                'property_claim': [property_claim],
                'vehicle_claim': [vehicle_claim],
                'auto_make': [auto_make],
                'auto_model': [auto_model],
                'auto_year': [auto_year]
            })

            # Encode categorical columns
            cat_enc_data = pd.DataFrame(encoder.transform(input_data[categorical_columns]).toarray(), columns=encoder.get_feature_names_out())

            # Scale numerical columns
            num_scaled_data = scaler.transform(input_data[numerical_columns])

            # Combine encoded categorical and scaled numerical columns
            final_input = pd.concat([pd.DataFrame(num_scaled_data, columns=numerical_columns), cat_enc_data], axis=1)

            obj = PredictionPipeline()
            predict = obj.predict(final_input)

            if predict == 1:
                prediction_text = "It might be a fraud vehicle insurance claim"
            else:
                prediction_text = "It might be a true vehicle insurance claim"

            return render_template('results.html', prediction_text=prediction_text)

        except Exception as e:
            print('The Exception message is: ', str(e))
            return 'something is wrong'+str(e)

    else:
        return render_template('index.html')


if __name__ == "__main__":
    #app.run(host="0.0.0.0", port=8080, debug=True)
    app.run(host="0.0.0.0", port=8080)

