from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Function to check if models exist, if not train them
def load_or_train_models():
    model_files = [
        "Ran_model.pkl", 
        "disease_model.pkl",
        "label_encoders.pkl",
        "one_hot_encoder.pkl",
        "outcome_encoder.pkl",
        "disease_encoder.pkl",
        "feature_columns.pkl"
    ]
    
    # Check if all model files exist
    models_exist = all(os.path.exists(f) for f in model_files)
    
    if not models_exist:
        print("Model files not found. Training models...")
        os.system("python train_models.py")
    
    # Load models / encoders
    Ran_model = joblib.load("Ran_model.pkl")
    disease_Rmodel = joblib.load("disease_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    one_hot_encoder = joblib.load("one_hot_encoder.pkl")
    outcome_encoder = joblib.load("outcome_encoder.pkl")
    disease_encoder = joblib.load("disease_encoder.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    
    return Ran_model, disease_Rmodel, label_encoders, one_hot_encoder, outcome_encoder, disease_encoder, feature_columns

# Load or train models
Ran_model, disease_Rmodel, label_encoders, one_hot_encoder, outcome_encoder, disease_encoder, feature_columns = load_or_train_models()

binary_cols = [
    'gender',
    'fever',
    'fatigue',
    'difficulty_breathing',
    'cough'
]

nominal_cols = [
    'blood_pressure',
    'cholesterol_level'
]


def predict_patient(patient_data):
    patient = pd.DataFrame([patient_data])

    for col in patient.columns:
        if patient[col].dtype == object:
            patient[col] = patient[col].astype(str).str.strip().str.title()

    for col in binary_cols:
        patient[col] = label_encoders[col].transform(patient[col])

    encoded = one_hot_encoder.transform(patient[nominal_cols])

    encoded_df = pd.DataFrame(
        encoded,
        columns=one_hot_encoder.get_feature_names_out(nominal_cols),
        index=patient.index
    )

    patient = patient.drop(columns=nominal_cols)
    patient = pd.concat([patient, encoded_df], axis=1)

    patient = patient.reindex(columns=feature_columns, fill_value=0)

    outcome_pred = Ran_model.predict(patient)[0]
    outcome_label = outcome_encoder.inverse_transform([outcome_pred])[0]

    if outcome_label == 'Negative':
        return "Negative", None

    disease_pred = disease_Rmodel.predict(patient)[0]
    disease_label = disease_encoder.inverse_transform([disease_pred])[0]

    return "Positive", disease_label


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    disease = None

    if request.method == 'POST':
        patient_data = {
            'age': int(request.form['age']),
            'gender': request.form['gender'],
            'fever': request.form['fever'],
            'cough': request.form['cough'],
            'fatigue': request.form['fatigue'],
            'difficulty_breathing': request.form['difficulty_breathing'],
            'blood_pressure': request.form['blood_pressure'],
            'cholesterol_level': request.form['cholesterol_level']
        }

        prediction, disease = predict_patient(patient_data)

    return render_template(
        'index.html',
        prediction=prediction,
        disease=disease
    )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)