import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv('Disease_symptom_and_patient_profile_dataset.csv')
print(f'Data loaded: {df.shape}')

# Convert column names to lowercase and replace spaces with underscores
df.columns = df.columns.str.lower().str.replace(' ', '_')
print(f'Columns: {df.columns.tolist()}')

# Remove duplicates
df_clean = df.drop_duplicates()
print(f'After removing duplicates: {df_clean.shape}')

# Split features and targets
X = df_clean.drop(['outcome_variable', 'disease'], axis=1)
y_outcome = df_clean['outcome_variable']
y_disease = df_clean['disease']

print(f'Features shape: {X.shape}')
print(f'Feature columns: {X.columns.tolist()}')

# Encode Binary Features
binary_cols = ['gender', 'fever', 'fatigue', 'difficulty_breathing', 'cough']
label_encoders = {}

for col in binary_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
    print(f'Encoded {col}')

# One-Hot Encode Nominal Features
nominal_cols = ['blood_pressure', 'cholesterol_level']

one_hot_encoder = OneHotEncoder(
    drop='first',
    sparse_output=False,
    dtype=int
)

encoded_features = one_hot_encoder.fit_transform(X[nominal_cols])
encoded_df = pd.DataFrame(
    encoded_features,
    columns=one_hot_encoder.get_feature_names_out(nominal_cols),
    index=X.index
)

X = X.drop(columns=nominal_cols)
X = pd.concat([X, encoded_df], axis=1)
print(f'Final features shape: {X.shape}')

# Encode Targets
outcome_encoder = LabelEncoder()
y_outcome_encoded = outcome_encoder.fit_transform(y_outcome)

# Split dataset
X_train, X_test, y_outcome_train, y_outcome_test = train_test_split(
    X,
    y_outcome_encoded,
    test_size=0.2,
    random_state=42
)

# Train outcome model
Ran_model = RandomForestClassifier(random_state=42)
Ran_model.fit(X_train, y_outcome_train)
y_pred = Ran_model.predict(X_test)
print(f'Outcome Model Accuracy: {accuracy_score(y_outcome_test, y_pred):.4f}')

# Train disease model
positive_mask = y_outcome == 'Positive'
X_positive = X[positive_mask]
y_positive_disease = y_disease[positive_mask]

disease_encoder = LabelEncoder()
y_positive_encoded = disease_encoder.fit_transform(y_positive_disease)

disease_Rmodel = RandomForestClassifier(random_state=42)
disease_Rmodel.fit(X_positive, y_positive_encoded)
y_pred_dis = disease_Rmodel.predict(X_positive)
print(f'Disease Model Accuracy: {accuracy_score(y_positive_encoded, y_pred_dis):.4f}')

# Save models and encoders
joblib.dump(Ran_model, 'Ran_model.pkl')
joblib.dump(disease_Rmodel, 'disease_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(one_hot_encoder, 'one_hot_encoder.pkl')
joblib.dump(outcome_encoder, 'outcome_encoder.pkl')
joblib.dump(disease_encoder, 'disease_encoder.pkl')
joblib.dump(X.columns, 'feature_columns.pkl')

print('All models saved successfully!')
