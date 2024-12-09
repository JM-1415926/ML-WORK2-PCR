import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from joblib import load
import xgboost as xgb

# File path
test_file_path = "TestDatasetExample.xls"

# 1. Data preprocessing
# Load the data
data = pd.read_excel(test_file_path, engine='xlrd')

# Replace missing value identifier 999 with NaN
data.replace(999, np.nan, inplace=True)

# Remove irrelevant columns, keeping only the necessary ID and feature columns
data.drop(columns=['RelapseFreeSurvival (outcome)', 'pCR (outcome)'], errors='ignore', inplace=True)

# Extract the ID column and feature columns
IDs = data['ID']
X = data.drop(columns=['ID'], errors='ignore')

# Fill missing values in features using KNNImputer
imputer = KNNImputer(n_neighbors=5)
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Load feature selector and scaler (assumed to be saved during training)
selector = load("feature_selector.joblib")  # Pre-trained SelectKBest
scaler = load("scaler.joblib")  # Pre-trained StandardScaler

# Align test set features (apply feature selector consistent with training)
X_selected = pd.DataFrame(selector.transform(X_imputed), columns=X_imputed.columns[selector.get_support()])

# Standardize test set features
X_final = pd.DataFrame(scaler.transform(X_selected), columns=X_selected.columns)

# 2. Load model and make predictions
# Load preprocessed data
patient_ids, X_test = IDs, X_final

# Load the trained model
model_path = "Model-xgboost.json"  # Replace with the path of the final chosen model
model = xgb.Booster()
model.load_model(model_path)

# Generate predictions
dtest = xgb.DMatrix(X_test)
predictions = model.predict(dtest)
predictions = np.round(predictions).astype(int)  # Ensure predictions are integers

# Save prediction results
output_file = "PCRPrediction.csv"
output = pd.DataFrame({"ID": patient_ids, "Prediction": predictions})
output.to_csv(output_file, index=False)
print(f"Predictions saved to {output_file}")
