import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from joblib import load
import json
import warnings
warnings.filterwarnings('ignore')

# File path
test_file_path = "TestDatasetExample.xls"

# Load the exact feature order used in training
with open('feature_order.json', 'r') as f:
    feature_order = json.load(f)

# Load data
data = pd.read_excel(test_file_path, engine='xlrd')

# Replace missing value identifier 999 with NaN
data.replace(999, np.nan, inplace=True)

# Remove irrelevant columns
data.drop(columns=['RelapseFreeSurvival (outcome)', 'pCR (outcome)'], errors='ignore', inplace=True)

# Extract the ID column and feature columns
IDs = data['ID']
X = data.drop(columns=['ID'], errors='ignore')

# Fill missing values
imputer = IterativeImputer(random_state=42)
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Load scaler
scaler = load("scaler_RFS.joblib")

# Create DataFrame with all required features in the correct order
X_selected = pd.DataFrame(columns=feature_order)
for feature in feature_order:
    if feature in X_imputed.columns:
        X_selected[feature] = X_imputed[feature]
    else:
        X_selected[feature] = 0

# Standardize features
X_final = pd.DataFrame(scaler.transform(X_selected), columns=feature_order)

# Verify feature order
assert (X_final.columns == feature_order).all(), "Feature order mismatch!"

# Load model and make predictions
model = load("Model-GB.pkl")
predictions = model.predict(X_final)

# Save predictions
output_file = "RFSPrediction.csv"
output = pd.DataFrame({"ID": IDs, "Prediction": predictions})
output.to_csv(output_file, index=False)
print(f"Predictions saved to {output_file}")
