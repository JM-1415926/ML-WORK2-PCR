import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from joblib import dump

# File path
file_path = "TrainDataset2024.xls"

# Load data
data = pd.read_excel(file_path, engine='xlrd')

# Replace missing value identifier 999 with NaN
data.replace(999, np.nan, inplace=True)

# Remove another target variable, keep only the pCR column
data.drop(columns=['RelapseFreeSurvival (outcome)'], errors='ignore', inplace=True)

# Extract features and target variable
X = data.drop(columns=['ID', 'pCR (outcome)'], errors='ignore')
y = data['pCR (outcome)']

# Remove samples with missing target variable values
valid_indices = y.notna()
X = X.loc[valid_indices].reset_index(drop=True)
y = y[valid_indices].reset_index(drop=True)

# Fill missing values in features using KNNImputer
imputer = KNNImputer(n_neighbors=5)
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Handle class imbalance
print("Original class distribution:")
print(y.value_counts())
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("Class distribution after balancing:")
print(y_resampled.value_counts())

# Retain important features
important_features = ['ER', 'HER2', 'Gene']
selected_features = SelectKBest(score_func=mutual_info_classif, k=15).fit(X_resampled, y_resampled)
X_selected = pd.DataFrame(selected_features.transform(X_resampled), columns=X_resampled.columns[selected_features.get_support()])

# Force retention of important features
for feature in important_features:
    if feature not in X_selected.columns:
        X_selected[feature] = X_resampled[feature]

# Standardize data
scaler = StandardScaler()
X_final = pd.DataFrame(scaler.fit_transform(X_selected), columns=X_selected.columns)

# Combine features and target variable
final_data = pd.concat([data['ID'][valid_indices].reset_index(drop=True), X_final, y_resampled.reset_index(drop=True)], axis=1)

# Save the result
output_path = "TrainDataset2024_preprocessed.xlsx"
final_data.to_excel(output_path, index=False, engine='openpyxl')
print(f"The preprocessed data has been saved to: {output_path}")

# Save feature selector, scaler, and important features
dump(selected_features, "feature_selector.joblib")
dump(scaler, "scaler.joblib")
print("Feature selector, scaler, and important features have been saved.")
