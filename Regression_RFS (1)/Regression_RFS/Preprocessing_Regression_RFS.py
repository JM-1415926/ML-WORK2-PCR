import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from joblib import dump

# File path
file_path = "TrainDataset2024.xls"

# Load data
data = pd.read_excel(file_path)

# Replace missing value identifier 999 with NaN
data.replace(999, np.nan, inplace=True)

# Remove another target variable, keep only the RFS column
data.drop(columns=['pCR (outcome)'], errors='ignore', inplace=True)

# Extract features and target variable
X = data.drop(columns=['ID', 'RelapseFreeSurvival (outcome)'], errors='ignore')
y = data['RelapseFreeSurvival (outcome)']

# Remove samples with missing target variable values
valid_indices = y.notna()
X = X.loc[valid_indices].reset_index(drop=True)
y = y[valid_indices].reset_index(drop=True)

# Handle outliers by capping them using the IQR method
def cap_outliers(df):
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower_bound, upper_bound)
    return df

X = cap_outliers(X)

# Fill missing values in features using IterativeImputer (Regression Imputation)
imputer = IterativeImputer(random_state=42)
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Feature Selection using LASSO (L1 regularization)
print("Performing feature selection using LASSO...")
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X, y)

# Select features where the coefficient is non-zero
lasso_selected_features = X.columns[(lasso.coef_ != 0)]

# Alternatively, use Random Forest feature importance for selection
print("Performing feature selection using Random Forest...")
rf = RandomForestRegressor(random_state=42)
rf.fit(X, y)

# Use SelectFromModel to select top features based on importance from Random Forest
rf_selector = SelectFromModel(rf, threshold=0.01, max_features=15)
rf_selector.fit(X, y)  # Fit with feature names
X_rf_selected = rf_selector.transform(X)
rf_selected_features = X.columns[rf_selector.get_support()]

# Combine the features from both methods (LASSO and Random Forest)
final_selected_features = list(set(lasso_selected_features).union(set(rf_selected_features)))
X_selected = X[final_selected_features].copy()

# Force retention of important features
important_features = ['ER', 'HER2', 'Gene']
for feature in important_features:
    if feature in X.columns and feature not in X_selected.columns:
        X_selected.loc[:, feature] = X.loc[:, feature]

# Standardize data using RobustScaler
scaler = RobustScaler()
X_final = pd.DataFrame(scaler.fit_transform(X_selected), columns=X_selected.columns)

# Combine processed features with target variable and IDs
final_data = pd.concat([data.loc[valid_indices, 'ID'].reset_index(drop=True), X_final, y.reset_index(drop=True)], axis=1)

# Save the result
output_path = "TrainDataset2024_RFS_preprocessed.xlsx"
final_data.to_excel(output_path, index=False, engine='openpyxl')
print(f"The preprocessed data has been saved to: {output_path}")

# Save feature selector, scaler, and important features
dump(scaler, "scaler_RFS.joblib")
print("scaler, and important features have been saved.")

#save selected feature
selected_features = list(X_final.columns)
pd.Series(selected_features).to_csv('selected_features.csv', index=False)
