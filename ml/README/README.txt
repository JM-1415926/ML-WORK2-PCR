---File Structure---

1. train_data_preprocessing.py: Handles missing data imputation, feature selection, and scaling for the training dataset.
2. Train-xgboost.py: Trains the XGBoost model for PCR classification, saves the model, and evaluates its performance.
3. Train-decision_tree.py: Trains a Decision Tree model for comparison.
4. Train-MLP.py: Trains a Multi-Layer Perceptron model.
5. Train-SVM.py: Trains a Support Vector Machine (SVM) model.
6. FinalTestPCR.py: Prepares the test dataset, loads the trained XGBoost model, and generates PCRPrediction.csv for PCR.

Models
- Model-xgboost.json: The final XGBoost model selected based on its superior performance.
- Other Models: Models trained for comparison include:
  - Model-SVM.pkl
  - Model-MLP.pkl
  - Model-decision_tree.pkl

Utilities
- feature_selector.joblib: Pre-trained feature selector.
- scaler.joblib: StandardScaler used for feature normalization.

Data
- TrainDataset2024.xlsx: Raw training data.
- TrainDataset2024_preprocessed.xlsx: Preprocessed training data after cleaning.
- TestDatasetExample.xls: Example test dataset provided for validation.
- PCRPrediction.csv: Final predictions for the PCR task.

---Model Tuning and Selection---

XGBoost Tuning Process
The following hyperparameters were tuned to achieve the best performance:
- max_depth: Controlled the depth of trees to prevent overfitting. Optimal value: 6.
- learning_rate: Adjusted to balance the speed of convergence and model performance. Optimal value: 0.1.
- n_estimators: Determined the number of boosting rounds. Optimal value: 100.
- eval_metric: Used logloss for binary classification optimization.
- random_state: Ensured reproducibility with a seed value of 42.

Comparison with Other Models
XGBoost outperformed Decision Tree, MLP, and SVM in terms of balanced accuracy on the validation dataset. 

---How to Run---

Training
1. Use train_data_preprocessing.py to preprocess the dataset.
2. Train the models using respective training scripts, e.g., Train-xgboost.py.
3. Save the trained models for future predictions.

Testing
1. Run FinalTestPCR.py to preprocess the test dataset and generate predictions using the trained XGBoost model.
2. The output file PCRPrediction.csv will contain the predictions for the test dataset.



