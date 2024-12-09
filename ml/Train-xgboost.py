import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np

def load_preprocessed_data(file_path):
    """
    Load the preprocessed dataset. The preprocessed file should contain a complete feature matrix and target variable.

    Args:
        file_path (str): Dataset file path (.xlsx format).

    Returns:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
    """
    data = pd.read_excel(file_path, engine='openpyxl')
    X = data.drop(columns=['ID', 'pCR (outcome)'], errors='ignore')
    y = data['pCR (outcome)']
    return X, y

def plot_learning_curve(estimator, X, y):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5), scoring='balanced_accuracy'
    )
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label="Train score")
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label="Test score")
    plt.xlabel("Training Size")
    plt.ylabel("Score")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid()
    plt.show()

def train_and_save_xgboost(X, y, output_model_path="xgboost_model.pkl"):
    """
    Train an XGBoost classifier, save the trained model, and return the model.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        output_model_path (str): Path to save the model.

    Returns:
        model: Trained XGBoost model.
    """
    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Initialize the model
    model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        random_state=42
    )

    # Train the model
    model.fit(X_train, y_train)

    # Save the model
    model.save_model(output_model_path)
    print(f"Model saved to {output_model_path}")

    # Test set evaluation
    y_pred = model.predict(X_test)
    print("Classification report on test set:")
    print(classification_report(y_test, y_pred))
    print(f"Balanced accuracy on test set: {balanced_accuracy_score(y_test, y_pred):.4f}")

    # Plot feature importance
    model = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    plot_learning_curve(model, X, y)

    return model

if __name__ == "__main__":
    # Input dataset file path
    data_file_path = "TrainDataset2024_preprocessed.xlsx"

    # Load the preprocessed data
    X, y = load_preprocessed_data(data_file_path)

    # Train and save the model
    model_path = "Model-xgboost.json"
    best_model = train_and_save_xgboost(X, y, model_path)
