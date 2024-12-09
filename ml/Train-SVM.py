import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.svm import SVC
from sklearn.metrics import classification_report, balanced_accuracy_score
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

def load_preprocessed_data(file_path):
    """
    Load preprocessed dataset. The preprocessed file should contain the full feature matrix and target variable.

    Args:
        file_path (str): Path to the dataset file (.xlsx format).

    Returns:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
    """
    data = pd.read_excel(file_path, engine='openpyxl')
    X = data.drop(columns=['ID', 'pCR (outcome)'], errors='ignore')
    y = data['pCR (outcome)']
    return X, y

def plot_learning_curve(estimator, X, y, title="Learning Curve"):
    """
    Plot the learning curve.

    Args:
        estimator: Trained model.
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        title (str): Plot title.
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, scoring='balanced_accuracy', n_jobs=-1,
        train_sizes=[0.1, 0.33, 0.55, 0.78, 1.0]
    )

    train_scores_mean = train_scores.mean(axis=1)
    test_scores_mean = test_scores.mean(axis=1)

    plt.figure()
    plt.plot(train_sizes, train_scores_mean, label="Training Score", marker='o')
    plt.plot(train_sizes, test_scores_mean, label="Validation Score", marker='o')
    plt.title(title)
    plt.xlabel("Training Examples")
    plt.ylabel("Balanced Accuracy")
    plt.legend(loc="best")
    plt.grid()
    plt.show()

def train_and_save_svm(X, y, output_model_path="svm_model.pkl"):
    """
    Train a Support Vector Machine (SVM) classification model, save the trained model, and return the best model.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        output_model_path (str): Path to save the model.

    Returns:
        best_model: Best model (for evaluation or prediction).
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Define SVM model and parameter range
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC())
    ])

    param_grid = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'svm__gamma': ['scale', 'auto'],
        'svm__degree': [2, 3, 4],
        'svm__class_weight': ['balanced', None]
    }

    # Use StratifiedKFold
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Grid search
    grid_search = GridSearchCV(
        pipeline, param_grid, scoring='balanced_accuracy', cv=cv_strategy, verbose=2, n_jobs=-1, error_score='raise'
    )
    grid_search.fit(X_train, y_train)

    # Best parameters and model
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Balanced Accuracy: {grid_search.best_score_}")
    best_model = grid_search.best_estimator_

    # Save model
    joblib.dump(best_model, output_model_path)
    print(f"Model saved to {output_model_path}")

    # Test set evaluation
    y_pred = best_model.predict(X_test)
    print("Test Set Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Test Set Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")

    return best_model

if __name__ == "__main__":
    # Input data file path
    data_file_path = "TrainDataset2024_preprocessed.xlsx"

    # Load preprocessed data
    X, y = load_preprocessed_data(data_file_path)

    # Train and save model
    model_path = "Model-SVM.pkl"
    best_model = train_and_save_svm(X, y, model_path)

    # Plot learning curve
    plot_learning_curve(best_model, X, y, title="SVM Learning Curve")

    # Load model for validation
    loaded_model = joblib.load(model_path)
    print("\nModel loaded successfully!")
