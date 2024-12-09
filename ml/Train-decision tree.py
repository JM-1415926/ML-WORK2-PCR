import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

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

def train_and_save_decision_tree(X, y, output_model_path="decision_tree_model.pkl"):
    """
    Train a decision tree classification model, save the trained model, and return the best model.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        output_model_path (str): Path to save the model.

    Returns:
        best_model: The best model (for evaluation or prediction).
    """
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Define the decision tree model and hyperparameter grid
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('dt', DecisionTreeClassifier(random_state=42))
    ])

    param_grid = {
        'dt__max_depth': [None, 10, 20, 30, 40],
        'dt__min_samples_split': [2, 5, 10],
        'dt__min_samples_leaf': [1, 2, 4],
        'dt__class_weight': [None, 'balanced']
    }

    # Use StratifiedKFold
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Perform grid search
    grid_search = GridSearchCV(
        pipeline, param_grid, scoring='balanced_accuracy', cv=cv_strategy, verbose=2, n_jobs=-1, error_score='raise'
    )
    grid_search.fit(X_train, y_train)

    # Best parameters and model
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Balanced Accuracy: {grid_search.best_score_}")
    best_model = grid_search.best_estimator_

    # Save the model
    joblib.dump(best_model, output_model_path)
    print(f"Model saved to {output_model_path}")

    # Evaluate on the test set
    y_pred = best_model.predict(X_test)
    print("Test Set Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Test Set Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")

    return best_model

if __name__ == "__main__":
    # Input dataset file path
    data_file_path = "TrainDataset2024_preprocessed.xlsx"

    # Load the preprocessed data
    X, y = load_preprocessed_data(data_file_path)

    # Train and save the model
    model_path = "Model-decision_tree.pkl"
    best_model = train_and_save_decision_tree(X, y, model_path)

    # Plot the learning curve
    plot_learning_curve(best_model, X, y, title="Decision Tree Learning Curve")

    # Load the model for validation
    loaded_model = joblib.load(model_path)
    print("\nModel loaded successfully!")
