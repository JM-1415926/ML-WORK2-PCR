import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

def load_preprocessed_data(file_path):
    """
    Args:
        file_path (str): Path to the dataset file (.xlsx format).

    Returns:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
    """
    data = pd.read_excel(file_path, engine='openpyxl')
    X = data.drop(columns=['ID', 'RelapseFreeSurvival (outcome)'], errors='ignore')
    y = data['RelapseFreeSurvival (outcome)']
    return X, y

def plot_learning_curve(estimator, X, y, title="Learning Curve"):
    """
    Args:
        estimator: Trained model.
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        title (str): Plot title.
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1,
        train_sizes=[0.1, 0.33, 0.55, 0.78, 1.0]
    )

    train_scores_mean = -train_scores.mean(axis=1)
    test_scores_mean = -test_scores.mean(axis=1)

    plt.figure()
    plt.plot(train_sizes, train_scores_mean, label="Training MAE", marker='o')
    plt.plot(train_sizes, test_scores_mean, label="Validation MAE", marker='o')
    plt.title(title)
    plt.xlabel("Training Examples")
    plt.ylabel("Mean Absolute Error")
    plt.legend(loc="best")
    plt.grid()
    plt.show()

def train_and_save_gb(X, y, output_model_path="gb_model.pkl"):
    """
    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        output_model_path (str): Path to save the model.

    Returns:
        best_model: Best model (for evaluation or prediction).
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define Gradient Boosting model and parameter range
    gb = GradientBoostingRegressor(random_state=42)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Use KFold cross-validation
    cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

    selected_features_order = X_train.columns.tolist()
    # 保存为JSON格式以保持顺序
    import json
    with open('feature_order.json', 'w') as f:
        json.dump(selected_features_order, f)

    # Grid search
    grid_search = GridSearchCV(
        gb, param_grid, scoring='neg_mean_absolute_error', cv=cv_strategy, verbose=2, n_jobs=-1, error_score='raise'
    )
    grid_search.fit(X_train, y_train)

    # Best parameters and model
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best MAE Score: {-grid_search.best_score_}")
    best_model = grid_search.best_estimator_

    # Save model
    joblib.dump(best_model, output_model_path)
    print(f"Model saved to {output_model_path}")

    # Test set evaluation
    y_pred = best_model.predict(X_test)
    print("Test Set Evaluation:")
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

    return best_model

if __name__ == "__main__":
    # Input data file path
    data_file_path = "TrainDataset2024_RFS_preprocessed.xlsx"

    # Load preprocessed data
    X, y = load_preprocessed_data(data_file_path)

    # Train and save model
    model_path = "Model-GB.pkl"
    best_model = train_and_save_gb(X, y, model_path)

    # Plot learning curve
    plot_learning_curve(best_model, X, y, title="Gradient Boosting Learning Curve")

    # Load model for validation
    loaded_model = joblib.load(model_path)
    print("\nModel loaded successfully!")
