import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score
import matplotlib.pyplot as plt

def load_preprocessed_data(file_path):
    """
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
    Args:
        estimator: Trained model.
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        title (str): Plot title.
    """
    from sklearn.model_selection import learning_curve
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

def train_and_save_mlp(X, y, output_model_path="mlp_model.pkl"):
    """
    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        output_model_path (str): Path to save the model.

    Returns:
        model: Trained MLP model.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Initialize model
    model = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=500,
        random_state=42,
        activation='relu',
        solver='adam',
        learning_rate='adaptive'
    )

    # Train model
    model.fit(X_train, y_train)

    # Save model
    import joblib
    joblib.dump(model, output_model_path)
    print(f"Model saved to {output_model_path}")

    # Evaluate on test set
    y_pred = model.predict(X_test)
    print("Classification report on test set:")
    print(classification_report(y_test, y_pred))
    print(f"Balanced accuracy on test set: {balanced_accuracy_score(y_test, y_pred):.4f}")

    return model

if __name__ == "__main__":
    # Input dataset file path
    data_file_path = "TrainDataset2024_preprocessed.xlsx"

    # Load preprocessed data
    X, y = load_preprocessed_data(data_file_path)

    # Train and save the model
    model_path = "Model-MLP.pkl"
    best_model = train_and_save_mlp(X, y, model_path)

    # Plot the learning curve
    plot_learning_curve(best_model, X, y, title="MLP Learning Curve")
