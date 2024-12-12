import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
import joblib

def load_preprocessed_data(file_path):
    data = pd.read_excel(file_path, engine='openpyxl')
    X = data.drop(columns=['ID', 'RelapseFreeSurvival (outcome)'], errors='ignore')
    y = data['RelapseFreeSurvival (outcome)']
    return X, y

def create_model(optimizer='adam', init='uniform', neurons=32):
    model = Sequential()
    model.add(Dense(neurons, input_dim=X.shape[1], kernel_initializer=init, activation='relu'))
    model.add(Dense(1, kernel_initializer=init))
    model.compile(loss='mean_absolute_error', optimizer=optimizer)
    return model

def plot_learning_curve(estimator, X, y, title="Learning Curve"):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1,
        train_sizes=[0.1, 0.33, 0.55, 0.78, 1.0]
    )

    train_scores_mean = -train_scores.mean(axis=1)
    test_scores_mean = -test_scores.mean(axis=1)

    plt.figure()
    plt.plot(train_sizes, train_scores_mean, label="Training Score (MAE)", marker='o')
    plt.plot(train_sizes, test_scores_mean, label="Validation Score (MAE)", marker='o')
    plt.title(title)
    plt.xlabel("Training Examples")
    plt.ylabel("Mean Absolute Error")
    plt.legend(loc="best")
    plt.grid()
    plt.show()

def train_and_save_nn(X, y, output_model_path="nn_model.pkl"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = KerasRegressor(model=create_model, verbose=0)

    param_grid = {
        'batch_size': [10, 20, 40],
        'epochs': [50, 100, 200],
        'model__optimizer': ['adam', 'rmsprop'],
        'model__init': ['uniform', 'normal'],
        'model__neurons': [32, 64, 128]
    }

    cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=cv_strategy, verbose=2, n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best MAE Score: {-grid_search.best_score_}")
    best_model = grid_search.best_estimator_

    joblib.dump(best_model, output_model_path)
    print(f"Model saved to {output_model_path}")

    y_pred = best_model.predict(X_test)
    print("Test Set Evaluation:")
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

    return best_model

if __name__ == "__main__":
    data_file_path = "TrainDataset2024_RFS_preprocessed.xlsx"
    X, y = load_preprocessed_data(data_file_path)
    model_path = "Model-NN.pkl"
    best_model = train_and_save_nn(X, y, model_path)
    plot_learning_curve(best_model, X, y, title="Neural Network Learning Curve")
    loaded_model = joblib.load(model_path)
    print("\nModel loaded successfully!")