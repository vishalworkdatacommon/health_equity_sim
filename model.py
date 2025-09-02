import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os
import mlflow
import mlflow.lightgbm

def train_model(data_path='sahie_processed.csv'):
    """
    Trains a LightGBM model and logs the experiment with MLflow.

    Args:
        data_path (str): The path to the processed data file.
    """
    # --- File Paths ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Load from the local project directory
    full_data_path = os.path.join(script_dir, data_path)

    # --- Load Data ---
    df = pd.read_csv(full_data_path)
    features = ['year', 'statefips', 'countyfips', 'agecat', 'racecat', 'sexcat', 'iprcat']
    target = 'PCTUI'
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- MLflow Experiment Tracking ---
    # Set the experiment name
    mlflow.set_experiment("SAHIE Uninsured Rate Prediction")

    with mlflow.start_run() as run:
        print("Starting MLflow run:", run.info.run_name)

        # --- Model Training ---
        # Define model parameters to be logged
        params = {
            "random_state": 42,
            "n_estimators": 100,
            "learning_rate": 0.1,
            "num_leaves": 31
        }
        
        mlflow.log_params(params) # Log parameters

        print("Training the LightGBM model...")
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        print("Model training complete.")

        # --- Model Evaluation ---
        print("\nEvaluating the model...")
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            "mae": mae,
            "r2": r2
        }
        mlflow.log_metrics(metrics) # Log metrics

        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R-squared (R2): {r2:.4f}")

        # --- Log Model Artifact ---
        print("\nLogging model to MLflow...")
        mlflow.lightgbm.log_model(model, "lgbm_model")
        print("Model logged successfully.")

        # --- Save Model Locally within the project directory ---
        model_path = os.path.join(script_dir, 'lgbm_model.txt')
        model.booster_.save_model(model_path)
        print(f"\nLightGBM model also saved locally to {model_path}")

if __name__ == '__main__':
    train_model()
