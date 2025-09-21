import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os
import mlflow
import mlflow.tensorflow

def train_deep_learning_model(data_path='sahie_processed.csv'):
    """
    Trains a deep learning model and logs the experiment with MLflow.

    Args:
        data_path (str): The path to the processed data file.
    """
    # --- File Paths ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_data_path = os.path.join(script_dir, data_path)

    # --- Load and Prepare Data ---
    df = pd.read_csv(full_data_path)
    features = ['year', 'statefips', 'countyfips', 'agecat', 'racecat', 'sexcat', 'iprcat']
    target = 'PCTUI'
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --- MLflow Experiment Tracking ---
    mlflow.set_experiment("SAHIE Uninsured Rate Prediction")

    with mlflow.start_run(run_name="DeepLearning_Run") as run:
        print("Starting MLflow run:", run.info.run_name)

        # --- Build and Train Model ---
        # Log parameters
        params = {
            "epochs": 10,
            "batch_size": 32,
            "optimizer": "adam",
            "loss_function": "mean_absolute_error",
            "architecture": "Dense(64) -> Dense(64) -> Dense(1)"
        }
        mlflow.log_params(params)

        model = Sequential([
            Dense(64, activation='relu', input_shape=[X_train.shape[1]]),
            Dense(64, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=params['optimizer'], loss=params['loss_function'])

        print("Training the deep learning model...")
        model.fit(X_train, y_train, 
                  epochs=params['epochs'], 
                  batch_size=params['batch_size'], 
                  validation_split=0.2, 
                  verbose=1)
        print("Model training complete.")

        # --- Model Evaluation ---
        print("\nEvaluating the model...")
        loss = model.evaluate(X_test, y_test, verbose=0)
        print(f"Mean Absolute Error (MAE) on test data: {loss:.4f}")
        mlflow.log_metric("mae", loss)

        # --- Log Model Artifact ---
        print("\nLogging model to MLflow...")
        mlflow.tensorflow.log_model(model, "dl_model")
        print("Model logged successfully.")

        # --- Save Model Locally within the project directory ---
        model_path = os.path.join(script_dir, 'deep_learning_model.h5')
        model.save(model_path)
        print(f"\nDeep learning model also saved locally to {model_path}")

if __name__ == '__main__':
    train_deep_learning_model()
