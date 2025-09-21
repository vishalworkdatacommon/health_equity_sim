
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

def generate_shap_summary(data_path='sahie_processed.csv', model_path='lgbm_model.txt'):
    """
    Generates and saves a SHAP summary plot for the trained LightGBM model.

    Args:
        data_path (str): Path to the processed data.
        model_path (str): Path to the trained LightGBM model.
    """
    # --- File Paths ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_data_path = os.path.join(script_dir, data_path)
    full_model_path = os.path.join(script_dir, model_path)
    
    # --- Load Data and Model ---
    print("Loading data and model...")
    df = pd.read_csv(full_data_path)
    model = lgb.Booster(model_file=full_model_path)
    print("Data and model loaded.")

    # --- Prepare Data ---
    features = ['year', 'statefips', 'countyfips', 'agecat', 'racecat', 'sexcat', 'iprcat']
    target = 'PCTUI'
    X = df[features]
    y = df[target]
    _, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Calculate SHAP Values ---
    print("Calculating SHAP values... (This may take a moment)")
    # Using TreeExplainer for tree-based models like LightGBM
    explainer = shap.TreeExplainer(model)
    # We'll use a subset of the test data for performance reasons
    shap_values = explainer.shap_values(X_test.sample(1000, random_state=42))
    print("SHAP values calculated.")

    # --- Generate and Save Plot ---
    print("Generating SHAP summary plot...")
    plt.figure()
    shap.summary_plot(shap_values, X_test.sample(1000, random_state=42), show=False)
    
    # Tweak plot for better readability
    plt.title('Feature Impact on Model Output (SHAP Summary)')
    plt.tight_layout()
    
    output_path = os.path.join(script_dir, 'shap_summary_plot.png')
    plt.savefig(output_path)
    print(f"SHAP summary plot saved to {output_path}")
    plt.close()

if __name__ == '__main__':
    # First, ensure you have the necessary libraries
    # pip install -r requirements.txt
    
    # Then, ensure you have trained the model and have the lgbm_model.txt file
    # python3 model.py
    
    generate_shap_summary()
