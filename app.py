
import streamlit as st
import pandas as pd
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_PATH = os.path.join(SCRIPT_DIR, 'sahie_processed.csv')
LGB_MODEL_PATH = os.path.join(SCRIPT_DIR, 'lgbm_model.txt')
DL_MODEL_PATH = os.path.join(SCRIPT_DIR, 'deep_learning_model.h5')

# --- Model and Explainer Loading ---
@st.cache_resource
def load_models_and_explainer():
    """Loads the trained models and the SHAP explainer."""
    # Load LightGBM model
    lgbm = lgb.Booster(model_file=LGB_MODEL_PATH)
    
    # Load Deep Learning model
    dl_model = load_model(DL_MODEL_PATH)
    
    # Create SHAP Explainer for the LightGBM model
    explainer = shap.TreeExplainer(lgbm)
    
    return lgbm, dl_model, explainer



# --- Main App UI ---
st.set_page_config(layout="wide")
st.title('Health Equity & Policy Simulation Tool')

st.sidebar.header('Simulation Parameters')

# Load data for dropdowns
@st.cache_data
def load_data(path):
    """Loads the processed data for UI elements."""
    df = pd.read_csv(path)
    # Clean up state and county names to prevent mismatches
    if 'state_name' in df.columns:
        df['state_name'] = df['state_name'].str.strip()
    if 'county_name' in df.columns:
        df['county_name'] = df['county_name'].str.strip()
        # Drop rows with null/NaN county names
        df.dropna(subset=['county_name'], inplace=True)
        # Filter out rows where county name is an empty string
        df = df[df['county_name'] != '']
    return df

df = load_data(PROCESSED_DATA_PATH)
features = ['year', 'statefips', 'countyfips', 'agecat', 'racecat', 'sexcat', 'iprcat']

# Create dropdowns for user input
year = st.sidebar.selectbox('Year', sorted(df['year'].unique()))
state = st.sidebar.selectbox('State', sorted(df['state_name'].unique()))
county = st.sidebar.selectbox('County', sorted(df[df['state_name'] == state]['county_name'].unique()))
age = st.sidebar.selectbox('Age Group', sorted(df['agecat_label'].unique()))
race = st.sidebar.selectbox('Race/Ethnicity', sorted(df['racecat_label'].dropna().unique()))
sex = st.sidebar.selectbox('Sex', sorted(df['sexcat_label'].unique()))
income = st.sidebar.selectbox('Income Level', sorted(df['iprcat_label'].unique()))

# --- Prediction and Explanation ---
if st.sidebar.button('Run Simulation'):
    # Get FIPS codes and category codes from the dataframe
    statefips = df[df['state_name'] == state]['statefips'].iloc[0]
    countyfips = df[df['county_name'] == county]['countyfips'].iloc[0]
    agecat = df[df['agecat_label'] == age]['agecat'].iloc[0]
    racecat = df[df['racecat_label'] == race]['racecat'].iloc[0]
    sexcat = df[df['sexcat_label'] == sex]['sexcat'].iloc[0]
    iprcat = df[df['iprcat_label'] == income]['iprcat'].iloc[0]

    # Create input array for models
    input_data = pd.DataFrame([[year, statefips, countyfips, agecat, racecat, sexcat, iprcat]], columns=features)

    # --- Scale the input data for the DL model ---
    scaler = StandardScaler()
    scaler.fit(df[features])
    input_data_scaled = scaler.transform(input_data)

    # Load models and explainer
    lgbm_model, dl_model, explainer = load_models_and_explainer()

    # Make predictions
    lgbm_pred = lgbm_model.predict(input_data)[0]
    dl_pred = dl_model.predict(input_data_scaled)[0][0]

    st.subheader('Prediction Results')
    col1, col2 = st.columns(2)
    col1.metric("Predicted Uninsured Rate (LightGBM)", f"{lgbm_pred:.2f}%")
    col2.metric("Predicted Uninsured Rate (Deep Learning)", f"{dl_pred:.2f}%")
    
    st.markdown("---")

    st.subheader('Why did the LightGBM model make this prediction?')
    
    # Calculate and display SHAP explanation
    shap_values = explainer.shap_values(input_data)
    st.info("The plot below shows the contribution of each factor to the final prediction. Features in red increased the predicted uninsured rate, while features in blue decreased it.")
    
    # Generate the SHAP plot with matplotlib
    shap.force_plot(explainer.expected_value, shap_values, input_data, matplotlib=True, show=False)
    fig = plt.gcf()
    st.pyplot(fig, bbox_inches='tight')

else:
    st.info('Please select parameters in the sidebar and click "Run Simulation".')

