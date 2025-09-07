import streamlit as st
import pandas as pd
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
import shap
import plotly.express as px
from urllib.request import urlopen
import json
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_PATH = os.path.join(SCRIPT_DIR, 'sahie_processed.csv.zip')
LGB_MODEL_PATH = os.path.join(SCRIPT_DIR, 'lgbm_model.txt')
DL_MODEL_PATH = os.path.join(SCRIPT_DIR, 'deep_learning_model.h5')
GEOJSON_URL = 'https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json'

# --- Data Loading and Preparation ---

@st.cache_data
def load_geojson(url):
    """Loads the US counties GeoJSON file."""
    with urlopen(url) as response:
        return json.load(response)

def clean_dataframe(df):
    """Applies a series of cleaning steps to the dataframe."""
    df['countyfips'] = df['countyfips'].str.zfill(5)
    df.dropna(subset=['county_name'], inplace=True)
    df['county_name'] = df['county_name'].str.strip()
    df = df[df['county_name'] != '']
    return df

@st.cache_data
def load_data_and_scaler(path):
    """Loads data, prepares it, and fits a scaler."""
    df = pd.read_csv(path, compression='zip', dtype={"statefips": str, "countyfips": str})
    df = clean_dataframe(df)
    
    numeric_features = ['year', 'statefips', 'countyfips', 'agecat', 'racecat', 'sexcat', 'iprcat']
    scaler = StandardScaler().fit(df[numeric_features])
    
    return df, scaler

# --- Model Loading ---

@st.cache_resource
def load_models_and_explainer():
    """Loads the trained models and the SHAP explainer."""
    lgbm = lgb.Booster(model_file=LGB_MODEL_PATH)
    dl_model = load_model(DL_MODEL_PATH)
    explainer = shap.TreeExplainer(lgbm)
    return lgbm, dl_model, explainer

# --- UI Components ---

def st_shap(plot, height=None):
    """A function to display a SHAP plot in Streamlit."""
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(layout="wide")
    st.title('Health Equity & Policy Simulation Tool')

    # --- Load Data ---
    try:
        df, scaler = load_data_and_scaler(PROCESSED_DATA_PATH)
        counties_geojson = load_geojson(GEOJSON_URL)
        lgbm_model, dl_model, explainer = load_models_and_explainer()
    except Exception as e:
        st.error(f"An error occurred during application startup: {e}")
        return

    # --- Sidebar Inputs ---
    st.sidebar.header('Simulation Parameters')
    features = ['year', 'statefips', 'countyfips', 'agecat', 'racecat', 'sexcat', 'iprcat']
    
    year = st.sidebar.selectbox('Year', sorted(df['year'].unique()))
    state = st.sidebar.selectbox('State', sorted(df['state_name'].unique()))
    county = st.sidebar.selectbox('County', sorted(df[df['state_name'] == state]['county_name'].unique()))
    age = st.sidebar.selectbox('Age Group', sorted(df['agecat_label'].unique()))
    race = st.sidebar.selectbox('Race/Ethnicity', sorted(df['racecat_label'].dropna().unique()))
    sex = st.sidebar.selectbox('Sex', sorted(df['sexcat_label'].unique()))
    income = st.sidebar.selectbox('Income Level', sorted(df['iprcat_label'].unique()))

    if st.sidebar.button('Run Simulation'):
        # --- Prediction Logic ---
        try:
            statefips = df[df['state_name'] == state]['statefips'].iloc[0]
            countyfips = df[df['county_name'] == county]['countyfips'].iloc[0]
            agecat = df[df['agecat_label'] == age]['agecat'].iloc[0]
            racecat = df[df['racecat_label'] == race]['racecat'].iloc[0]
            sexcat = df[df['sexcat_label'] == sex]['sexcat'].iloc[0]
            iprcat = df[df['iprcat_label'] == income]['iprcat'].iloc[0]

            input_data = pd.DataFrame([[year, statefips, countyfips, agecat, racecat, sexcat, iprcat]], columns=features)
            input_data_numeric = input_data.apply(pd.to_numeric)
            input_data_scaled = scaler.transform(input_data_numeric)

            lgbm_pred = lgbm_model.predict(input_data_numeric)[0]
            dl_pred = dl_model.predict(input_data_scaled)[0][0]

            # --- Display Results ---
            st.subheader('Prediction Results')
            # ... (rest of the display logic) ...

        except IndexError:
            st.error("Could not find data for the selected combination. Please try a different selection.")
        except Exception as e:
            st.error(f"An unexpected error occurred during prediction: {e}")

if __name__ == "__main__":
    main()