
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

# --- Caching GeoJSON data ---
@st.cache_data
def load_geojson():
    """Loads the US counties GeoJSON file."""
    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        return json.load(response)

counties_geojson = load_geojson()

# --- Model and Explainer Loading ---
@st.cache_resource
def load_models_and_explainer():
    """Loads the trained models and the SHAP explainer."""
    lgbm = lgb.Booster(model_file=LGB_MODEL_PATH)
    dl_model = load_model(DL_MODEL_PATH)
    explainer = shap.TreeExplainer(lgbm)
    return lgbm, dl_model, explainer

# --- UI Component for SHAP Plot ---
def st_shap(plot, height=None):
    """A function to display a SHAP plot in Streamlit."""
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)

# --- Main App UI ---
st.set_page_config(layout="wide")
st.title('Health Equity & Policy Simulation Tool')

st.sidebar.header('Simulation Parameters')

# --- Data and Scaler Loading ---
@st.cache_data
def load_data_and_scaler(path):
    """Loads data, prepares it, and fits a scaler."""
    df = pd.read_csv(path, compression='zip', dtype={"statefips": str, "countyfips": str})
    df['countyfips'] = df['countyfips'].str.zfill(5)
    
    # Clean county_name column to remove NaNs, trailing spaces, and empty strings
    df.dropna(subset=['county_name'], inplace=True)
    df['county_name'] = df['county_name'].str.strip()
    df = df[df['county_name'] != '']
    
    # Fit a scaler ONLY on the numeric features used by the model
    numeric_features = ['year', 'statefips', 'countyfips', 'agecat', 'racecat', 'sexcat', 'iprcat']
    scaler = StandardScaler().fit(df[numeric_features])
    
    return df, scaler

df, scaler = load_data_and_scaler(PROCESSED_DATA_PATH)
features = ['year', 'statefips', 'countyfips', 'agecat', 'racecat', 'sexcat', 'iprcat']

# --- User Inputs ---
year = st.sidebar.selectbox('Year', sorted(df['year'].unique()))
state = st.sidebar.selectbox('State', sorted(df['state_name'].unique()))
county = st.sidebar.selectbox('County', sorted(df[df['state_name'] == state]['county_name'].unique()))
age = st.sidebar.selectbox('Age Group', sorted(df['agecat_label'].unique()))
race = st.sidebar.selectbox('Race/Ethnicity', sorted(df['racecat_label'].dropna().unique()))
sex = st.sidebar.selectbox('Sex', sorted(df['sexcat_label'].unique()))
income = st.sidebar.selectbox('Income Level', sorted(df['iprcat_label'].unique()))

# --- Main Logic ---
if st.sidebar.button('Run Simulation'):
    # --- Data Filtering ---
    statefips = df[df['state_name'] == state]['statefips'].iloc[0]
    countyfips = df[df['county_name'] == county]['countyfips'].iloc[0]
    agecat = df[df['agecat_label'] == age]['agecat'].iloc[0]
    racecat = df[df['racecat_label'] == race]['racecat'].iloc[0]
    sexcat = df[df['sexcat_label'] == sex]['sexcat'].iloc[0]
    iprcat = df[df['iprcat_label'] == income]['iprcat'].iloc[0]

    input_data = pd.DataFrame([[year, statefips, countyfips, agecat, racecat, sexcat, iprcat]], columns=features)
    input_data_numeric = input_data.apply(pd.to_numeric)

    # --- Model Prediction ---
    lgbm_model, dl_model, explainer = load_models_and_explainer()
    
    # Predict with LightGBM (expects unscaled data)
    lgbm_pred = lgbm_model.predict(input_data_numeric)[0]
    
    # Scale data for the Deep Learning model and then predict
    input_data_scaled = scaler.transform(input_data_numeric)
    dl_pred = dl_model.predict(input_data_scaled)[0][0]

    st.subheader('Prediction Results')
    col1, col2 = st.columns(2)
    col1.metric("Predicted Uninsured Rate (LightGBM)", f"{lgbm_pred:.2f}%")
    col2.metric("Predicted Uninsured Rate (Deep Learning)", f"{dl_pred:.2f}%")
    
    st.markdown("---")

    # --- Comparative Analysis ---
    st.subheader("Comparative Analysis")
    st.markdown(f"Averages are shown for the **{age}** age group in the selected year for a more stable comparison.")
    
    # Use a broader filter for more reliable averages
    broad_filter = (
        (df['agecat_label'] == age) &
        (df['year'] == year)
    )
    
    # State Average
    state_avg = df[broad_filter & (df['state_name'] == state)]['PCTUI'].mean()
    
    # National Average
    national_avg = df[broad_filter]['PCTUI'].mean()
    
    # Get actual county rate for the original, specific group if available
    specific_filter = (
        (df['agecat_label'] == age) & (df['racecat_label'] == race) &
        (df['sexcat_label'] == sex) & (df['iprcat_label'] == income) &
        (df['year'] == year) &
        (df['county_name'] == county)
    )
    county_actual_df = df[specific_filter]
    county_actual = county_actual_df['PCTUI'].iloc[0] if not county_actual_df.empty else None

    col1, col2, col3 = st.columns(3)
    col1.metric(f"Actual Rate in {county} (for specific group)", f"{county_actual:.2f}%" if county_actual is not None else "N/A")
    col2.metric(f"State Average (for {age})", f"{state_avg:.2f}%" if pd.notna(state_avg) else "N/A")
    col3.metric(f"National Average (for {age})", f"{national_avg:.2f}%" if pd.notna(national_avg) else "N/A")

    st.markdown("---")

    # --- Historical Trend Analysis ---
    st.subheader(f"Historical Trend for Selected Group in {county}")
    historical_data = df[
        (df['county_name'] == county) & (df['agecat_label'] == age) &
        (df['racecat_label'] == race) & (df['sexcat_label'] == sex) &
        (df['iprcat_label'] == income)
    ]
    
    if not historical_data.empty:
        historical_data = historical_data.set_index('year')
        st.line_chart(historical_data['PCTUI'])
    else:
        st.warning("No historical data available for this specific demographic combination.")

    st.markdown("---")

    # --- Map-Based Visualization ---
    st.subheader(f"Uninsured Rate Overview for {state} in {year}")
    map_data = df[(df['state_name'] == state) & (df['year'] == year)]
    county_map_data = map_data.groupby('countyfips')['PCTUI'].mean().reset_index()

    fig = px.choropleth(
        county_map_data, geojson=counties_geojson, locations='countyfips',
        color='PCTUI', color_continuous_scale="Viridis", scope="usa",
        labels={'PCTUI': 'Avg. Uninsured Rate (%)'}
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- SHAP Explanation ---
    st.subheader('Why did the LightGBM model make this prediction?')
    shap_values = explainer.shap_values(input_data_numeric)
    st.info("The plot below shows the contribution of each factor to the final prediction. Features in red increased the predicted uninsured rate, while features in blue decreased it.")
    st_shap(shap.force_plot(explainer.expected_value, shap_values, input_data_numeric))

else:
    st.info('Please select parameters in the sidebar and click "Run Simulation".')
