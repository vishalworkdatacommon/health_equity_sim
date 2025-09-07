import os
import pandas as pd
import pytest
from unittest.mock import MagicMock
import sys

# Add the app directory to the path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import clean_dataframe, main

# --- Test Fixtures ---
@pytest.fixture
def dirty_dataframe():
    """Creates a sample DataFrame with messy data for cleaning tests."""
    data = {
        'county_name': ['Good County', None, '  Spaced Out County  ', 'Another Good', '  ', ''],
        'countyfips': ['01001'] * 6,
        'PCTUI': [10.0] * 6
    }
    return pd.DataFrame(data)

# --- Tests ---
def test_clean_dataframe(dirty_dataframe):
    """Tests the standalone data cleaning logic."""
    cleaned_df = clean_dataframe(dirty_dataframe)
    
    assert len(cleaned_df) == 3
    assert 'Good County' in cleaned_df['county_name'].values
    assert 'Spaced Out County' in cleaned_df['county_name'].values
    assert 'Another Good' in cleaned_df['county_name'].values

def test_main_function_runs(mocker):
    """
    Tests that the main application function can be executed without raising an error.
    """
    # Mock the data and model loading functions within the app module
    mocker.patch('app.load_data_and_scaler', return_value=(pd.DataFrame({
        'year': [2022], 'state_name': ['Alabama'], 'county_name': ['Autauga County'],
        'agecat_label': ['21 to 64 years'], 'racecat_label': ['All races'],
        'sexcat_label': ['Both sexes'], 'iprcat_label': ['<= 138% of poverty'],
        'statefips': ['01'], 'countyfips': ['01001'], 'agecat': [1], 'racecat': [1],
        'sexcat': [1], 'iprcat': [1]
    }), MagicMock()))
    mocker.patch('app.load_geojson', return_value={})
    mocker.patch('app.load_models_and_explainer', return_value=(MagicMock(), MagicMock(), MagicMock()))
    mocker.patch('app.st.sidebar.button', return_value=False)

    try:
        main()
        assert True
    except Exception as e:
        pytest.fail(f"The main application function raised an unexpected exception: {e}")