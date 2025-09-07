
import pandas as pd
import pytest

def clean_data(df):
    """A standalone function to test the cleaning logic."""
    df.dropna(subset=['county_name'], inplace=True)
    df['county_name'] = df['county_name'].str.strip()
    df = df[df['county_name'] != '']
    return df

@pytest.fixture
def dirty_dataframe():
    """Creates a sample DataFrame with messy data."""
    data = {
        'county_name': ['Good County', None, '  Spaced Out County  ', 'Another Good', '  ', ''],
        'PCTUI': [10.0] * 6
    }
    return pd.DataFrame(data)

def test_data_cleaning(dirty_dataframe):
    """Tests the standalone data cleaning logic."""
    cleaned_df = clean_data(dirty_dataframe)
    
    assert len(cleaned_df) == 3
    assert 'Good County' in cleaned_df['county_name'].values
    assert 'Spaced Out County' in cleaned_df['county_name'].values
    assert 'Another Good' in cleaned_df['county_name'].values
    assert None not in cleaned_df['county_name'].values
    assert '' not in cleaned_df['county_name'].values
    assert '  ' not in cleaned_df['county_name'].values
