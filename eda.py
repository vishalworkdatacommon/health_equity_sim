import pandas as pd
from data_processing import load_and_merge_data

def clean_data(df):
    """
    Cleans the merged SAHIE DataFrame.

    Args:
        df (pandas.DataFrame): The merged DataFrame.

    Returns:
        pandas.DataFrame: The cleaned DataFrame.
    """
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Drop the 'Unnamed: 25' column if it exists
    if 'Unnamed: 25' in df.columns:
        df = df.drop(columns=['Unnamed: 25'])
        
    # Convert object columns to numeric, coercing errors to NaN
    for col in ['NIPR', 'nipr_moe', 'NUI', 'nui_moe', 'NIC', 'nic_moe', 
                'PCTUI', 'pctui_moe', 'PCTIC', 'pctic_moe', 'PCTELIG', 
                'pctelig_moe', 'PCTLIIC', 'pctliic_moe']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    return df

if __name__ == '__main__':
    data_path = 'input_files'
    merged_data = load_and_merge_data(data_path)
    
    print("Cleaning the data...")
    cleaned_data = clean_data(merged_data)
    
    print("\nData cleaning complete.")
    print("Cleaned data shape:", cleaned_data.shape)
    
    print("\nData types after cleaning:")
    print(cleaned_data.info())
    
    print("\nSummary statistics for numeric columns:")
    print(cleaned_data.describe())
    
    print("\nValue counts for 'agecat':")
    print(cleaned_data['agecat'].value_counts())
    
    print("\nValue counts for 'racecat':")
    print(cleaned_data['racecat'].value_counts())
    
    print("\nValue counts for 'sexcat':")
    print(cleaned_data['sexcat'].value_counts())
    
    print("\nValue counts for 'iprcat':")
    print(cleaned_data['iprcat'].value_counts())

    print("\nMissing values per column:")
    print(cleaned_data.isnull().sum())
