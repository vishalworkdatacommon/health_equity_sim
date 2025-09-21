import pandas as pd
import glob
import os

def load_and_merge_data(data_path):
    """
    Loads all CSV files from the specified path, merges them, and returns a single DataFrame.

    Args:
        data_path (str): The path to the directory containing the CSV files.

    Returns:
        pandas.DataFrame: The merged DataFrame.
    """
    csv_files = glob.glob(os.path.join(data_path, 'sahie_*.csv'))
    df_list = []
    for filename in csv_files:
        df = pd.read_csv(filename)
        df_list.append(df)
    
    merged_df = pd.concat(df_list, ignore_index=True)
    return merged_df

if __name__ == '__main__':
    data_path = 'input_files'
    merged_data = load_and_merge_data(data_path)
    print("Successfully loaded and merged the data.")
    print("Merged data shape:", merged_data.shape)
    print("\nFirst 5 rows of the merged data:")
    print(merged_data.head())
    print("\nColumns in the merged data:")
    print(merged_data.columns)
    print("\nData types of the columns:")
    print(merged_data.info())
