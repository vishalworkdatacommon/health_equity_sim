
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from eda import clean_data
from data_processing import load_and_merge_data

def feature_engineer(df):
    """
    Performs feature engineering on the cleaned SAHIE DataFrame.

    Args:
        df (pandas.DataFrame): The cleaned DataFrame.

    Returns:
        pandas.DataFrame: The DataFrame with new features.
    """
    # Handle missing values
    df = df.dropna()

    # Create FIPS code
    df['fips'] = df['statefips'].astype(str).str.zfill(2) + df['countyfips'].astype(str).str.zfill(3)

    # Decode categorical variables
    age_map = {
        0: 'Under 65 years',
        1: '18 to 64 years',
        2: '40 to 64 years',
        3: '50 to 64 years',
        4: '21 to 64 years',
        5: '0 to 18 years'
    }
    race_map = {
        0: 'All races',
        1: 'White alone, not Hispanic',
        2: 'Black alone, not Hispanic',
        3: 'Hispanic (any race)'
    }
    sex_map = {
        0: 'Both sexes',
        1: 'Male',
        2: 'Female'
    }
    ipr_map = {
        0: 'All income levels',
        1: '<= 200% of poverty',
        2: '<= 250% of poverty',
        3: '<= 138% of poverty',
        4: '<= 400% of poverty',
        5: '139% to 400% of poverty'
    }

    df['agecat_label'] = df['agecat'].map(age_map)
    df['racecat_label'] = df['racecat'].map(race_map)
    df['sexcat_label'] = df['sexcat'].map(sex_map)
    df['iprcat_label'] = df['iprcat'].map(ipr_map)

    return df

def visualize_data(df):
    """
    Creates visualizations of the engineered data.

    Args:
        df (pandas.DataFrame): The engineered DataFrame.
    """
    # Set plot style
    sns.set_style("whitegrid")

    # Histogram of uninsured rate (PCTUI)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['PCTUI'], bins=30, kde=True)
    plt.title('Distribution of Uninsured Rate (PCTUI)')
    plt.xlabel('Percentage Uninsured')
    plt.ylabel('Frequency')
    plt.savefig('pctui_distribution.png')
    plt.close()

    # Average uninsured rate by state
    plt.figure(figsize=(12, 8))
    avg_uninsured_by_state = df.groupby('state_name')['PCTUI'].mean().sort_values(ascending=False)
    sns.barplot(x=avg_uninsured_by_state.values, y=avg_uninsured_by_state.index, palette='viridis')
    plt.title('Average Uninsured Rate by State')
    plt.xlabel('Average Percentage Uninsured')
    plt.ylabel('State')
    plt.tight_layout()
    plt.savefig('avg_uninsured_by_state.png')
    plt.close()
    
    print("Visualizations saved as pctui_distribution.png and avg_uninsured_by_state.png")


if __name__ == '__main__':
    data_path = 'input_files'
    merged_data = load_and_merge_data(data_path)
    cleaned_data = clean_data(merged_data)
    engineered_data = feature_engineer(cleaned_data)

    print("Feature engineering complete.")
    print("Engineered data shape:", engineered_data.shape)
    print("\nFirst 5 rows of the engineered data:")
    print(engineered_data.head())

    # Save the processed data locally within the project directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'sahie_processed.csv')
    engineered_data.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to {output_path}")
    
    visualize_data(engineered_data)
