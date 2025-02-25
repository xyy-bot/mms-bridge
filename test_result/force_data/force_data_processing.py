import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 8,
    'font.family': 'Consolas'
})
def extract_force_segment(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Ensure the data contains a force column (assumes the column name includes 'Force' or 'force')
    force_cols = [col for col in df.columns if 'Force' in col or 'force' in col]
    if not force_cols:
        print(f"No force column found in {file_path}")
        return None, None
    force_col = force_cols[0]  # Select the first matching column

    # Find the starting index where the force begins to change
    start_idx = df[force_col].gt(2).idxmax() - 2

    # Define the ending index (e.g., start index plus 1200 samples)
    end_idx = start_idx + 1200

    # Extract the corresponding data segment
    extracted_df = df.iloc[start_idx:end_idx].copy()

    # Remove any original time columns, if they exist
    time_columns = [col for col in extracted_df.columns if 'Time' in col or 'time' in col]
    extracted_df.drop(columns=time_columns, errors='ignore', inplace=True)

    # Create a new time column in seconds
    extracted_df['Time'] = range(len(extracted_df))  # time in ms
    extracted_df['Time'] = extracted_df['Time'] / 1000  # convert to seconds

    # Reorder columns so that Time comes first
    extracted_df = extracted_df[['Time', force_col]]

    return extracted_df, force_col

def plot_force_profile(file_path):
    df, force_col = extract_force_segment(file_path)
    if df is None:
        return

    plt.figure(figsize=(3.5, 2.5))
    plt.plot(df['Time'], df[force_col])
    plt.xlabel("Time (s)")
    plt.ylabel("Force")
    # plt.title("Force Profile")
    plt.grid(True)
    # plt.legend()
    plt.tight_layout()
    plt.savefig('garlic_force_profile.png', dpi=600)
    plt.show()

# Example usage:
csv_file = "pipeline_test_garlic_v3/garlic_1.csv"  # Replace with the path to your CSV file
plot_force_profile(csv_file)
