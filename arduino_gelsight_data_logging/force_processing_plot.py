import os
import pandas as pd
import matplotlib.pyplot as plt

def extract_force_segment(file_path, output_folder):
    # Read CSV file
    df = pd.read_csv(file_path)

    # Ensure the data contains a force column
    force_col = [col for col in df.columns if 'Force' in col or 'force' in col]
    if not force_col:
        print(f"No force column found in {file_path}")
        return None, None
    force_col = force_col[0]  # Select the first matching column

    # Find where force starts changing
    start_idx = df[force_col].gt(2).idxmax() - 2
    end_idx = start_idx + 1200

    # Extract relevant segment
    extracted_df = df.iloc[start_idx:end_idx].copy()

    # Remove original time column if it exists
    time_columns = [col for col in extracted_df.columns if 'Time' in col or 'time' in col]
    extracted_df.drop(columns=time_columns, errors='ignore', inplace=True)

    # Redefine time column
    extracted_df['Time'] = range(len(extracted_df))
    extracted_df['Time'] = extracted_df['Time'] / 1000  # Convert to seconds

    # Reorder columns so Time is first
    extracted_df = extracted_df[['Time', force_col]]

    # Save processed CSV
    os.makedirs(output_folder, exist_ok=True)
    output_csv_path = os.path.join(output_folder, os.path.basename(file_path))
    extracted_df.to_csv(output_csv_path, index=False)
    print(f"Processed CSV saved: {output_csv_path}")

    return extracted_df, force_col

def process_and_plot_single_file(file_path, output_folder):
    extracted_df, force_col = extract_force_segment(file_path, output_folder)
    if extracted_df is not None:
        plt.figure(figsize=(4, 3))
        plt.plot(extracted_df['Time'], extracted_df[force_col], label=os.path.basename(file_path))
        plt.xlabel("Time (s)")
        plt.ylabel("Force")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()  # Ensure the layout does not crop ticks

        output_plot_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(file_path))[0]}.png")
        plt.savefig(output_plot_path, bbox_inches='tight')  # Save with tight bounding box to prevent cropping
        plt.close()
        print(f"Saved plot: {output_plot_path}")

# Set input file and output folder
input_file = "./Force_data/full_cola_can_2.csv"  # Replace with your CSV file
output_folder = "./Force_for_vqvae_drawing/processed"

process_and_plot_single_file(input_file, output_folder)
