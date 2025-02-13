import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

def extract_label(filename):
    """
    Extract the label from a filename.
    Assumes the filename has the pattern: <label>_<index>.csv
    For example:
       "aluminum cube_0.csv"   --> "aluminum cube"
       "fake_green_paprika_0.csv"  --> "fake_green_paprika"
    """
    # Get the base name (without path) and remove the extension
    base = os.path.splitext(os.path.basename(filename))[0]
    # Split from the right on underscore, limit to one split
    parts = base.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    else:
        return base

class TimeSeriesCSVFolderDataset(Dataset):
    def __init__(self, folder_path, has_header=True, delimiter=',', transform=None):
        """
        Args:
            folder_path (str): Path to the folder containing CSV files.
            has_header (bool): Whether the CSV files have a header row.
            delimiter (str): Delimiter used in the CSV files.
            transform (callable, optional): Optional transform to apply to the data.
        """
        self.folder_path = folder_path
        self.has_header = has_header
        self.delimiter = delimiter
        self.transform = transform

        # Get list of CSV files in the folder (ignore non-csv files)
        self.csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        self.csv_files = [f for f in self.csv_files if f.lower().endswith('.csv')]

        # Build list of label strings for each file
        self.labels_str = [extract_label(f) for f in self.csv_files]

        # Create a mapping from string labels to integer indices
        self.label_set = sorted(list(set(self.labels_str)))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.label_set)}
        print("Found labels:", self.label_to_idx)

    def __len__(self):
        return len(self.csv_files)

    def __getitem__(self, idx):
        csv_file = self.csv_files[idx]

        # Determine the number of header rows to skip.
        # If has_header is True, skip the first row.
        skiprows = 1 if self.has_header else 0

        # Load the CSV file; expected to have two columns: time and force.
        # You can adjust np.loadtxt parameters as needed (e.g., delimiter, encoding).
        try:
            data = np.loadtxt(csv_file, delimiter=self.delimiter, skiprows=skiprows)
        except Exception as e:
            raise RuntimeError(f"Error loading {csv_file}: {e}")

        # data should be a 2D array with shape (sequence_length, 2)
        # If necessary, you could perform additional processing (e.g., normalization, resampling).

        # Get the label as an integer using our mapping
        label_str = extract_label(csv_file)
        label = self.label_to_idx[label_str]

        # Convert data to a torch tensor (dtype float32)
        data_tensor = torch.tensor(data, dtype=torch.float32)

        # Apply optional transformation
        if self.transform:
            data_tensor = self.transform(data_tensor)

        # Return the time series and its label
        return data_tensor, label

# =======================
# Example usage
# =======================
if __name__ == '__main__':
    # Set the folder path containing your CSV files
    folder_path = '../arduino_gelsight_data_logging/force_processed'  # change this to your folder path

    # Create an instance of the dataset
    dataset = TimeSeriesCSVFolderDataset(folder_path, has_header=True, delimiter=',')

    # Print the number of samples and the label mapping
    print(f"Total samples: {len(dataset)}")
    print("Label mapping (string -> int):", dataset.label_to_idx)

    # Example: Retrieve a sample and print its shape and label
    sample_data, sample_label = dataset[2]
    print("Sample data shape:", sample_data.shape)  # e.g., (sequence_length, 2)
    print("Sample label (int):", sample_label)
