import os
import shutil
import re

# Define source and target folders
source_folders = ["./Force_data_v2/processed", "../vqvae_for_force/dataset/data_for_force_v2"]  # Replace with your actual folder paths
target_folder = "../vqvae_for_force/dataset/data_for_force_merged"  # Replace with your actual target folder path

# Ensure target folder exists
os.makedirs(target_folder, exist_ok=True)

# Regular expression to match filenames like "empty_cola_can_0.csv"
pattern = re.compile(r"^(.*)_(\d+)\.csv$")


def get_next_filename(target_folder, filename):
    """ Check if file exists and return the next available indexed filename. """
    base, ext = os.path.splitext(filename)

    match = pattern.match(filename)
    if match:
        name, index = match.groups()
        index = int(index)
    else:
        name, index = base, 0  # Default index if no number is found

    # Find the next available index
    while os.path.exists(os.path.join(target_folder, f"{name}_{index}{ext}")):
        index += 1

    return f"{name}_{index}{ext}"


# Move and rename files
for folder in source_folders:
    for filename in os.listdir(folder):
        src_path = os.path.join(folder, filename)

        if os.path.isfile(src_path):  # Check if it's a file
            new_filename = get_next_filename(target_folder, filename)
            dest_path = os.path.join(target_folder, new_filename)

            shutil.move(src_path, dest_path)  # Move file to target folder
            print(f"Moved: {src_path} -> {dest_path}")

print("File merging completed!")