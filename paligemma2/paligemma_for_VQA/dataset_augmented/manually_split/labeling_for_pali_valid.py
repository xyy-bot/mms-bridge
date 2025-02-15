import os
import json

# ---------------------------
# Description mapping for suffix
# ---------------------------
description = {
    "full_water_bottle": "smooth, no bumps, no texture, it is full.",
    "empty_water_bottle": "smooth, no bumps, no texture, it is empty.",
    "empty_cola_can": "smooth, no bumps, no texture, it is empty.",
    "full_cola_can": "smooth, no bumps, no texture, it is full.",
    "fake_banana": "smooth, a rising edge, no texture, it is fake.",
    "real_banana": "slightly rough, sparse dimples, no texture, it is real.",
    "fake_cucumber": "smooth, slightly bumpy, no texture, it is fake.",
    "real_cucumber": "rough, uneven, a few grooves, it is real.",
    "fake_eggplant": "sparse dots, uneven, not texture, it is fake.",
    "real_eggplant": "smooth, uniform, no texture, it is real.",
    "fake_garlic": "rough, dense groove, directional ridges, it is fake.",
    "real_garlic": "mostly smooth, no bumps, few small ridges, it is real.",
    "fake_green_bell_pepper": "mostly smooth, a few small dimples, no texture, it is fake.",
    "real_green_bell_pepper": "smooth, no bumps, wrinkled texture, it is real.",
    "fake_lemon": "rough, highly bumpy, many grainy dots, it is fake.",
    "real_lemon": "rough, evenly distributed bumps, grid-like texture, it is real.",
    "fake_potato": "smooth, no bumps, no texture, it is fake.",
    "real_potato": "rough, small indentations, wrinkled texture, it is real.",
    "fake_red_bell_pepper": "mostly smooth, no bumps, no texture, it is fake.",
    "real_red_bell_pepper": "smooth, a few small dots, wrinkled texture, it is real.",
    "fake_tomato": "mostly smooth, uneven, a few sparse dots, it is fake.",
    "real_tomato": "smooth, uniform, no texture, it is real."
}

# ---------------------------
# Deformation mapping for prefix
# ---------------------------
deformation_mapping = {
    "empty": "irregular deformation",
    "full": "minimal deformation",
    "real": "minimal deformation",
    "fake": "minimal deformation"
}


def parse_folder_name(folder_name):
    """
    Parse the folder name to extract authenticity and class label.

    Expected folder name examples:
      - "real_potato_1"
      - "fake_tomato_4"
      - "empty_cola_can_0"
      - "full_cola_can_3"

    This function splits the folder name on underscores, and if the last token
    is numeric it is dropped.

    Returns:
        authenticity (str): one of "empty", "full", "real", or "fake".
        class_label (str): the object label (e.g., "cola can", "potato").
    """
    tokens = folder_name.split('_')
    if tokens[-1].isdigit():
        tokens = tokens[:-1]
    authenticity = tokens[0].lower()
    class_label = " ".join(tokens[1:]).lower()
    return authenticity, class_label


# ---------------------------
# User configuration
# ---------------------------
dataset_dir = './validation'  # Directory containing your subfolders (e.g., "empty_cola_can_0", "real_potato_1", etc.)
output_jsonl = 'annotation_for_paligemma_valid.jsonl'  # Output JSONL file

# ---------------------------
# Build JSONL records
# ---------------------------
records = []

# Iterate over every subfolder in the dataset directory.
for folder in os.listdir(dataset_dir):
    folder_path = os.path.join(dataset_dir, folder)
    if os.path.isdir(folder_path):
        # Parse folder name to get authenticity and class label.
        authenticity, class_label = parse_folder_name(folder)

        # Build a key for description mapping.
        # Replace spaces with underscores in class_label.
        desc_key = f"{authenticity}_{class_label.replace(' ', '_')}"
        suffix_text = description.get(desc_key, "No description available.")

        # Get deformation description.
        deformation = deformation_mapping.get(authenticity, "unknown deformation")
        if authenticity == 'full' and class_label == "water bottle":
            deformation = "slightly deformation"

        # Build the prefix text based on authenticity.
        if authenticity in ['empty', 'full']:
            prefix_text = (f"Describe the tactile image for {class_label} regarding surface property, "
                           f"and tell me if it's full or empty as it underwent {deformation} when grasping.")
        elif authenticity in ['real', 'fake']:
            prefix_text = (f"Describe the tactile image for {class_label} and tell me if it's real or fake "
                           f"as it underwent {deformation} when grasping.")
        else:
            prefix_text = (f"Describe the tactile image for {class_label} and tell me it's {authenticity} "
                           f"as it underwent {deformation} when grasping.")

        # Get a sorted list of all JPG files in the folder.
        image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')])

        for image_file in image_files:
            # Build the relative image path (e.g., "real_potato_1/frame_0050.jpg").
            relative_image_path = os.path.join(folder, image_file)
            # Create the JSON record.
            record = {
                "image": relative_image_path,
                "prefix": prefix_text,
                "suffix": suffix_text
            }
            records.append(record)

# ---------------------------
# Write the records to a JSONL file.
# ---------------------------
with open(output_jsonl, 'w') as f:
    for record in records:
        json_line = json.dumps(record)
        f.write(json_line + "\n")

print(f"JSONL file '{output_jsonl}' has been created with {len(records)} records.")
