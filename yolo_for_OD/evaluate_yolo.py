import os
import glob
import time
import json
import yaml
import numpy as np
from PIL import Image
from ultralytics import YOLO

# -------------------------------
# 1. Load Dataset Config & Model
# -------------------------------

# Load the dataset configuration.
config_path = "/home/au-robotics/PycharmProjects/object_detection_xy/yolo_for_OD/dataset/dataset.yaml"
with open(config_path, "r") as f:
    dataset_config = yaml.safe_load(f)

# Get the class names from the config.
CLASSES = dataset_config["names"]

# Build the full validation images directory.
valid_dir = os.path.join(dataset_config["path"], dataset_config["val"])
# Build the full validation labels directory. (Assuming labels are stored in "valid/labels")
labels_dir = os.path.join(dataset_config["path"], "valid", "labels")

# Load the YOLOv8 model from the fine-tuned checkpoint.
model = YOLO("./runs/detect/train11/weights/best.pt")

# -------------------------------
# 2. Prepare Validation Dataset
# -------------------------------

# Get image files from the validation directory (adjust extensions if needed).
image_paths = glob.glob(os.path.join(valid_dir, "*.jpg")) + glob.glob(os.path.join(valid_dir, "*.png"))


# -------------------------------
# 3. Helper Function for Ground Truth
# -------------------------------

def load_ground_truth(label_path, image_width, image_height):
    """
    Load YOLO-format ground truth from a label file.
    Each line should be: <class> <x_center> <y_center> <width> <height>
    (with coordinates normalized to [0,1]).
    Returns:
      - boxes: list of [x1, y1, x2, y2] in absolute pixel coordinates.
      - class_ids: list of class indices.
    """
    boxes = []
    class_ids = []
    if not os.path.exists(label_path):
        return boxes, class_ids
    with open(label_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, x_center, y_center, w_norm, h_norm = map(float, parts)
        # Convert normalized values to absolute pixel coordinates.
        x_center *= image_width
        y_center *= image_height
        w_abs = w_norm * image_width
        h_abs = h_norm * image_height
        x1 = x_center - w_abs / 2
        y1 = y_center - h_abs / 2
        x2 = x_center + w_abs / 2
        y2 = y_center + h_abs / 2
        boxes.append([x1, y1, x2, y2])
        class_ids.append(int(cls))
    return boxes, class_ids


# -------------------------------
# 4. Run Inference & Record Results
# -------------------------------

predictions = []
targets = []
inference_times = []
image_names = []

for img_path in image_paths:
    # Open image.
    image = Image.open(img_path).convert("RGB")
    w, h = image.size
    image_names.append(os.path.basename(img_path))

    # Construct label file path.
    # Get the base filename (without extension) and append ".txt" in the labels directory.
    base_filename = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(labels_dir, base_filename + ".txt")

    # Load ground truth.
    gt_boxes, gt_class_ids = load_ground_truth(label_path, w, h)
    target_dict = {
        "xyxy": gt_boxes,  # List of ground truth bounding boxes.
        "class_id": gt_class_ids  # Corresponding class indices.
    }
    targets.append(target_dict)

    # Run inference and record inference time.
    start_time = time.time()
    results = model(img_path)  # Model accepts an image path.
    inference_time = time.time() - start_time
    inference_times.append(inference_time)

    # Process YOLOv8 output. Here, we take the first result (one per image).
    res = results[0]
    if res.boxes is not None and len(res.boxes) > 0:
        boxes = res.boxes.xyxy.cpu().numpy()  # shape: (N, 4)
        confidences = res.boxes.conf.cpu().numpy()  # shape: (N,)
        class_ids = res.boxes.cls.cpu().numpy().astype(int)  # shape: (N,)
    else:
        boxes = np.array([])
        confidences = np.array([])
        class_ids = np.array([])

    pred_dict = {
        "xyxy": boxes.tolist() if boxes.size > 0 else [],
        "confidence": confidences.tolist() if confidences.size > 0 else [],
        "class_id": class_ids.tolist() if class_ids.size > 0 else []
    }
    predictions.append(pred_dict)

# -------------------------------
# 5. Save All Results to JSON
# -------------------------------

output = {
    "images": image_names,  # List of image filenames.
    "predictions": predictions,  # Predictions for each image.
    "targets": targets,  # Ground-truth annotations for each image.
    "inference_times": inference_times  # Inference time per image (in seconds).
}

with open("yolov8n_results.json", "w") as f:
    json.dump(output, f, indent=2)

print("YOLOv8n results saved to yolov8n_results.json")
