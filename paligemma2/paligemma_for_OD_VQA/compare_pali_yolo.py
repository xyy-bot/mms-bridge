import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from supervision.metrics import MeanAveragePrecision, MetricTarget
plt.rcParams.update({
    'font.size': 8,
    'font.family': 'Consolas'
})
# -------------------------------
# 1. Detection Class & Conversion
# -------------------------------
class Detections:
    def __init__(self, xyxy, class_id, confidence, mask=None, tracker_id=None, data=None):
        self.xyxy = np.array(xyxy)         # shape: (N, 4)
        self.class_id = np.array(class_id)   # shape: (N,)
        self.confidence = np.array(confidence)  # shape: (N,)
        self.mask = mask  # For segmentation masks; set to None if not available.
        self.tracker_id = tracker_id  # For tracking; set to None if not available.
        self.data = data
    def __len__(self):
        return len(self.xyxy)
    def is_empty(self):
        return len(self.xyxy) == 0

def dict_to_detections(det_dict, is_prediction=True):
    if is_prediction:
        conf = det_dict.get("confidence", [1.0] * len(det_dict["xyxy"]))
    else:
        conf = [1.0] * len(det_dict["xyxy"])
    return Detections(det_dict["xyxy"], det_dict["class_id"], conf, mask=None, tracker_id=None, data=None)

# -------------------------------
# 2. Exclusion and Filtering Functions
# -------------------------------
EXCLUDED_CLASSES = ["beer can", "soft roller"]

def filter_detections(det, CLASSES, excluded=EXCLUDED_CLASSES):
    if det.is_empty():
        return det
    # Keep indices whose class name (looked up via CLASSES) is not excluded.
    indices = [i for i, cls in enumerate(det.class_id.tolist()) if CLASSES[cls] not in excluded]
    if len(indices) == 0:
        return Detections([], [], [])
    filtered_xyxy = det.xyxy[indices] if indices else np.empty((0,4))
    filtered_class_id = det.class_id[indices] if indices else np.empty((0,), dtype=int)
    filtered_conf = det.confidence[indices] if indices else np.empty((0,))
    return Detections(filtered_xyxy, filtered_class_id, filtered_conf, mask=det.mask, tracker_id=det.tracker_id, data=det.data)

def pair_filtered_class_ids(pred_det, gt_det, CLASSES, excluded=EXCLUDED_CLASSES):
    # Convert class ids to names.
    pred_ids = pred_det.class_id.tolist()
    gt_ids = gt_det.class_id.tolist()
    pred_names = [CLASSES[c] for c in pred_ids]
    gt_names = [CLASSES[c] for c in gt_ids]
    # Filter out indices for detections in the excluded set.
    pred_filtered = [pred_ids[i] for i, name in enumerate(pred_names) if name not in excluded]
    gt_filtered = [gt_ids[i] for i, name in enumerate(gt_names) if name not in excluded]
    n = min(len(pred_filtered), len(gt_filtered))
    return pred_filtered[:n], gt_filtered[:n]

# -------------------------------
# 3. Load Results for Both Models
# -------------------------------

# YOLOv8n results
with open("yolov8n_results.json", "r") as f:
    yolo_data = json.load(f)
predictions_list_yolo = yolo_data["predictions"]
targets_list_yolo = yolo_data["targets"]
pred_detections_yolo = [dict_to_detections(pred, is_prediction=True) for pred in predictions_list_yolo]
gt_detections_yolo = [dict_to_detections(t, is_prediction=False) for t in targets_list_yolo]

# Paligemma2 results
with open("predictions.json", "r") as f:
    pal_predictions = json.load(f)
with open("targets.json", "r") as f:
    pal_targets = json.load(f)
pred_detections_pal = [dict_to_detections(pred, is_prediction=True) for pred in pal_predictions]
gt_detections_pal = [dict_to_detections(t, is_prediction=False) for t in pal_targets]

# Full class list from your dataset.yaml.
CLASSES = ['banana', 'beer can', 'cola can', 'cucumber', 'eggplant', 'garlic',
           'green bell pepper', 'lemon', 'potato', 'red bell pepper', 'soft roller', 'tomato', 'water bottle']

# Create filtered class list for plotting (exclude "beer can" and "soft roller").
CLASSES_FILTERED = [cls for cls in CLASSES if cls not in EXCLUDED_CLASSES]
# Get indices for filtered classes.
labels_filtered = [CLASSES.index(cls) for cls in CLASSES_FILTERED]

# -------------------------------
# 4. Filter Detections for mAP Computation
# -------------------------------
pred_detections_yolo_filt = [filter_detections(det, CLASSES) for det in pred_detections_yolo]
gt_detections_yolo_filt = [filter_detections(det, CLASSES) for det in gt_detections_yolo]
pred_detections_pal_filt = [filter_detections(det, CLASSES) for det in pred_detections_pal]
gt_detections_pal_filt = [filter_detections(det, CLASSES) for det in gt_detections_pal]

# -------------------------------
# 5. Compute mAP for Both Models
# -------------------------------
map_metric_yolo = MeanAveragePrecision(metric_target=MetricTarget.BOXES)
map_metric_yolo.update(pred_detections_yolo_filt, gt_detections_yolo_filt)
map_result_yolo = map_metric_yolo.compute()
print("YOLO mAP (excluding 'beer can' & 'soft roller'):", map_result_yolo)

map_metric_pal = MeanAveragePrecision(metric_target=MetricTarget.BOXES)
map_metric_pal.update(pred_detections_pal_filt, gt_detections_pal_filt)
map_result_pal = map_metric_pal.compute()
print("Paligemma2 mAP (excluding 'beer can' & 'soft roller'):", map_result_pal)

# -------------------------------
# 6. Prepare Data for Confusion Matrix
# -------------------------------
all_preds_yolo = []
all_targets_yolo = []
for pred_det, gt_det in zip(pred_detections_yolo, gt_detections_yolo):
    p, t = pair_filtered_class_ids(pred_det, gt_det, CLASSES, excluded=EXCLUDED_CLASSES)
    all_preds_yolo.extend(p)
    all_targets_yolo.extend(t)

all_preds_pal = []
all_targets_pal = []
for pred_det, gt_det in zip(pred_detections_pal, gt_detections_pal):
    p, t = pair_filtered_class_ids(pred_det, gt_det, CLASSES, excluded=EXCLUDED_CLASSES)
    all_preds_pal.extend(p)
    all_targets_pal.extend(t)

# Check for consistent pairing
if len(all_preds_yolo) != len(all_targets_yolo):
    print("Warning: YOLO aggregated predictions and targets differ in length.")
if len(all_preds_pal) != len(all_targets_pal):
    print("Warning: Paligemma2 aggregated predictions and targets differ in length.")

# Compute confusion matrices using the filtered labels.
cm_yolo = confusion_matrix(all_targets_yolo, all_preds_yolo, labels=labels_filtered)
cm_yolo_norm = cm_yolo.astype('float') / cm_yolo.sum(axis=1)[:, np.newaxis]

cm_pal = confusion_matrix(all_targets_pal, all_preds_pal, labels=labels_filtered)
cm_pal_norm = cm_pal.astype('float') / cm_pal.sum(axis=1)[:, np.newaxis]

# -------------------------------
# 7. Plot Confusion Matrices Side-by-Side
# -------------------------------

import matplotlib.gridspec as gridspec

# --- Dummy Data (Replace with your actual computed values) ---


# Filter out the excluded classes and use their numeric label IDs.
CLASSES_FILTERED = [cls for cls in CLASSES if cls not in EXCLUDED_CLASSES]
# For plotting, we use the indices of the filtered classes (0, 1, 2, ...).
labels_filtered = list(range(len(CLASSES_FILTERED)))

# mAP values for demonstration (replace with your actual results)
yolo_map5095 = 0.9574
yolo_map50   = 0.9825
yolo_map75   = 0.9825

pal_map5095  = 0.9411
pal_map50    = 0.9847
pal_map75    = 0.9847

categories = ["mAP50-95", "mAP50", "mAP75"]
yolo_values = [yolo_map5095, yolo_map50, yolo_map75]
pal_values  = [pal_map5095, pal_map50, pal_map75]

# --- Create a 3×1 Grid (first subplot spans two rows) ---
fig = plt.figure(figsize=(3.5, 3.6))
# Define a grid with 3 rows and 1 column.
gs = gridspec.GridSpec(3, 1, height_ratios=[2, 2, 2])
# First subplot: spans rows 0 and 1.
ax1 = plt.subplot(gs[0:2, 0])
# Second subplot: occupies row 2.
ax2 = plt.subplot(gs[2, 0])

# --- Subplot 1: Combined Confusion Matrix ---
# Plot YOLO's matrix as background.
sns.heatmap(cm_yolo_norm, annot=False, cmap="Blues",
            xticklabels=labels_filtered, yticklabels=labels_filtered, ax=ax1, cbar=False)
# Overlay dual annotations in each cell.
n_rows, n_cols = cm_yolo_norm.shape
for i in range(n_rows):
    for j in range(n_cols):
        yolo_val = cm_yolo_norm[i, j]
        pal_val  = cm_pal_norm[i, j]
        # Leave blank if value is 0.
        yolo_text = "" if yolo_val == 0 else f"{yolo_val:.2f}"
        pal_text  = "" if pal_val  == 0 else f"{pal_val:.2f}"
        # Place YOLO value at a slightly higher vertical offset.
        ax1.text(j + 0.5, i + 0.35, yolo_text, ha="center", va="center",
                 color="orange", fontsize=7)
        # Place Paligemma2 value at a slightly lower vertical offset.
        ax1.text(j + 0.5, i + 0.75, pal_text, ha="center", va="center",
                 color="white", fontsize=7)
# ax1.set_title("Combined Confusion Matrix\n(Top: YOLOv8n, Bottom: Paligemma2)")
ax1.set_xlabel("Predicted Label")
ax1.set_ylabel("True Label")

# --- Subplot 2: mAP Bar Plot ---
x = np.arange(len(categories))  # label locations for bars
width = 0.35  # width of each bar
rects1 = ax2.bar(x - width/2, yolo_values, width, label="YOLOv8n", color="gray")
rects2 = ax2.bar(x + width/2, pal_values, width, label="PaliGemma2", color="skyblue")

ax2.set_ylabel("mAP Score")
# ax2.set_title("Comparison of mAP Metrics")
ax2.set_xticks(x)
ax2.set_xticklabels(categories)
ax2.legend(loc="lower right")
ax.set_ylim(0, 1.0)

def autolabel(rects, axis):
    for rect in rects:
        height = rect.get_height()
        axis.annotate(f"{height:.3f}",
                      xy=(rect.get_x() + rect.get_width()/2, height-0.2),
                      xytext=(0, 3),  # vertical offset
                      textcoords="offset points",
                      ha="center", va="bottom")
autolabel(rects1, ax2)
autolabel(rects2, ax2)

plt.tight_layout()
plt.savefig("combined_cm_map.png", dpi=600)
plt.show()