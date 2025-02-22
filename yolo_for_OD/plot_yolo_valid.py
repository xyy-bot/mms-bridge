import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from supervision.metrics import MeanAveragePrecision, MetricTarget

# Updated custom detection class with is_empty() method.
class Detections:
    def __init__(self, xyxy, class_id, confidence, mask=None, tracker_id=None, data = None):
        self.xyxy = np.array(xyxy)         # shape: (N, 4)
        self.class_id = np.array(class_id)   # shape: (N,)
        self.confidence = np.array(confidence)  # shape: (N,)
        self.mask = mask  # For segmentation masks; set to None if not available.
        self.tracker_id = tracker_id  # For object tracking; set to None if not available.
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
    # Pass mask and tracker_id as None.
    return Detections(det_dict["xyxy"], det_dict["class_id"], conf, mask=None, tracker_id=None, data= None)



# ---------------------------
# 1. Load Saved Results
# ---------------------------
with open("yolov8n_results.json", "r") as f:
    data = json.load(f)

predictions_list = data["predictions"]
targets_list = data["targets"]

# Convert dictionaries to detection objects.
pred_detections = [dict_to_detections(pred, is_prediction=True) for pred in predictions_list]
gt_detections = [dict_to_detections(t, is_prediction=False) for t in targets_list]

# ---------------------------
# 2. Compute mAP Using Supervision
# ---------------------------
map_metric = MeanAveragePrecision(metric_target=MetricTarget.BOXES)
map_metric.update(pred_detections, gt_detections)
map_result = map_metric.compute()
print("mAP result:", map_result)

# ---------------------------
# 3. Compute and Plot the Confusion Matrix
# ---------------------------
all_preds = []
all_targets = []
for det in pred_detections:
    if not det.is_empty():
        all_preds.extend(det.class_id.tolist())
for det in gt_detections:
    if not det.is_empty():
        all_targets.extend(det.class_id.tolist())

cm = confusion_matrix(all_targets, all_preds)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

CLASSES = ['banana', 'beer can', 'cola can', 'cucumber', 'eggplant',
           'garlic', 'green bell pepper', 'lemon', 'potato',
           'red bell pepper', 'soft roller', 'tomato', 'water bottle']

plt.figure(figsize=(10, 8))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Normalized Confusion Matrix")
plt.show()
