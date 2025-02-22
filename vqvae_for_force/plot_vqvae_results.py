import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Set global font to Consolas at 8 pt.
plt.rcParams.update({
    'font.size': 8,
    'font.family': 'Consolas'
})

results_file = "vqvae_evaluation_results.txt"

true_labels = []
pred_labels = []
probabilities = []  # list of probability lists for each sample
sample_ids = []
overall_accuracy = None

# Read and parse the evaluation results file.
with open(results_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith("Overall Accuracy"):
            overall_accuracy = float(line.split(":")[1].strip().rstrip("%"))
        elif line.startswith("Sample"):
            try:
                # Extract sample id.
                sample_id = int(line.split(":")[0].split()[1])
                sample_ids.append(sample_id)
                # Extract the mapped true label (as an integer).
                true_label_str = line.split("mapped:")[1].split(")")[0].strip()
                true_label = int(true_label_str)
                true_labels.append(true_label)
                # Extract the predicted label.
                pred_label_str = line.split("Predicted label =")[1].split(",")[0].strip()
                pred_label = int(pred_label_str)
                pred_labels.append(pred_label)
                # Extract probabilities.
                prob_str = line.split("Probabilities =")[1].strip()
                prob_str = prob_str.replace("[", "").replace("]", "")
                prob_values = [float(x) for x in prob_str.split()]
                probabilities.append(prob_values)
            except Exception as e:
                print("Error parsing line:", line)
                continue

# Convert lists to numpy arrays.
true_labels = np.array(true_labels)
pred_labels = np.array(pred_labels)
sample_ids = np.array(sample_ids)

# Define the mapping from label indices to descriptive strings.
label_id_to_str = {0: "irregular", 1: "minimal", 2: "slightly", 3: "uniform"}

# Plot 1: Confusion Matrix
cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1, 2, 3])

# Normalize the confusion matrix by row (true labels).
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(3.5, 1.5))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", cbar=False,
            xticklabels=[label_id_to_str[i] for i in range(4)],
            yticklabels=[label_id_to_str[i] for i in range(4)])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
# plt.title(f"Overall Accuracy: {overall_accuracy:.2f}%")
plt.tight_layout()
plt.savefig("valid_vqvae_confusion.png", dpi=600)
plt.show()


# Plot 2: Histogram of Label Distribution
plt.figure(figsize=(3.5, 1.5))
bins = np.arange(-0.5, 4.5, 1)
plt.hist(true_labels, bins=bins, alpha=0.5, label="True", facecolor="blue", edgecolor="black", density=True)
plt.hist(pred_labels, bins=bins, alpha=0.5, label="Predicted", facecolor="green", edgecolor="black", density=True)
plt.xticks(range(4), [label_id_to_str[i] for i in range(4)])
plt.xlabel("Label")
plt.ylabel("Density")
# plt.title("Normalized Label Distribution")
plt.legend()
plt.tight_layout()
plt.savefig("valid_vqvae_distribution.png", dpi=600)
plt.show()



# Plot 3: Sample-wise Prediction Scatter Plot
correct = (true_labels == pred_labels)
plt.figure(figsize=(3.5, 2.5))
plt.scatter(sample_ids[correct], pred_labels[correct], color='green', label="Correct", s=10)
plt.scatter(sample_ids[~correct], pred_labels[~correct], color='blue', label="Incorrect", s=10)
plt.xlabel("Sample ID")
plt.ylabel("Predicted Label")
plt.yticks(range(4), [label_id_to_str[i] for i in range(4)])
plt.title("Predictions per Sample")
plt.legend(fontsize=6)
plt.tight_layout()
plt.show()
