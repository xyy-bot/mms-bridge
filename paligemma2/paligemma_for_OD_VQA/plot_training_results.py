import ast
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 8,
    'font.family': 'Consolas'
})
# Path to the text file containing the training results.
file_path = "finetune_10epoch_augmented_dataset_process.txt"

# Lists to hold the training metrics.
epochs = []
losses = []
grad_norms = []
learning_rates = []

# Open the file and parse each line.
with open(file_path, "r") as f:
    for line in f:
        line = line.strip()
        # Skip empty lines.
        if not line:
            continue
        try:
            # Convert the string representation of the dictionary to an actual dict.
            record = ast.literal_eval(line)
        except Exception as e:
            print("Error parsing line:", line)
            continue

        # For plotting training curves, we focus on lines that include a 'loss' key.
        if "loss" in record:
            epochs.append(record.get("epoch"))
            losses.append(record.get("loss"))
            # grad_norm and learning_rate might not be present in all records.
            grad_norms.append(record.get("grad_norm", None))
            learning_rates.append(record.get("learning_rate", None))

# Create subplots for loss, grad norm, and learning rate vs. epoch.
plt.figure(figsize=(3.5, 1.5))

plt.subplot(1, 2, 1)
plt.plot(epochs, losses, linestyle='-')
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
# plt.title("Training Loss")
plt.grid(True)

# plt.subplot(2, 2, 2)
# plt.plot(epochs, grad_norms, linestyle='-')
# plt.xlabel("Epoch")
# plt.ylabel("Gradient Norm")
# plt.title("Gradient Norm")
# plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, learning_rates, linestyle='-')
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
# plt.title("Learning Rate")
plt.grid(True)

plt.tight_layout()
plt.savefig("paligemma2_FT_loss.png", dpi=600)
plt.show()

