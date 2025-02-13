# inference.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import (
    DATA_FOLDER, TRAIN_DATA_FOLDER, HAS_HEADER, DELIMITER,
    INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS,
    NUM_EMBEDDINGS, EMBEDDING_DIM, COMMITMENT_COST, NUM_CLASSES,
    BATCH_SIZE, CHECKPOINT_DIR
)
from models.vqvae import VQVAEClassifier
from utils.dataset import TimeSeriesCSVFolderDataset
from utils.collate_fn import custom_collate_fn
from utils.transform import NormalizeTransform


def load_model_from_checkpoint(checkpoint_path, device):
    print("Loading VQVAE Model...")
    model = VQVAEClassifier(
        INPUT_SIZE,
        HIDDEN_SIZE,
        NUM_LAYERS,
        NUM_EMBEDDINGS,
        EMBEDDING_DIM,
        COMMITMENT_COST,
        OUTPUT_SIZE,
        NUM_CLASSES  # Number of classes as used during training.
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # Set to evaluation mode
    return model


def classify_sample_with_classifier(model, sample, device):
    """
    Classify a new sample using the classifier head of the model.

    Args:
        model: The trained VQVAEClassifier model.
        sample: A tensor of shape (seq_len, input_size) for a single sample.
        device: torch.device.

    Returns:
        predicted_label: The predicted class label (integer).
        class_probs: The softmax probabilities for each class.
    """
    model.eval()
    # Add batch dimension: (1, seq_len, input_size)
    sample = sample.unsqueeze(0).to(device)
    # Create a tensor for the actual sequence length.
    lengths = torch.tensor([sample.shape[1]], dtype=torch.long).to(device)

    with torch.no_grad():
        # The forward method returns (reconstructed, class_logits, vq_loss)
        _, class_logits, _ = model(sample, lengths)
        # Compute softmax probabilities if desired
        class_probs = torch.softmax(class_logits, dim=1)
        # Predicted label is the argmax over the logits.
        predicted_label = torch.argmax(class_logits, dim=1).item()

    return predicted_label, class_probs.cpu().numpy()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = f"{CHECKPOINT_DIR}/best_checkpoint.pth"
    model = load_model_from_checkpoint(checkpoint_path, device)

    # Load the inference dataset.
    transform = NormalizeTransform(force_divisor=100.0)
    test_dataset = TimeSeriesCSVFolderDataset(
        DATA_FOLDER,
        has_header=HAS_HEADER,
        delimiter=DELIMITER,
        transform=transform
    )

    # Build a mapping from label IDs to label strings.
    idx_to_label = {v: k for k, v in test_dataset.label_to_idx.items()}

    # Define your custom mapping from the original label string to the deformation label.
    custom_mapping = {'aluminum cube': 1, 'empty_cola_can': 0, 'empty_water_bottle': 0, 'fake_banana': 1, 'fake_cucumber': 1,
     'fake_eggplant': 1, 'fake_garlic': 1, 'fake_green_paprika': 1, 'fake_lemon': 1, 'fake_paprika': 1,
     'fake_potato': 1, 'fake_red_paprika': 1, 'fake_tomato': 1, 'full_cola_can': 1, 'full_water_bottle': 2,
     'real_banana': 1, 'real_cucumber': 1, 'real_eggplant': 1, 'real_garlic': 1, 'real_green_paprika': 1,
     'real_lemon': 1, 'real_potato': 1, 'real_red_paprika': 1, 'real_tomato': 1, 'rigid': 1, 'soft': 3}

    total = 0
    correct = 0
    for i in range(len(test_dataset)):
        sample, true_label = test_dataset[i]
        # Get the original label string from the dataset.
        true_label_str = idx_to_label.get(true_label, str(true_label))
        # Replace the true label string with your custom deformation label.
        # Here we assign a default value (e.g., 4) if the true label is not one of the four.
        true_label_mapped = custom_mapping.get(true_label_str, 5)

        predicted_label, probs = classify_sample_with_classifier(model, sample, device)
        total += 1
        print(
            f"Sample {i}: True label = {true_label_str} (mapped: {true_label_mapped}), "
            f"Predicted label = {predicted_label}, Probabilities = {probs}"
        )
        if predicted_label == true_label_mapped:
            correct += 1

    accuracy = correct / total if total > 0 else 0
    print("Overall Accuracy: {:.2f}%".format(accuracy * 100))
