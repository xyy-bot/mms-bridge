# inference.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import DATA_FOLDER, TRAIN_DATA_FOLDER, HAS_HEADER, DELIMITER, INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZE, \
    NUM_LAYERS, \
    NUM_EMBEDDINGS, EMBEDDING_DIM, COMMITMENT_COST, NUM_CLASSES, BATCH_SIZE, CHECKPOINT_DIR
from models.vqvae import VQVAEClassifier
from utils.dataset import TimeSeriesCSVFolderDataset
from utils.collate_fn import custom_collate_fn
from utils.transform import NormalizeTransform


def load_model_from_checkpoint(checkpoint_path, device):
    model = VQVAEClassifier(
        INPUT_SIZE,
        HIDDEN_SIZE,
        NUM_LAYERS,
        NUM_EMBEDDINGS,
        EMBEDDING_DIM,
        COMMITMENT_COST,
        OUTPUT_SIZE,
        NUM_CLASSES  # Should be 2 when training only on rigid and uniform.
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def compute_class_latents(model, dataloader, device):
    """
    Pass the training data (rigid and uniform) through the model to extract discrete latent representations (z_q)
    and collect all the z_q's for each class.

    Returns:
        class_latents (dict): A dictionary mapping class index (0 or 1) to a tensor containing all z_q vectors
                              (shape: (num_samples, hidden_size)) for that class.
    """
    class_latents = {}
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x, labels, lengths = batch
            x = x.to(device)
            lengths = lengths.to(device)
            z_e = model.encoder(x, lengths)
            z_q, _ = model.vector_quantizer(z_e)  # (batch, hidden_size)
            z_q = z_q.cpu()  # Move to CPU for later similarity comparisons
            for latent, label in zip(z_q, labels):
                label = int(label.item())
                if label not in class_latents:
                    class_latents[label] = []
                class_latents[label].append(latent)
    # Convert lists to a single tensor per class.
    for label in class_latents:
        class_latents[label] = torch.stack(class_latents[label], dim=0)
    return class_latents


def classify_sample(model, sample, class_latents, device, sim_threshold=0.8):
    """
    Classify a new sample based on the maximum cosine similarity between its quantized latent representation (z_q)
    and each training z_q in the known classes.

    For each class, the function computes cosine similarities between the sample's z_q and every training z_q in that class,
    takes the maximum similarity value, and then selects the class with the highest maximum similarity. If that value is below
    sim_threshold, the sample is classified as irregular deformation (label 2).

    Args:
        model: The trained VQ-VAE model.
        sample: Tensor of shape (seq_len, input_size) for a single sample.
        class_latents: Dictionary mapping class index (0 or 1) to tensor of training z_q's for that class.
        device: torch.device.
        sim_threshold: Minimum cosine similarity required for a sample to be considered as one of the known classes.

    Returns:
        predicted_label: 0 or 1 if similar enough to one class, otherwise 2 (irregular deformation).
        similarities: Dictionary mapping each class index to the maximum cosine similarity value.
    """
    model.eval()
    sample = sample.unsqueeze(0).to(device)  # (1, seq_len, input_size)
    lengths = torch.tensor([sample.shape[1]], dtype=torch.long).to(device)
    with torch.no_grad():
        z_e = model.encoder(sample, lengths)
        z_q, _ = model.vector_quantizer(z_e)
    latent = z_q.squeeze(0)  # (hidden_size,)
    latent = latent.cpu()  # Ensure latent is on CPU

    similarities = {}
    max_sim = -1.0
    best_class = None
    # Loop over each class and compute cosine similarity with every training latent in that class.
    for class_idx, latents in class_latents.items():
        # latents: tensor of shape (num_samples, hidden_size)
        # Compute cosine similarity between latent (1, hidden_size) and each row in latents.
        sims = F.cosine_similarity(latent.unsqueeze(0), latents, dim=1)  # shape: (num_samples,)
        class_max_sim = sims.max().item()
        similarities[class_idx] = class_max_sim
        if class_max_sim > max_sim:
            max_sim = class_max_sim
            best_class = class_idx
    if max_sim < sim_threshold:
        predicted_label = 2  # irregular deformation
    else:
        predicted_label = best_class
    return predicted_label, similarities


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = f"{CHECKPOINT_DIR}/best_checkpoint.pth"
    model = load_model_from_checkpoint(checkpoint_path, device)

    # Load the training dataset (only rigid and uniform) to collect all training z_q's.
    transform = NormalizeTransform(force_divisor=100.0)
    train_dataset = TimeSeriesCSVFolderDataset(TRAIN_DATA_FOLDER, has_header=HAS_HEADER, delimiter=DELIMITER,
                                               transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
    class_latents = compute_class_latents(model, train_loader, device)

    # Load the inference dataset (which may include all three classes).
    test_dataset = TimeSeriesCSVFolderDataset(TRAIN_DATA_FOLDER, has_header=HAS_HEADER, delimiter=DELIMITER,
                                              transform=transform)

    # Build a mapping from label IDs to label strings for true labels.
    # This mapping comes from the dataset's label_to_idx dictionary.
    idx_to_label = {v: k for k, v in test_dataset.label_to_idx.items()}
    # Add an entry for irregular deformation if not present.
    # if 2 not in idx_to_label:
    #     idx_to_label[2] = "irregular deformation"

    total = 0
    correct = 0
    for i in range(len(test_dataset)):
        sample, true_label = test_dataset[i]
        predicted_label, sims = classify_sample(model, sample, class_latents, device, sim_threshold=0.5)
        total += 1
        true_label_str = idx_to_label.get(true_label, str(true_label))
        print(f"Sample {i}: True label = {true_label_str}, Predicted label = {predicted_label}, Similarities = {sims}")
        if predicted_label == true_label:
            correct += 1
    accuracy = correct / total if total > 0 else 0
    print("Overall Accuracy: {:.2f}%".format(accuracy * 100))

