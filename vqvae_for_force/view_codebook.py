import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

# Import your model, dataset, and configuration parameters
from models.vqvae import VQVAEClassifier
# from utils.artificial_data import ArtificialTimeSeriesDataset  # or your own dataset
from utils.collate_fn import custom_collate_fn
from utils.dataset import TimeSeriesCSVFolderDataset
from utils.transform import NormalizeTransform
from config import INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_EMBEDDINGS, EMBEDDING_DIM, \
    COMMITMENT_COST, OUTPUT_SIZE, NUM_CLASSES, DATA_FOLDER, HAS_HEADER, DELIMITER, BATCH_SIZE, TRAIN_DATA_FOLDER


def load_model_from_checkpoint(checkpoint_path, device):
    """
    Load the VQ-VAE model from the given checkpoint and set it to evaluation mode.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        device (torch.device): Device on which to load the model (CPU or GPU).

    Returns:
        model (VQVAEClassifier): The loaded VQ-VAE model.
    """
    model = VQVAEClassifier(
        INPUT_SIZE,
        HIDDEN_SIZE,
        NUM_LAYERS,
        NUM_EMBEDDINGS,
        EMBEDDING_DIM,
        COMMITMENT_COST,
        OUTPUT_SIZE,
        NUM_CLASSES
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def get_encoder_representations(model, dataloader, device):
    """
    Iterate through the dataloader, pass each batch through the encoder to obtain
    the continuous latent representation z_e, and collect these representations along with their labels.

    Args:
        model: The trained VQ-VAE model.
        dataloader: A DataLoader that returns (padded_sequences, labels, lengths).
        device: The device for inference.

    Returns:
        latent_all (Tensor): Tensor of shape (N, hidden_size) containing z_e representations for all samples.
        labels_all (Tensor): Tensor of shape (N,) containing the corresponding labels for all samples.
    """
    latent_list = []
    label_list = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x, labels, lengths = batch  # x: (batch, max_seq_len, input_size)
            x = x.to(device)
            lengths = lengths.to(device)
            # Get the encoder output z_e (without quantization) to observe the continuous latent representation
            z_e = model.encoder(x, lengths)  # (batch, hidden_size)
            latent_list.append(z_e.cpu())
            label_list.append(labels)

    latent_all = torch.cat(latent_list, dim=0)  # (N, hidden_size)
    labels_all = torch.cat(label_list, dim=0)      # (N,)
    return latent_all, labels_all


def visualize_ze_distribution(latent_all, labels_all):
    """
    Reduce the encoder output z_e to 2 dimensions using PCA and plot a scatter plot.
    Each point is annotated with its corresponding label, allowing you to observe the distribution of different labels in the latent space.

    Args:
        latent_all (Tensor): Tensor of shape (N, hidden_size) containing latent representations.
        labels_all (Tensor): Tensor of shape (N,) containing the corresponding labels.
    """
    # Convert the tensor to a numpy array
    latent_np = latent_all.numpy()

    # Use PCA to reduce the latent representations to 2D
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_np)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1],
                          c=labels_all.numpy(), cmap='tab20', alpha=0.7, s=50)

    # Annotate each point with its corresponding label
    for i, label in enumerate(labels_all.numpy()):
        plt.annotate(
            str(label),
            (latent_2d[i, 0], latent_2d[i, 1]),
            textcoords="offset points",
            xytext=(0, 5),  # Offset text 5 points above the marker
            ha='center',
            fontsize=8
        )

    plt.colorbar(scatter, ticks=range(NUM_CLASSES))
    plt.title("Distribution of Encoder Output z_e")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Specify the checkpoint path (modify as needed)
    checkpoint_path = "checkpoints/fourclass_v1/best_checkpoint.pth"

    # Load the trained model
    model = load_model_from_checkpoint(checkpoint_path, device)

    # Load your dataset (ensure the dataset includes the manually added third label data)
    transform = NormalizeTransform(force_divisor=400.0)
    dataset = TimeSeriesCSVFolderDataset(TRAIN_DATA_FOLDER, has_header=HAS_HEADER, delimiter=DELIMITER, transform=transform)

    # Create a DataLoader using a custom collate function
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

    # Extract the encoder output z_e and the corresponding labels for all samples
    latent_all, labels_all = get_encoder_representations(model, dataloader, device)

    # Visualize the distribution of z_e to check if the third label data is clearly separated from the first two labels
    visualize_ze_distribution(latent_all, labels_all)
