# train.py
import os
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from config_for_half_training import DATA_FOLDER, TRAIN_DATA_FOLDER, HAS_HEADER, DELIMITER, INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, \
    NUM_EMBEDDINGS, EMBEDDING_DIM, COMMITMENT_COST, NUM_CLASSES, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, CHECKPOINT_DIR
from utils.dataset import TimeSeriesCSVFolderDataset
from utils.collate_fn import custom_collate_fn
from utils.transform import NormalizeTransform
from models.vqvae import VQVAEClassifier


def train_epoch(model, dataloader, optimizer, device, recon_loss_weight=1.0, class_loss_weight=1.0):
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    for batch in dataloader:
        x, labels, lengths = batch
        x = x.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        reconstructed, logits, vq_loss = model(x, lengths)
        recon_loss = F.mse_loss(reconstructed, x)
        class_loss = F.cross_entropy(logits, labels)
        loss = recon_loss_weight * recon_loss + class_loss_weight * class_loss + vq_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total_samples

    return avg_loss, accuracy


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            x, labels, lengths = batch
            x = x.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            reconstructed, logits, vq_loss = model(x, lengths)
            recon_loss = F.mse_loss(reconstructed, x)
            class_loss = F.cross_entropy(logits, labels)
            loss = recon_loss + class_loss + vq_loss
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total_samples
    return avg_loss, accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    torch.random.manual_seed(1)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Load the dataset.
    transform = NormalizeTransform(force_divisor=100.0)  # Adjust the divisor based on your data.
    dataset = TimeSeriesCSVFolderDataset(TRAIN_DATA_FOLDER, has_header=HAS_HEADER, delimiter=DELIMITER, transform=transform)

    Shuffle and split dataset (80% train, 20% test).
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    split = int(0.5 * dataset_size)
    train_indices, test_indices = indices[:split], indices[split:]
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

    model = VQVAEClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_EMBEDDINGS,
                            EMBEDDING_DIM, COMMITMENT_COST, OUTPUT_SIZE, NUM_CLASSES)
    # checkpoint = torch.load(f'{CHECKPOINT_DIR}/best_checkpoint.pth', map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_train_loss = float('inf')
    best_path = os.path.join(CHECKPOINT_DIR, "best_checkpoint.pth")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, device)
        test_loss, test_accuracy = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f}| Test Loss: {test_loss:.4f}| Test Acc: {test_accuracy * 100:.2f}%")

        # Save only the best checkpoint.
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, best_path)
            print(f"Best checkpoint at epoch {epoch}: {best_path}")


if __name__ == '__main__':
    main()
