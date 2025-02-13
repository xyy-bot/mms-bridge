# config.py
import os

# Dataset-related configurations (CSV file storage path, be sure to modify it to your actual path)
DATA_FOLDER = "./dataset/data_for_force"
TRAIN_DATA_FOLDER = "./dataset/train_data_four_class"
HAS_HEADER = True        # Whether the CSV file contains a header
DELIMITER = ','          # CSV file delimiter

# Model-related hyperparameters
INPUT_SIZE = 2           # The CSV file contains two columns: time and force
OUTPUT_SIZE = 2          # The output dimension during reconstruction should match the input dimension
HIDDEN_SIZE = 64         # GRU hidden layer dimension (also the latent dimension)
NUM_LAYERS = 2           # Number of GRU layers
NUM_EMBEDDINGS = 256      # Codebook size for VQ-VAE
EMBEDDING_DIM = HIDDEN_SIZE  # Codebook dimension for VQ-VAE, should match the latent dimension
COMMITMENT_COST = 0.25

# Training-related configurations
NUM_CLASSES = 4          # Number of classes (adjust according to your actual data)
BATCH_SIZE = 128
NUM_EPOCHS = 1200
LEARNING_RATE = 1e-3

# Checkpoint saving configurations
CHECKPOINT_DIR = "checkpoints/fourclass_v3_further"
# CHECKPOINT_FREQ = 5      # Save a checkpoint every specified number of epochs
