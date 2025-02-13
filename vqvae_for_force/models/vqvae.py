# models/vqvae.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(EncoderGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x, lengths):
        """
        Args:
            x: (batch, max_seq_len, input_size)
            lengths: (batch,) with the true lengths of each sequence.
        Returns:
            encoded: (batch, hidden_size) – the last hidden state from the GRU.
        """
        # Pack the padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, hidden = self.gru(packed)
        # hidden shape: (num_layers, batch, hidden_size)
        encoded = hidden[-1]  # take the last layer's hidden state.
        return encoded

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # Initialize the embeddings (the codebook)
        self.embedding = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, inputs):
        """
        Args:
            inputs: (batch, embedding_dim) – continuous latent vectors.
        Returns:
            quantized: (batch, embedding_dim) – quantized latent vectors.
            vq_loss: scalar – the VQ loss.
        """
        # Flatten input (if necessary, here inputs is already (batch, embedding_dim))
        flat_input = inputs

        # Compute L2 distances between inputs and embedding vectors.
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.embedding**2, dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.t())
        )
        # Find the nearest embedding indices.
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize the inputs.
        quantized = torch.matmul(encodings, self.embedding)
        quantized = quantized.view_as(inputs)

        # Compute the VQ losses.
        codebook_loss = F.mse_loss(quantized.detach(), inputs)
        commitment_loss = self.commitment_cost * F.mse_loss(inputs, quantized.detach())
        vq_loss = codebook_loss + commitment_loss

        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        return quantized, vq_loss

class DecoderGRU(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers):
        super(DecoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.gru = nn.GRU(output_size, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, z, seq_len):
        """
        Args:
            z: (batch, hidden_size) – the quantized latent representation.
            seq_len: The length of the sequence to be reconstructed.
        Returns:
            reconstructed: (batch, seq_len, output_size)
        """
        hidden = z.unsqueeze(0).repeat(self.num_layers, 1, 1)
        batch_size = z.size(0)
        # Create a dummy input sequence (zeros)
        input_seq = torch.zeros(batch_size, seq_len, self.output_size, device=z.device)
        outputs, _ = self.gru(input_seq, hidden)
        reconstructed = self.output_layer(outputs)
        return reconstructed

class VQVAEClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_embeddings,
                 embedding_dim, commitment_cost, output_size, num_classes):
        super(VQVAEClassifier, self).__init__()
        self.encoder = EncoderGRU(input_size, hidden_size, num_layers)
        self.vector_quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = DecoderGRU(hidden_size, output_size, num_layers)
        # Optionally, you can include a classifier head for additional supervision.
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        """
        Args:
            x: (batch, max_seq_len, input_size)
            lengths: (batch,) actual lengths of each sequence.
        Returns:
            reconstructed: (batch, max_seq_len, output_size)
            class_logits: (batch, num_classes)
            vq_loss: scalar – VQ loss.
        """
        z_e = self.encoder(x, lengths)  # (batch, hidden_size)
        z_q, vq_loss = self.vector_quantizer(z_e)  # (batch, hidden_size)
        seq_len = x.size(1)
        reconstructed = self.decoder(z_q, seq_len)
        class_logits = self.classifier(z_q)
        return reconstructed, class_logits, vq_loss
