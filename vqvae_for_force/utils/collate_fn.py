# utils/collate_fn.py
from torch.nn.utils.rnn import pad_sequence
import torch


def custom_collate_fn(batch):
    """
    参数 batch 是一个列表，每个元素为 (data_tensor, label)，
    其中 data_tensor 的形状为 (seq_len, 2)

    返回：
      - padded_sequences: (batch_size, max_seq_len, 2)
      - labels: (batch_size,)
      - lengths: (batch_size,) 每个样本原始的序列长度
    """
    # 分别提取数据和标签
    sequences, labels = zip(*batch)

    # 计算每个序列的长度
    lengths = torch.tensor([s.size(0) for s in sequences], dtype=torch.long)

    # 使用 pad_sequence 将所有序列 padding 到同一长度（默认填充 0）
    padded_sequences = pad_sequence(sequences, batch_first=True)

    # 转换标签为 tensor
    labels = torch.tensor(labels, dtype=torch.long)

    return padded_sequences, labels, lengths
