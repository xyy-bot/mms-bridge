import numpy as np
import torch
from torch.utils.data import Dataset


class ArtificialTimeSeriesDataset(Dataset):
    """
    生成具有 21 个类别的人工时序信号数据集。

    每个样本为二维时序数据，第一列为时间，第二列为信号值。
    不同类别的数据生成公式如下：
      - 类别 0~6：正弦波，参数（幅值、频率、相位）随 label 增加；
      - 类别 7~13：余弦波，参数随 label 增加；
      - 类别 14~20：正弦波和余弦波的组合，构成更复杂的波形。

    参数:
      num_sequences_per_label: 每个类别生成的样本数量（默认 100）
      sequence_length: 每个样本的时间步数（默认 100）
      num_labels: 类别数（默认 21）
      noise_std: 噪声标准差（默认 0.1，可根据需要调整）
    """

    def __init__(self, num_sequences_per_label=100, sequence_length=100, num_labels=21, noise_std=0.1):
        self.num_sequences_per_label = num_sequences_per_label
        self.sequence_length = sequence_length
        self.num_labels = num_labels
        self.noise_std = noise_std
        self.data, self.labels = self.generate_data()

    def generate_data(self):
        data_list = []
        label_list = []
        # 定义时间向量，假设从 0 到 10 秒均匀采样
        t = np.linspace(0, 5, self.sequence_length)

        for label in range(self.num_labels):
            for _ in range(self.num_sequences_per_label):
                # 根据 label 的不同区间采用不同的信号生成方式
                if label < 11:
                    # 正弦波：参数随着 label 增加
                    amplitude = 1.0 + label * 0.5  # 幅值
                    frequency = 0.5 + label * 0.5  # 频率
                    phase = label * 0.2  # 相位
                    signal = amplitude * np.sin(2 * np.pi * frequency * t + phase)
                elif label < 14:
                    # 余弦波
                    amplitude = 1.0 + (label - 7) * 0.5
                    frequency = 0.5 + (label - 7) * 0.5
                    phase = (label - 7) * 0.3
                    signal = amplitude * np.sin(2 * np.pi * frequency * t + phase)
                else:
                    # 正弦波和余弦波的组合
                    amplitude = 1.0 + (label - 14) * 0.5
                    frequency = 0.5 + (label - 14) * 0.2
                    phase = (label - 14) * 0.4
                    signal = amplitude * np.sin(2 * np.pi * frequency * t + phase) \
                             + 0.5 * amplitude * np.cos(2 * np.pi * (frequency / 2) * t + phase / 2)

                # 添加高斯噪声
                noise = np.random.normal(scale=self.noise_std, size=t.shape)
                signal = signal + noise

                # 构造 sample，第一列为时间，第二列为信号
                sample = np.stack([t, signal], axis=-1)  # shape: (sequence_length, 2)
                data_list.append(sample)
                label_list.append(label)

        data_array = np.array(data_list, dtype=np.float32)
        label_array = np.array(label_list, dtype=np.int64)
        return data_array, label_array

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.data[idx]  # shape: (sequence_length, 2)
        label = self.labels[idx]
        # 转换为 torch.Tensor
        sample_tensor = torch.tensor(sample, dtype=torch.float32)
        return sample_tensor, label


# 示例：如何使用该数据集
if __name__ == '__main__':
    # 创建数据集，每个 label 生成 50 个样本，序列长度为 100
    dataset = ArtificialTimeSeriesDataset(num_sequences_per_label=50, sequence_length=100, num_labels=21, noise_std=0.1)
    print("数据集总样本数：", len(dataset))
    # 随机取一个样本查看
    sample, label = dataset[0]
    print("样本 shape:", sample.shape)  # 应为 (100, 2)
    print("样本 label:", label)
