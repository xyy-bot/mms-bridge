# utils/dataset.py
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

def extract_label(filename):
    """
    根据文件名提取 label
    假设文件名格式为: <label>_<index>.csv
    如 "fake_green_paprika_0.csv" 提取 label 为 "fake_green_paprika"
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    else:
        return base

class TimeSeriesCSVFolderDataset(Dataset):
    def __init__(self, folder_path, has_header=True, delimiter=',', transform=None):
        """
        Args:
            folder_path (str): CSV 文件所在的文件夹路径
            has_header (bool): CSV 是否包含表头
            delimiter (str): CSV 分隔符
            transform (callable, optional): 可选的数据变换
        """
        self.folder_path = folder_path
        self.has_header = has_header
        self.delimiter = delimiter
        self.transform = transform

        # 搜索所有 CSV 文件
        self.csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        self.csv_files = [f for f in self.csv_files if f.lower().endswith('.csv')]

        # 提取每个文件的 label 字符串
        self.labels_str = [extract_label(f) for f in self.csv_files]

        # 构建 label 到整数的映射（字典按字母序排序）
        self.label_set = sorted(list(set(self.labels_str)))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.label_set)}
        print("Found labels:", self.label_to_idx)

    def __len__(self):
        return len(self.csv_files)

    def __getitem__(self, idx):
        csv_file = self.csv_files[idx]

        # 如果有表头则跳过第一行
        skiprows = 1 if self.has_header else 0

        try:
            # 加载 CSV 数据，期望 CSV 有两列数据：time 和 force
            data = np.loadtxt(csv_file, delimiter=self.delimiter, skiprows=skiprows)
        except Exception as e:
            raise RuntimeError(f"Error loading {csv_file}: {e}")

        # 获取对应的 label
        label_str = extract_label(csv_file)
        label = self.label_to_idx[label_str]

        # 转为 float32 的 tensor
        data_tensor = torch.tensor(data, dtype=torch.float32)

        if self.transform:
            data_tensor = self.transform(data_tensor)

        return data_tensor, label
