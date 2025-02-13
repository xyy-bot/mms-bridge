# utils/transform.py
import torch

class NormalizeTransform:
    def __init__(self, force_divisor=800.0):
        """
        参数:
          force_divisor: 用于缩放 force 列的常数，
                         假设 force 最大值约为 800，则可以设置为 800.0，
                         这样缩放后数据大致落在 [0,1] 范围内。
        """
        self.force_divisor = force_divisor

    def __call__(self, sample):
        """
        sample: Tensor，形状为 (seq_len, 2)
                第 0 列为 time， 第 1 列为 force
        返回:
          标准化后的 sample
        """
        # 对 force 列进行缩放
        sample[:, 1] = sample[:, 1] / self.force_divisor
        return sample
