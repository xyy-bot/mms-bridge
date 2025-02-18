import os
import pandas as pd
import matplotlib.pyplot as plt


def extract_force_segment(file_path, output_folder):
    # 读取 CSV 文件
    df = pd.read_csv(file_path)

    # 确保数据中包含力的列，这里假设列名包含 'Force'
    force_col = [col for col in df.columns if 'Force' in col or 'force' in col]
    if not force_col:
        print(f"No force column found in {file_path}")
        return None, None
    force_col = force_col[0]  # 选择第一个匹配的列

    # 找到力开始变化的位置
    start_idx = df[force_col].gt(2).idxmax() -2

    # 找到力结束变化的位置（从后往前找到第一个大于2的位置）
    # end_idx = df[force_col].gt(2)[::-1].idxmax()
    end_idx = start_idx + 1200
    # 提取对应数据段
    extracted_df = df.iloc[start_idx:end_idx].copy()

    # **删除原始时间列（如果存在）**
    time_columns = [col for col in extracted_df.columns if 'Time' in col or 'time' in col]
    extracted_df.drop(columns=time_columns, errors='ignore', inplace=True)

    # **重新定义时间列**
    extracted_df['Time'] = range(len(extracted_df))  # 以 ms 为单位
    extracted_df['Time'] = extracted_df['Time'] / 1000  # **转换为秒 (s)**

    # **调整列顺序，使 Time 在前**
    extracted_df = extracted_df[['Time', force_col]]

    # 保存处理后的 CSV 文件
    os.makedirs(output_folder, exist_ok=True)
    output_csv_path = os.path.join(output_folder, os.path.basename(file_path))
    extracted_df.to_csv(output_csv_path, index=False)
    print(f"Processed CSV saved: {output_csv_path}")

    return extracted_df, force_col


def process_and_plot(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    object_groups = {}

    for file in os.listdir(input_folder):
        if file.endswith(".csv"):
            file_path = os.path.join(input_folder, file)
            extracted_df, force_col = extract_force_segment(file_path, output_folder)
            if extracted_df is not None:
                object_name = "_".join(file.split("_")[:-1])  # 提取物体名称（去掉实验编号）
                if object_name not in object_groups:
                    object_groups[object_name] = []
                object_groups[object_name].append((file, extracted_df))

    for object_name, experiments in object_groups.items():
        plt.figure(figsize=(5, 4))
        for file, df in experiments:
            plt.plot(df['Time'], df[force_col], label=file)

        plt.xlabel("Time (s)")  # **单位改为秒**
        plt.ylabel("Force")
        # plt.title(f"Force Comparison for {object_name}")
        plt.grid(True)
        plt.legend()

        output_plot_path = os.path.join(output_folder, f"{object_name}.png")
        plt.savefig(output_plot_path)
        plt.close()
        print(f"Saved plot: {output_plot_path}")


# 设置输入和输出文件夹
input_folder = "./pipeline_test_garlic_v3"  # 这里替换成你的 CSV 文件所在目录
output_folder = "./processed"

process_and_plot(input_folder, output_folder)
