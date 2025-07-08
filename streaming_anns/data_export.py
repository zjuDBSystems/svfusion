import pandas as pd
import sys
import matplotlib.pyplot as plt
import os
import numpy as np
from benchmark.datasets import DATASETS
from benchmark.plotting.utils  import compute_metrics_all_runs
from benchmark.results import load_all_results


def plot_recall_curve(recalls, dataset_name, output_dir="plots"):
    # 创建图形
    plt.figure(figsize=(12, 6))
    
    # 生成步骤编号（从1开始）
    steps = np.arange(1, len(recalls) + 1)
    
    # 绘制折线图
    plt.plot(steps, recalls,
             marker='^',           # 使用三角形标记
             color='#2E86C1',
             linewidth=2,
             markersize=6,        # 增大标记尺寸
             linestyle='-',
             label='Recall Rate')
    
    # # 在每个点上添加数值标签
    # for x, y in zip(steps, recalls):
    #     plt.text(x, y + 0.01, f'{y:.4f}',
    #             ha='center', va='bottom',
    #             fontsize=10)
    # 在每个点上添加数值标签（每隔10个点）
    for x, y in zip(steps, recalls):
        if x % 10 == 1 or x == len(recalls):  # 每隔10个点（从第1个开始）和最后一个点
            plt.text(x, y + 0.01, f'{y:.4f}',
                    ha='center', va='bottom',
                    fontsize=12)
    
    # 设置标题和标签
    title = f'Recall Rate Analysis\n{dataset_name}'
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel('Recall Rate', fontsize=12)
    
    # 添加平均值线
    # mean_recall = np.mean(recalls)
    # plt.axhline(y=mean_recall, 
    #             color='red', 
    #             linestyle='--', 
    #             alpha=0.8,
    #             label=f'Mean Recall: {mean_recall:.4f}')
    
    # 设置网格和图例
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 设置y轴范围，从0开始
    plt.ylim(0.7, 1.0)  # 根据数据范围调整
    
    # 设置x轴刻度
    # 设置x轴刻度（每隔10个step显示一次）
    if len(steps) > 20:  # 如果步骤数较多，则每隔10个显示一次
        xticks = np.arange(1, len(recalls) + 1, 10)
        # 确保最后一个步骤也显示
        if len(recalls) % 10 != 0:
            xticks = np.append(xticks, len(recalls))
        plt.xticks(xticks)
    else:
        plt.xticks(steps)  # 如果步骤较少，则全部显示
    
    plt.tight_layout()
    
    # 构建输出图片路径
    output_filename = f"{dataset_name}_recall_analysis.png"
    output_path = os.path.join(output_dir, output_filename)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存图片
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    datasets = DATASETS.keys()
    # runbook_path = "runbooks/test1M.yaml"
    runbook_path = "runbooks/delete1M.yaml"
    # runbook_path = "runbooks/test.yaml"
    dataset_name = "msturing-30M"
    dataset = DATASETS[dataset_name]()
    
    # result = load_all_results(dataset_name, runbook_path)
    result = load_all_results(dataset_name, runbook_path, True)
    recalls = compute_metrics_all_runs(dataset, dataset_name, result, runbook_path=runbook_path)
    print(recalls)

    # plot_recall_curve(recalls, dataset_name)