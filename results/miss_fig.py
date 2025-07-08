import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def draw_miss_rate_plot(dataset_name, max_chunks=3000, y_limit=0.25):
    """
    Draw cache miss rate analysis plot for FFANNS.
    
    Args:
        dataset_name (str): Name of the dataset
        max_chunks (int): Maximum number of chunks to plot (default: 3000)
        y_limit (float): Upper limit for y-axis (default: 0.25)
    """
    # 构建输入CSV文件路径
    csv_filename = f"search_miss.csv"
    csv_path = os.path.join(dataset_name, csv_filename)
    
    # 读取数据
    df = pd.read_csv(csv_path)
    df = df[df['chunk_id'] <= max_chunks].copy()

    # 数据分组和重采样
    df['group'] = df['chunk_id'] // 10
    grouped_df = df.groupby('group').agg({
        'chunk_id': 'mean',
        'miss_rate': 'mean'
    }).reset_index()

    # 创建图形
    plt.figure(figsize=(12, 7))
    
    # 绘制主折线图
    plt.plot(grouped_df['chunk_id'], grouped_df['miss_rate'], 
             marker='.',
             color='#2E86C1',
             linewidth=2,
             markersize=6,
             linestyle='-')

    # 设置标题和标签
    title = f'GPU Cache Miss Rate Analysis\n{dataset_name}'
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('Chunk ID', fontsize=12)
    plt.ylabel('Miss Rate', fontsize=12)

    # 添加平均值线
    mean_rate = df['miss_rate'].mean()
    plt.axhline(y=mean_rate, 
                color='red', 
                linestyle='--', 
                alpha=0.8,
                label=f'Mean Miss Rate: {mean_rate:.3f}')

    # 设置网格和图例
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # 设置坐标轴范围和刻度
    step = 100
    xticks = np.arange(0, max_chunks + 1, step)
    plt.xticks(xticks)
    plt.xlim(0, max_chunks)
    plt.ylim(0, y_limit)
    plt.tight_layout()

    # 构建输出图片路径
    output_filename = f"{dataset_name}_miss_rate.png"
    output_path = os.path.join(dataset_name, output_filename)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形，释放内存

    print(f"Miss rate plot saved to: {output_path}")


# 使用示例
if __name__ == "__main__":
    # 单个模式的绘图
    # draw_miss_rate_plot("Sift1M", 3000, 0.14)
    draw_miss_rate_plot("MSTuring-30M", 3000, 0.0025)
