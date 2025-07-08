import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def draw_performance_plot(dataset_name, mode, chunk_size, max_chunks=2000):
    """
    Draw performance analysis plot for FFANNS.
    
    Args:
        dataset_name (str): Name of the dataset
        mode (str): Running mode (e.g., 'device', 'hd')
        chunk_size (int): Size of chunks used
        max_chunks (int): Maximum number of chunks to plot (default: 2000)
    """
    # 构建输入CSV文件路径
    csv_filename = f"{mode}_insert_{chunk_size}.csv"
    csv_path = os.path.join(dataset_name, csv_filename)
    
    # 读取数据
    df = pd.read_csv(csv_path)
    df = df[df['chunk_id'] <= max_chunks].copy()

    # 数据分组和重采样
    df['group'] = df['chunk_id'] // 10
    grouped_df = df.groupby('group').agg({
        'chunk_id': 'mean',
        'total_time_ms': 'mean'
    }).reset_index()

    # 创建图形
    plt.figure(figsize=(12, 6))
    
    # 绘制主折线图
    plt.plot(grouped_df['chunk_id'], grouped_df['total_time_ms'], 
             marker='.',
             color='#2E86C1',
             linewidth=2,
             markersize=6,
             linestyle='-')

    # 设置标题和标签
    title = f'FFANNS Performance Analysis\n{dataset_name} ({mode}, chunk_size={chunk_size})'
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('Chunk ID', fontsize=12)
    plt.ylabel('Total Time (ms)', fontsize=12)

    # 添加平均值线
    mean_time = df['total_time_ms'].mean()
    plt.axhline(y=mean_time, 
                color='red', 
                linestyle='--', 
                alpha=0.8,
                label=f'Mean Time: {mean_time:.2f} ms')

    # 设置网格和图例
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # 设置x轴刻度
    step = 100
    xticks = np.arange(0, max_chunks + 1, step)
    plt.xticks(xticks)
    plt.xlim(0, max_chunks)
    plt.tight_layout()

    # 构建输出图片路径
    output_filename = f"{dataset_name}_{mode}_insert_{chunk_size}.png"
    output_path = os.path.join(dataset_name, output_filename)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形，释放内存

    print(f"Plot saved to: {output_path}")


# 使用示例
if __name__ == "__main__":
    # 示例调用
    # draw_performance_plot("Sift1M", "device", 16)
    # draw_performance_plot("Sift1M", "device", 32)
    # draw_performance_plot("Sift1M", "hd", 16)
    # draw_performance_plot("Sift1M", "hd", 32)
    # draw_performance_plot("MSTuring-30M", "device", 16)
    draw_performance_plot("MSTuring-30M", "hd", 16)
    
    # modes = ["device", "hd"]
    # chunk_sizes = [32, 64, 128]
    
    # for mode in modes:
    #     for chunk_size in chunk_sizes:
    #         draw_performance_plot("msturing_10m", mode, chunk_size)