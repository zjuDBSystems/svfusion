import os
import numpy as np
from dataset_io import read_fbin

import sys
[sys.path.append(i) for i in ['.', '..']]

def ensure_dir(file_path):
    """确保文件所在目录存在，如果不存在则创建"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"创建目录: {directory}")
        except Exception as e:
            raise Exception(f"创建目录失败: {e}")
        
def extract_partial_fbin(input_file, output_file, num_vectors):
    """
    从fbin文件中提取指定数量的向量并保存为新的fbin文件
    
    Args:
        input_file (str): 输入fbin文件路径
        output_file (str): 输出fbin文件路径
        num_vectors (int): 需要提取的向量数量
    """
    ensure_dir(output_file)
    # 读取指定数量的向量
    vectors = read_fbin(input_file, start_idx=0,chunk_size=num_vectors)
    
    # 写入新文件
    with open(output_file, "wb") as f:
        # 写入文件头: nvecs和dim (两个int32)
        np.array([num_vectors, vectors.shape[1]], dtype=np.int32).tofile(f)
        # 写入向量数据
        vectors.astype(np.float32).tofile(f)
    
    print(f"已提取{num_vectors}个向量到{output_file}")
    print(f"向量维度: {vectors.shape[1]}")
    print(f"输出文件大小: {vectors.nbytes + 8:,} bytes")

if __name__ == "__main__":
    input_file = "/data2/pyc/data/msturing/base1b.fbin"
    output_file = "/data2/pyc/data/msturing30M/msturing30M.fbin"
    num_vectors = 30_000_000  # 30M
    extract_partial_fbin(input_file, output_file, num_vectors)
