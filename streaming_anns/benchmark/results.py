from __future__ import absolute_import

import json
import os
import re
import traceback
import numpy as np

from benchmark.streaming.load_runbook import get_res_path

def get_baseline_res_path(dataset, algorithm=None, runbook_path=None):
    """
    获取简化的结果路径，用于存储.bin格式的搜索结果
    
    参数:
    - dataset: 数据集名称
    - algorithm: 算法名称
    - runbook_path: runbook路径
    """
    # 基础目录
    base_dir = 'results'
    
    # 构建路径
    # 首先添加数据集名称
    if dataset:
        base_dir = os.path.join(base_dir, dataset)
    
    # 然后添加runbook名称（保留.yaml扩展名）
    if runbook_path:
        runbook_name = os.path.basename(runbook_path)  # 保留.yaml扩展名
        base_dir = os.path.join(base_dir, runbook_name)
    
    # 添加算法名称作为子文件夹
    if algorithm:
        base_dir = os.path.join(base_dir, algorithm)
    
    # 确保目录存在
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # 最后的文件名
    filename = 'search_res.bin'
    
    # 返回完整路径
    return os.path.join(base_dir, filename)


def store_results(dataset, count, definition, query_arguments,
                    attrs, results, search_type, runbook_path=None):
    """
    将搜索结果存储为简单的.bin格式
    
    参数:
    - dataset: 数据集名称
    - all_results: 所有搜索步骤的结果列表，每个元素是一个步骤的查询结果
    - topk: 每次查询返回的最近邻数量
    - runbook_path: runbook路径，用于结果文件命名
    """
    res_path = get_baseline_res_path(dataset, definition.algorithm, runbook_path)
    print(res_path)
    
    try:
        with open(res_path, 'wb') as f:
            # 写入文件头
            num_steps = len(results)

            if num_steps == 0:
                print("没有搜索结果可存储")
                return
                
            # 假设所有步骤的查询数量相同
            num_queries = results[0].shape[0]
            # 写入文件头信息
            np.array([num_steps], dtype=np.uint32).tofile(f)
            np.array([num_queries], dtype=np.uint32).tofile(f)
            np.array([count], dtype=np.uint32).tofile(f)
            
            # 写入每个步骤的结果
            for step_results in results:
                # 确保结果是uint32类型，如果不是则进行转换
                if step_results.dtype != np.uint32:
                    step_results = step_results.astype(np.uint32)
                # 直接写入扁平化的数组
                step_results.flatten().tofile(f)
            
            print(f"已将 {num_steps} 个步骤的结果保存到 {res_path}")
            print(f"每个步骤的形状: [{num_queries}, {count}]")
            
    except Exception as e:
        print(f"保存结果到 {res_path} 时出错: {e}")
        raise

def load_all_results(dataset=None, runbook_path=None, baseline=False):
    res_path = None
    if baseline:
        res_path = get_baseline_res_path(dataset, "diskann", runbook_path)
    else:
        res_path = get_res_path(dataset, runbook_path)
    
    results = []
    try:
        with open(res_path, 'rb') as f:
            # 读取文件头
            num_steps = np.fromfile(f, dtype=np.uint32, count=1)[0]
            num_queries = np.fromfile(f, dtype=np.uint32, count=1)[0]
            topk = np.fromfile(f, dtype=np.uint32, count=1)[0]
            print(num_steps, num_queries, topk)
            # 读取每个step的结果
            for step in range(num_steps):
                # 读取这个step的所有neighbors
                neighbors = np.fromfile(f, dtype=np.uint32, count=num_queries * topk)
                # 重塑为 [num_queries, topk] 的形状
                neighbors = neighbors.reshape(num_queries, topk)
                results.append(neighbors)   
                
            print(f"Loaded {num_steps} steps of results from {res_path}")
            print(f"Shape of each step: [{num_queries}, {topk}]")
            
    except Exception as e:
        print(f"Error loading results from {res_path}: {e}")
        raise

    return results
    