import yaml
import random

def generate_large_runbook():
    dataset_name = "sift-1B"
    total_vectors = 1000000000  # 10亿向量
    
    config = {
        dataset_name: {
            "max_pts": total_vectors,
            "res_path": "/data2/pyc/workspace/ffanns/results/sift-1B/search_results/search_64.bin"
        }
    }
    
    # 初始化步骤计数
    step = 1
    
    # 第一阶段：插入0-800M的数据，每批10M
    batch_size = 10000000  # 10M
    first_phase_batches = 80  # 80个批次，共800M
    
    current_pos = 0
    
    # 添加第一阶段的插入操作，每两次插入执行一次搜索
    for i in range(first_phase_batches):
        # 添加插入操作
        config[dataset_name][step] = {
            "operation": "insert",
            "start": current_pos,
            "end": current_pos + batch_size
        }
        current_pos += batch_size
        step += 1
        
        # 每两次插入后添加一次搜索操作
        if (i + 1) % 2 == 0:
            config[dataset_name][step] = {
                "operation": "search"
            }
            step += 1
    
    # 第二阶段：每次先删除0-800M中的随机10M数据，再插入800M-1000M中的10M数据
    second_phase_batches = 20  # 20个批次
    
    # 创建一个列表，跟踪0-800M范围内所有可以删除的块
    # 将0-800M数据分成80个10M大小的块
    available_blocks = list(range(0, 800000000, batch_size))
    
    for i in range(second_phase_batches):
        # 从可用块列表中随机选择一个进行删除
        if available_blocks:
            random_index = random.randint(0, len(available_blocks) - 1)
            delete_start = available_blocks.pop(random_index)
            
            # 添加删除操作
            config[dataset_name][step] = {
                "operation": "delete",
                "start": delete_start,
                "end": delete_start + batch_size
            }
            step += 1
        else:
            print("警告：没有更多可删除的块!")
            # 如果没有可用块，可以跳过删除或使用其他策略
        
        # 从800M-1000M区间插入10M数据
        insert_start = 800000000 + i * batch_size
        
        # 添加插入操作
        config[dataset_name][step] = {
            "operation": "insert",
            "start": insert_start,
            "end": insert_start + batch_size
        }
        step += 1
        
        # # 每两轮操作后添加一次搜索操作
        # if (i + 1) % 2 == 0:
        config[dataset_name][step] = {
            "operation": "search"
        }
        step += 1
    
    # 将配置写入YAML文件
    with open('runbooks/sift1B_large.yaml', 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    print(f"已生成runbook文件：runbooks/sift1B_large.yaml")
    print(f"总步骤数：{step-1}")
    print(f"第一阶段：{first_phase_batches}批插入操作，每2批执行1次搜索")
    print(f"第二阶段：{second_phase_batches}批删除+插入操作，每1轮执行1次搜索")
    print(f"使用了 {min(second_phase_batches, len(available_blocks))} 个唯一的删除块")

if __name__ == "__main__":
    generate_large_runbook()