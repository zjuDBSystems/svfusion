import yaml

def generate_test_yaml(total_vectors, first_insert, insert_batch_size, insert_rounds):
    dataset_name = "msturing-30M"
    config = {
        dataset_name: {
            "max_pts": total_vectors,
            "res_path": "/data2/pyc/workspace/ffanns/results/msturing-30M/search_results/search_64.bin"
        }
    }
    
    # 添加步骤配置
    step = 1
    
    # 添加第一次插入
    config[dataset_name][step] = {
        "operation": "insert",
        "start": 0,
        "end": first_insert
    }
    step += 1
    
    config[dataset_name][step] = {
        "operation": "search"
    }
    step += 1
    
    # 添加搜索和额外插入
    current_pos = first_insert
    
    consolidate_iter = 5
    for i in range(insert_rounds):
        # add delete
        config[dataset_name][step] = {
            "operation": "delete",
            "start": 8000 + 16000 * i,
            "end": 16000 + 16000 * i
            # "start": 1000 + 4000 * i * 2,
            # "end": 1000 + 4000 * i *2 + 4000
        }
        # current_pos += insert_batch_size
        step += 1
        
        
        # 添加插入操作
        config[dataset_name][step] = {
            "operation": "insert",
            "start": current_pos,
            "end": current_pos + insert_batch_size
        }
        current_pos += insert_batch_size
        step += 1
        
        # 添加搜索操作
        config[dataset_name][step] = {
            "operation": "search"
        }
        step += 1
        
        # if (i+1) % consolidate_iter == 0:
        #     config[dataset_name][step] = {
        #         "operation": "search"
        #     }
        #     step += 1
    
    # 写入YAML文件
    # with open('runbooks/test1M.yaml', 'w') as f:
    #     yaml.dump(config, f, sort_keys=False)
    
    with open('runbooks/delete1M.yaml', 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    # with open('runbooks/test.yaml', 'w') as f:
    #     yaml.dump(config, f, sort_keys=False)

# 使用示例
total_vectors = 30000000    # 总向量数量
first_insert = 1000000     # 第一次插入的数量
insert_batch_size = 8000   # 后续每次插入的数量
insert_rounds = 40    # 插入轮次

generate_test_yaml(total_vectors, first_insert, insert_batch_size, insert_rounds)
