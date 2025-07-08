from __future__ import absolute_import
import argparse
import logging
import logging.config
from benchmark.datasets import DATASETS
from benchmark.algorithms.definitions import (get_all_definitions, get_specific_algorithm_definitions)
import glob
import os
import psutil

import benchmark.common
from benchmark.runner import run_no_docker

def positive_int(s):
    i = None
    try:
        i = int(s)
    except ValueError:
        pass
    if not i or i < 1:
        raise argparse.ArgumentTypeError("%r is not a positive integer" % s)
    return i

def run_streaming_workload(args, algorithm, dataset_name, runbook_path, count):
    # 加载数据集
    dataset = DATASETS[dataset_name]()
    dimension = dataset.d
    distance = dataset.distance()
    
    definition_path = benchmark.common.track_path(algorithm)
    definition_files = glob.glob(os.path.join(definition_path, "*.yaml"), recursive=True)
    # 获取算法定义
    definitions = get_specific_algorithm_definitions(
        definition_files[0], algorithm,
        dimension, dataset_name, distance, count)
    
    # print(definitions)
    
    memory_margin = 500e6  # reserve some extra memory for misc stuff
    mem_limit = int((psutil.virtual_memory().available - memory_margin))
    # cpu_limit = "0-%d" % (os.cpu_count() - 1)
    cpu_limit = "0-32"
    print(memory_margin, mem_limit, cpu_limit)

    # 执行工作负载
    for definition in definitions:
        run_no_docker(definition, args.dataset, args.count,
                     args.runs, cpu_limit, mem_limit,
                     args.power_capture, args.runbook_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', required=True, help='算法名称')
    parser.add_argument('--dataset', required=True, help='数据集名称')
    parser.add_argument('--runbook_path', required=True, help='runbook路径')
    parser.add_argument(
            "-k", "--count",
            default=-1,
            type=int,
            help="the number of near neighbours to search for")
    parser.add_argument(
        '--runs',
        metavar='COUNT',
        type=positive_int,
        help='run each algorithm instance %(metavar)s times and use only'
             ' the best result',
        default=1)
    parser.add_argument(
        '--power-capture',
        help='Power capture parameters for the T3 competition. '
            'Format is "ip:port:capture_time_in seconds" (ie, 127.0.0.1:3000:10).',
        default="")
    # --rebuild false
    # --t3 false
    # --upload-index false
    # --download-index false
    # --blob_prefix -sas_string flase 
    # --private-query false
    args = parser.parse_args()

    dataset = DATASETS[args.dataset]()
    if args.count == -1:
        args.count = dataset.default_count()
    
    run_streaming_workload(args, args.algorithm, args.dataset, args.runbook_path, args.count)

if __name__ == "__main__":
    main()

# python run.py --algorithm diskann --dataset msturing-30M --runbook_path runbooks/test.yaml
