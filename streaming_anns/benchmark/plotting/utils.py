from __future__ import absolute_import

import itertools
import numpy
import os
import traceback
import sys
import pandas as pd

from benchmark.dataset_io import knn_result_read
import benchmark.streaming.compute_gt
from benchmark.streaming.load_runbook import load_runbook
from benchmark.plotting.metrics import knn

def compute_metrics_all_runs(dataset, dataset_name, res, recompute=False, 
        sensor_metrics=False, search_times=False,
        runbook_path=None):
    try:
        true_nn_across_steps = []
        gt_dir = benchmark.streaming.compute_gt.gt_dir(dataset, runbook_path)
        max_pts, runbook = load_runbook(dataset_name, dataset.nb, runbook_path)
        for step, entry in enumerate(runbook):
            if entry['operation'] == 'search':
                step_gt_path = os.path.join(gt_dir, 'step' + str(step+1) + '.gt100')
                true_nn = knn_result_read(step_gt_path)
                true_nn_across_steps.append(true_nn)
                
                # sorted_nn = numpy.sort(true_nn[0], axis=1)
                # step_csv_path = f'/data2/pyc/workspace/ffanns/results/MSTuring-30M/search_results/step{step+1}_gt.csv'
                # pd.DataFrame(sorted_nn).to_csv(step_csv_path, index=False, header=False)
    except:
        print(f"Groundtruth for {dataset} not found.")
        #traceback.print_exc()
        return

    search_type = dataset.search_type()
    run_nn_across_steps = res
    assert search_type in ("knn"), "Now only support knn search"
    
    # start to recall metrics 
    v = []
    assert len(true_nn_across_steps) == len(run_nn_across_steps)
    for (true_nn, run_nn) in zip(true_nn_across_steps, run_nn_across_steps):
        val_dict = knn(true_nn, run_nn, True)
        # print(val_dict['mean'])
        v.append(val_dict['mean'])
    return v
    
