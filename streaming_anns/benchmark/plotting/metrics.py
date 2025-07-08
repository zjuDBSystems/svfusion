from __future__ import absolute_import
import numpy as np
import itertools
import operator
import random
import sys
import copy

# from benchmark.plotting.eval_range_search import compute_AP
# from benchmark.sensors.power_capture import power_capture

def compute_recall_without_distance_ties(true_ids, run_ids, count):
    return len(set(true_ids) & set(run_ids))

def compute_recall_with_distance_ties(true_ids, true_dists, run_ids, count):
    # This function assumes "true_dists" is monotonic either increasing or decreasing

    found_tie = False
    gt_size = np.shape(true_dists)[0]

    if gt_size==count:
        # nothing fancy to do in this case
        recall =  len(set(true_ids[:count]) & set(run_ids))

    else:
        dist_tie_check = true_dists[count-1] # tie check anchored at count-1 in GT dists
     
        set_end = gt_size

        for i in range(count, gt_size):
          is_close = abs(dist_tie_check - true_dists[i] ) < 1e-6 
          if not is_close:
            set_end = i
            break

        found_tie = set_end > count

        recall =  len(set(true_ids[:set_end]) & set(run_ids))
 
    return recall, found_tie

def get_recall_values(true_nn, run_nn, count, count_ties=True):
    true_ids, true_dists = true_nn
    if not count_ties:
        true_ids = true_ids[:, :count]
        assert true_ids.shape == run_nn.shape
    recalls = np.zeros(len(run_nn))
    queries_with_ties = 0
    # TODO probably not very efficient
    for i in range(len(run_nn)):
        if count_ties:
            recalls[i], found_tie = compute_recall_with_distance_ties(true_ids[i], true_dists[i], run_nn[i], count)
            if found_tie: queries_with_ties += 1 
        else:
            recalls[i] = compute_recall_without_distance_ties(true_ids[i], run_nn[i], count)
    return (np.mean(recalls) / float(count),
            np.std(recalls) / float(count),
            recalls,
            queries_with_ties)

def knn(true_nn, run_nn, count):
    print('Computing knn metrics')
    knn_metrics = {}
    mean, std, recalls, queries_with_ties = get_recall_values(true_nn, run_nn, count)
    if queries_with_ties>0:
        print("Warning: %d/%d queries contained ties accounted for in recall" % (queries_with_ties, len(run_nn)))
    knn_metrics['mean'] = mean
    knn_metrics['std'] = std
    knn_metrics['recalls'] = recalls
  
    return knn_metrics