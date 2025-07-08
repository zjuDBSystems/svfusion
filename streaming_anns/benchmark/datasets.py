import gzip
import shutil
import math
import numpy
import os
import random
import sys
import struct
import time

import numpy as np
from scipy.sparse import csr_matrix

from urllib.request import urlretrieve

from .dataset_io import (
    xbin_mmap, download_accelerated, download, sanitize,
    knn_result_read, range_result_read, read_sparse_matrix,
    write_sparse_matrix,
)

BASEDIR = "/data2/pyc/data/"

class Dataset():
    def get_dataset_fn(self):
        """
        Return filename of dataset file.
        """
        pass
    def get_dataset(self):
        """
        Return memmapped version of the dataset.
        """
        pass
    def get_dataset_iterator(self, bs=512, split=(1, 0)):
        """
        Return iterator over blocks of dataset of size at most 512.
        The split argument takes a pair of integers (n, p) where p = 0..n-1
        The dataset is split in n shards, and the iterator returns only shard #p
        This makes it possible to process the dataset independently from several
        processes / threads.
        """
        pass
    def get_queries(self):
        """
        Return (nq, d) array containing the nq queries.
        """
        pass
    def get_private_queries(self):
        """
        Return (private_nq, d) array containing the private_nq private queries.
        """
        pass
    def get_groundtruth(self, k=None):
        """
        Return (nq, k) array containing groundtruth indices
        for each query."""
        pass

    def search_type(self):
        """
        "knn" or "range" or "knn_filtered"
        """
        pass

    def distance(self):
        """
        "euclidean" or "ip" or "angular"
        """
        pass

    def data_type(self):
        """
        "dense" or "sparse"
        """
        pass

    def default_count(self):
        """ number of neighbors to return """
        return 10

    def short_name(self):
        return f"{self.__class__.__name__}-{self.nb}"
    
    def __str__(self):
        return (
            f"Dataset {self.__class__.__name__} in dimension {self.d}, with distance {self.distance()}, "
            f"search_type {self.search_type()}, size: Q {self.nq} B {self.nb}")


class LocalDataset(Dataset):
    """本地数据集类"""
    def __init__(self, base_path, dataset_file, query_file, groundtruth_file=None):
        # elf.basedir = base_path
        self.basedir = os.path.join(BASEDIR, base_path)
        self.ds_fn = dataset_file
        self.qs_fn = query_file
        self.gt_fn = groundtruth_file

    def get_dataset_fn(self):
        fn = os.path.join(self.basedir, self.ds_fn)
        if os.path.exists(fn):
            return fn
        else:
            raise RuntimeError(f"文件 {fn} 不存在")

    def get_dataset_iterator(self, bs=512, split=(1,0)):
        nsplit, rank = split
        i0, i1 = self.nb * rank // nsplit, self.nb * (rank + 1) // nsplit
        filename = self.get_dataset_fn()
        x = xbin_mmap(filename, dtype=self.dtype, maxn=self.nb)
        assert x.shape == (self.nb, self.d)
        for j0 in range(i0, i1, bs): 
            j1 = min(j0 + bs, i1)
            yield sanitize(x[j0:j1])

    def get_data_in_range(self, start, end):
        assert start >= 0
        assert end <= self.nb
        filename = self.get_dataset_fn()
        x = xbin_mmap(filename, dtype=self.dtype, maxn=self.nb)
        return x[start:end]

    def get_dataset(self):
        assert self.nb <= 10**7, "dataset too large, use iterator"
        slice = next(self.get_dataset_iterator(bs=self.nb))
        return sanitize(slice)

    def get_queries(self):
        filename = os.path.join(self.basedir, self.qs_fn)
        x = xbin_mmap(filename, dtype=self.dtype)
        assert x.shape == (self.nq, self.d)
        return sanitize(x)

    def search_type(self):
        return "knn"
    
    def get_groundtruth(self, k=None):
        assert self.gt_fn is not None
        assert self.search_type() in ("knn", "knn_filtered")

        I, D = knn_result_read(os.path.join(self.basedir, self.gt_fn))
        assert I.shape[0] == self.nq
        if k is not None:
            assert k <= 100
            I = I[:, :k]
            D = D[:, :k]
        return I, D
    
class MSTuringANNS30M(LocalDataset):
    def __init__(self):
        # 首先设置数据集的基本属性
        nb_M = 30
        self.nb_M = nb_M
        self.nb = 10**6 * nb_M   
        self.d = 100           
        self.nq = 10000      
        self.dtype = "float32"   
        
        # 构建文件路径
        base_path = "msturing30M"
        dataset_file = "msturing30M.fbin"
        query_file = "testQuery10K.fbin"
        gt_file = "gt100-private10K-queries.bin"
        
        super().__init__(
            base_path=base_path,
            dataset_file=dataset_file,
            query_file=query_file,
            groundtruth_file=gt_file
        )
    
    def distance(self):
        return "euclidean"

class SIFT1B(LocalDataset):
    def __init__(self, nb_M=1000):
        # 首先设置数据集的基本属性
        self.nb_M = nb_M
        self.nb = 10**6 * nb_M 
        self.d = 128           
        self.nq = 10000      
        self.dtype = "uint8"
        
        # 构建文件路径
        base_path = "sift1B"
        dataset_file = "base.1B.u8bin"
        query_file = "query.public.10K.u8bin"
        gt_file = "gt100-private10K-queries.bin"
        
        super().__init__(
            base_path=base_path,
            dataset_file=dataset_file,
            query_file=query_file,
            groundtruth_file=gt_file
        )
    
    def distance(self):
        return "euclidean"
    
DATASETS = {
    'msturing-30M': lambda : MSTuringANNS30M(),
    'sift-1B': lambda : SIFT1B()
}