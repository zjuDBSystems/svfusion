> [!note]
> FFANNS (Fusion and Fresh Approximate Nearest Neighbor Search) is a new CPU/GPU cooperative framework designed for efficient and scalable streaming vector search. It builds on cuVS, a library derived from the approximate nearest neighbors and clustering algorithms in the [RAPIDS RAFT](https://github.com/rapidsai/raft) library, a suite of machine learning and data mining primitives. 

---
### What is FFANNS
FFANNS leverages the cooperative capabilities of both CPU and GPU to enhance system throughput and support large-scale streaming vector search. By combining the processing power of GPUs with the flexibility of CPUs, FFANNS can handle larger datasets—beyond GPU memory limitations—while maintaining high performance.  

---
### Installing FFANNS
As FFANNS is built upon cuVS 24.12, conda environment scripts are provided for installing the necessary dependencies to build FFANNS from source.
```
conda env create --name cuvs -f conda/environments/all_cuda-118_arch-x86_64.yaml
conda activate cuvs
```