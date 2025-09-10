"""
FAISS index helpers for CLIP features.

Provides functions to build a flat inner-product index and search it by a query vector.
"""

from typing import List
import numpy as np
import faiss
import torch


def build_index(feats: np.ndarray, use_gpu: bool = False) -> faiss.Index:
    d = feats.shape[1]
    idx = faiss.IndexFlatIP(d)
    if use_gpu and torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        idx = faiss.index_cpu_to_gpu(res, 0, idx)
    idx.add(feats.astype("float32"))
    return idx


def search_index(qvec: np.ndarray, index: faiss.Index, paths: List[str], k: int = 10):
    k = max(1, min(k, index.ntotal))
    sim, ids = index.search(qvec[None, :].astype("float32"), k)
    out = []
    for j, i in enumerate(ids[0]):
        if i == -1:
            continue
        out.append((paths[i], float(sim[0, j])))
    return out