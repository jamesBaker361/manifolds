# metrics for manifoldness
# subspace capture
# plaigarized from https://github.com/goodfire-ai/sae-manifold/blob/main/subspace_capture.py

import argparse
from pathlib import Path

import torch
import numpy as np
from sklearn.decomposition import PCA
from overcomplete.sae import SAE

def _detect_elbow(curve, min_k=1):
    """Maximum-distance-to-chord elbow detector on a monotone curve.

    Returns the index of the elbow point; used by the greedy methods below
    to suggest a cutoff when neither a variance threshold nor ``max_k`` is hit.
    """
    n = len(curve)
    if n <= min_k + 1:
        return n - 1
    x = np.arange(n, dtype=float)
    y = np.asarray(curve, dtype=float)
    p0 = np.array([x[0], y[0]])
    p1 = np.array([x[-1], y[-1]])
    line_vec = p1 - p0
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-10:
        return n - 1
    line_unit = line_vec / line_len
    dists = np.abs(np.cross(line_unit, p0 - np.column_stack([x, y])))
    dists[:min_k] = -1
    return int(np.argmax(dists))


def find_support_greedy(activations, decoder, max_k=100, var_threshold=0.95):
    """Greedy subspace pursuit over decoder directions.

    At each step adds the decoder atom whose direction captures the most
    remaining variance of the centered manifold activations. Returns the
    selected feature indices, the cumulative variance-explained curve, and
    the suggested elbow ``k``.
    """
    X = np.asarray(activations, dtype=np.float32)
    X = X - X.mean(0)
    total_ss = (X ** 2).sum()
    if total_ss < 1e-10:
        return np.array([], dtype=int), np.array([]), 0

    candidates = np.arange(decoder.shape[0])
    D_cand = decoder[candidates]
    d_norms_sq = (D_cand ** 2).sum(1)
    alive = d_norms_sq > 1e-10

    selected_local = []
    selected_global = []
    var_curve = []
    residual = X.copy()

    for _ in range(max_k):
        projections = residual @ D_cand.T
        scores = (projections ** 2).sum(0) / d_norms_sq.clip(1e-10)
        scores[~alive] = -np.inf
        for i in selected_local:
            scores[i] = -np.inf
        best = int(np.argmax(scores))
        if scores[best] <= 0:
            break
        selected_local.append(best)
        selected_global.append(int(candidates[best]))
        D_sel = decoder[selected_global]
        _, s, Vt = np.linalg.svd(D_sel, full_matrices=False)
        basis = Vt[s > 1e-8]
        residual = X - (X @ basis.T) @ basis
        explained = 1.0 - (residual ** 2).sum() / total_ss
        var_curve.append(float(explained))
        if explained >= var_threshold:
            break

    var_curve = np.array(var_curve)
    elbow_k = (_detect_elbow(var_curve, min_k=1) + 1
               if len(var_curve) > 2 else len(var_curve))
    return np.array(selected_global), var_curve, elbow_k


def get_R2(data,
            sae:SAE,
             max_k=100,
             var_threshold=0.95
            ):
    codes=sae.encode(data)
    decoder = sae.get_dictionary()
    selected_global,var_curve, elbow_k =find_support_greedy(codes,decoder,max_k,var_threshold)
    codes_selected=codes[:,selected_global]
    data_selected= codes_selected @ decoder[selected_global,:]

    data_mean = data.mean(dim=0)

    numerator = ((data - data_selected)**2).sum()
    denominator = ((data - data_mean)**2).sum()

    return 1 - numerator/denominator