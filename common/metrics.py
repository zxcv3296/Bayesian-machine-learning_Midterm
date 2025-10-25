#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
metrics.py
- 로그가능도, AIC/BIC
- Purity, ARI, NMI (NumPy로 직접 구현)
"""

from __future__ import annotations
import numpy as np
from typing import Tuple
from .utils import logsumexp, mvn_logpdf

def log_likelihood_gmm(X: np.ndarray, pis: np.ndarray, mus: np.ndarray, Sigmas: np.ndarray) -> float:
    """
    X: (N,D), pis: (K,), mus: (K,D), Sigmas: (K,D,D)
    """
    N, D = X.shape
    K = pis.shape[0]
    log_comp = np.zeros((N, K), dtype=float)
    for k in range(K):
        log_comp[:, k] = np.log(pis[k] + 1e-300) + mvn_logpdf(X, mus[k], Sigmas[k])
    ll = np.sum(logsumexp(log_comp, axis=1))
    return float(ll)

def num_params_gmm(K: int, D: int) -> int:
    """
    혼합계수: K-1
    평균: K*D
    공분산(대칭): K * D*(D+1)/2
    """
    return (K - 1) + K * D + K * (D * (D + 1) // 2)

def aic(ll: float, p: int) -> float:
    return -2.0 * ll + 2.0 * p

def bic(ll: float, p: int, N: int) -> float:
    return -2.0 * ll + p * np.log(N)

# ---------------------------
# 클러스터링 지표(Contingency 기반)
# ---------------------------

def contingency_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    t_classes = y_true.max() + 1
    p_classes = y_pred.max() + 1
    C = np.zeros((t_classes, p_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        C[t, p] += 1
    return C

def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    C = contingency_matrix(y_true, y_pred)
    return np.sum(C.max(axis=0)) / np.sum(C)

def _comb2(n: np.ndarray) -> np.ndarray:
    return n * (n - 1) // 2

def adjusted_rand_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    C = contingency_matrix(y_true, y_pred).astype(np.int64)
    sum_comb_c = _comb2(C).sum()
    a = C.sum(axis=1)
    b = C.sum(axis=0)
    sum_comb_a = _comb2(a).sum()
    sum_comb_b = _comb2(b).sum()
    N = C.sum()
    prod = sum_comb_a * sum_comb_b / _comb2(np.array([N], dtype=np.int64))[0]
    mean = (sum_comb_a + sum_comb_b) / 2.0
    denom = mean - prod
    if denom == 0:
        return 1.0
    ari = (sum_comb_c - prod) / denom
    return float(ari)

def normalized_mutual_info(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    """
    NMI = 2 * I(Y;Z) / (H(Y)+H(Z))
    """
    C = contingency_matrix(y_true, y_pred).astype(float)
    N = C.sum()
    py = C.sum(axis=1) / N
    pz = C.sum(axis=0) / N
    P = C / N
    # Mutual Information
    MI = 0.0
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if P[i, j] > 0:
                MI += P[i, j] * np.log((P[i, j] + eps) / ((py[i] + eps) * (pz[j] + eps)))
    HY = -np.sum(py * np.log(py + eps))
    HZ = -np.sum(pz * np.log(pz + eps))
    denom = HY + HZ
    if denom <= 0:
        return 0.0
    return float(2.0 * MI / denom)
