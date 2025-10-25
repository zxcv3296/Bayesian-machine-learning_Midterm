#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
utils.py
- 난수생성기, 수치 안정 유틸, 다변량 정규 로그확률
"""

from __future__ import annotations
import numpy as np

def get_rng(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(seed)

def logsumexp(a: np.ndarray, axis: int = None, keepdims: bool = False) -> np.ndarray:
    """
    수치안정 log-sum-exp
    """
    a_max = np.max(a, axis=axis, keepdims=True)
    s = np.sum(np.exp(a - a_max), axis=axis, keepdims=True)
    out = a_max + np.log(s + 1e-300)
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out

def regularize_cov(S: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    대칭화 + 대각선에 eps 추가
    """
    S = 0.5 * (S + S.T)
    d = S.shape[0]
    return S + eps * np.eye(d)

def mvn_logpdf(x: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """
    다변량 정규분포 로그확률.
    x: (N, D), mu: (D,), Sigma: (D,D)
    return: (N,)
    """
    D = x.shape[1]
    Sigma = regularize_cov(Sigma)
    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        # 최후의 보정
        vals, vecs = np.linalg.eigh(Sigma)
        vals = np.clip(vals, 1e-12, None)
        Sigma = (vecs * vals) @ vecs.T
        L = np.linalg.cholesky(Sigma)

    # (x - mu)
    xm = x - mu[None, :]
    # 해결: solve_triangular 없이 Cholesky로 선형시스템 해결
    # y = L^{-1} (x-mu)^T  -> 행렬 형태 계산
    y = np.linalg.solve(L, xm.T)  # (D, N)
    quad = np.sum(y**2, axis=0)   # (N,)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (D * np.log(2.0 * np.pi) + logdet + quad)
