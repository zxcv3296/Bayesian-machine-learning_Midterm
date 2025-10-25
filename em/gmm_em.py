#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gmm_em.py
- NumPy만으로 구현한 GMM EM 학습 스크립트 (수치안정/붕괴방지 포함)

실행 예:
  python em/gmm_em.py --data data/G2.txt --K 2 --max_iter 200 --tol 1e-6 \
    --seed 42 --plot_out runs/em_K2.png --curve_out runs/em_ll_curve.png \
    --save_params runs/em_K2.npz --standardize 0
"""

from __future__ import annotations
import argparse
import numpy as np
from pathlib import Path
import sys

# 로컬 import
sys.path.append(str(Path(__file__).resolve().parents[1]))  # gmm-g2/
from common.io_utils import load_g2, zscore
from common.utils import get_rng, logsumexp, mvn_logpdf, regularize_cov
from common.metrics import (
    log_likelihood_gmm, num_params_gmm, aic, bic,
    purity_score, adjusted_rand_index, normalized_mutual_info
)
from common.viz import plot_scatter_with_gaussians, plot_curve


# ---------------------------
# 수치안정 유틸
# ---------------------------
def _sym_psd(A: np.ndarray) -> np.ndarray:
    """대칭화 + 고윳값 0 하한(PSD 보정)"""
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, 0.0)
    return (V * w) @ V.T

def _stabilize_cov(S: np.ndarray, eps: float = 1e-6, min_eig: float = 1e-8) -> np.ndarray:
    """공분산 Σ를 대칭/PSD화하고 최소 고윳값 바닥 및 eps*I 추가"""
    S = _sym_psd(S)
    w, V = np.linalg.eigh(S)
    w = np.maximum(w, min_eig)
    S = (V * w) @ V.T
    S = S + eps * np.eye(S.shape[0])
    return S


# ---------------------------
# EM 단계
# ---------------------------
def init_params(X: np.ndarray, K: int, rng: np.random.Generator, eps: float = 1e-6):
    N, D = X.shape
    # 혼합계수: Dirichlet(1)
    alpha = np.ones(K)
    pi = rng.dirichlet(alpha)
    # 평균: 데이터에서 K개 랜덤 선택
    idx = rng.choice(N, size=K, replace=False)
    mus = X[idx].copy()
    # 공분산: 데이터 공분산에 epsI 추가, K개 복제
    S = np.cov(X.T)
    S = _stabilize_cov(S, eps=eps)
    Sigmas = np.stack([S.copy() for _ in range(K)], axis=0)
    return pi, mus, Sigmas

def e_step(X: np.ndarray, pis: np.ndarray, mus: np.ndarray, Sigmas: np.ndarray) -> np.ndarray:
    N, D = X.shape
    K = pis.shape[0]
    log_rho = np.zeros((N, K), dtype=float)
    for k in range(K):
        log_rho[:, k] = np.log(pis[k] + 1e-300) + mvn_logpdf(X, mus[k], Sigmas[k])
    log_norm = logsumexp(log_rho, axis=1, keepdims=True)
    gamma = np.exp(log_rho - log_norm)  # 책임도
    return gamma

def m_step(
    X: np.ndarray,
    gamma: np.ndarray,
    base_cov: np.ndarray,
    eps: float = 1e-6,
    min_comp_resp: float = 1e-6,
    min_eig: float = 1e-8,
    rng: np.random.Generator | None = None
):
    """
    수치안정 M-step:
      - 군집 책임합 Nk가 너무 작으면(≈빈 군집) 재초기화
      - 공분산 Σ는 대칭화/고윳값 바닥/εI 추가로 안정화
    """
    N, D = X.shape
    K = gamma.shape[1]
    Nk = gamma.sum(axis=0) + 1e-300
    pis = Nk / N

    mus = (gamma.T @ X) / Nk[:, None]
    Sigmas = np.zeros((K, D, D), dtype=float)

    for k in range(K):
        if Nk[k] < min_comp_resp:
            # 거의 빈 군집: 평균은 글로벌 평균 근처로, 공분산은 base_cov로 리셋
            if rng is None:
                rng = np.random.default_rng()
            jitter = rng.normal(scale=1e-3, size=D)
            mus[k] = X.mean(axis=0) + jitter
            Sigmas[k] = base_cov.copy()
            Sigmas[k] = _stabilize_cov(Sigmas[k], eps=eps, min_eig=min_eig)
            continue

        Xm = X - mus[k]
        # 가중 공분산 (외적 기반, 오프대각 포함)
        Sk = (gamma[:, [k]] * Xm).T @ Xm / Nk[k]
        Sk = _stabilize_cov(Sk, eps=eps, min_eig=min_eig)
        Sigmas[k] = Sk

    # π가 0에 너무 가까워지는 것 방지해 살짝 클리핑(옵션)
    pis = np.maximum(pis, 1e-12)
    pis = pis / pis.sum()
    return pis, mus, Sigmas

def predict_labels(gamma: np.ndarray) -> np.ndarray:
    return np.argmax(gamma, axis=1)

def run_em(
    X: np.ndarray,
    K: int,
    seed: int = 42,
    max_iter: int = 200,
    tol: float = 1e-6,
    eps_cov: float = 1e-6,
):
    rng = get_rng(seed)
    pis, mus, Sigmas = init_params(X, K, rng, eps=eps_cov)
    base_cov = Sigmas[0].copy()  # 재초기화용 베이스 공분산

    ll_hist: list[float] = []
    prev_ll = -np.inf

    for it in range(1, max_iter + 1):
        # E-step
        gamma = e_step(X, pis, mus, Sigmas)
        # M-step (수치안정 포함)
        pis, mus, Sigmas = m_step(
            X, gamma, base_cov, eps=eps_cov, rng=rng
        )
        # 로그우도
        ll = log_likelihood_gmm(X, pis, mus, Sigmas)
        ll_hist.append(ll)

        # 수렴 체크
        if np.abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    # 마지막 책임도로 라벨 추정
    gamma = e_step(X, pis, mus, Sigmas)
    y_pred = predict_labels(gamma)
    return pis, mus, Sigmas, y_pred, ll_hist


# ---------------------------
# 메인
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_iter", type=int, default=200)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--standardize", type=int, default=0, help="1이면 Z-score 적용")
    ap.add_argument("--plot_out", type=str, default="")
    ap.add_argument("--curve_out", type=str, default="")
    ap.add_argument("--save_params", type=str, default="")
    args = ap.parse_args()

    X, y_true = load_g2(args.data)
    if args.standardize == 1:
        X = zscore(X)

    pis, mus, Sigmas, y_pred, ll_hist = run_em(
        X, args.K, seed=args.seed, max_iter=args.max_iter, tol=args.tol
    )

    # 정량 평가
    N, D = X.shape
    ll = ll_hist[-1]
    p = num_params_gmm(args.K, D)
    _aic = aic(ll, p)
    _bic = bic(ll, p, N)

    print(f"[EM] K={args.K}  LL={ll:.4f}  AIC={_aic:.2f}  BIC={_bic:.2f}  iters={len(ll_hist)}")

    if y_true is not None and y_true.shape[0] == X.shape[0]:
        purity = purity_score(y_true, y_pred)
        ari = adjusted_rand_index(y_true, y_pred)
        nmi = normalized_mutual_info(y_true, y_pred)
        print(f"[EM] Purity={purity:.4f}  ARI={ari:.4f}  NMI={nmi:.4f}")
    else:
        purity = ari = nmi = None

    # 시각화 저장(2D일 때만) — 플롯은 반드시 "공분산"을 사용
    if X.shape[1] == 2 and args.plot_out:
        Path(args.plot_out).parent.mkdir(parents=True, exist_ok=True)
        plot_scatter_with_gaussians(
            X, y_pred, mus, Sigmas,
            title=f"EM (K={args.K})",
            save_path=args.plot_out,
            mat_type="cov"
        )

    if args.curve_out:
        Path(args.curve_out).parent.mkdir(parents=True, exist_ok=True)
        plot_curve(ll_hist, title="EM Log-Likelihood", ylabel="log-likelihood", save_path=args.curve_out)

    if args.save_params:
        Path(args.save_params).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.save_params,
            pis=pis, mus=mus, Sigmas=Sigmas, y_pred=y_pred, ll_hist=np.array(ll_hist, dtype=float),
            K=args.K, seed=args.seed, standardize=args.standardize,
            ll=ll, aic=_aic, bic=_bic, purity=purity, ari=ari, nmi=nmi
        )

if __name__ == "__main__":
    main()
