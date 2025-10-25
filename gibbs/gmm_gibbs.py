#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gmm_gibbs.py
- NumPy만으로 구현한 Bayesian GMM (유한 K), Uncollapsed Gibbs Sampling
- 사전분포: π ~ Dir(α),  (μ_k, Λ_k) ~ NIW(m0, κ0, ν0, W0)  (Λ는 precision, Wishart 분포)
- 샘플링 순서(반복):
  1) z_n | ...  ~ Cat( proportional to π_k * N(x_n | μ_k, Λ_k^{-1}) )
  2) π   | ...  ~ Dir(α + N_k)
  3) Λ_k | ...  ~ Wishart(ν_k, W_k)         (posterior)
     μ_k | Λ_k  ~ N(m_k, (κ_k Λ_k)^{-1})    (posterior)
- 최종 파라미터는 수집 구간의 샘플 평균으로 추정하고, 예측 라벨은 그 평균 파라미터에 대한 MAP로 계산.

실행 예:
  python gibbs/gmm_gibbs.py --data data/G2.txt --K 2 --burnin 500 --iters 2000 --thin 5 --seed 42 \
    --plot_out runs/gibbs_K2.png --save_params runs/gibbs_K2.npz
"""

from __future__ import annotations
import argparse
import numpy as np
from pathlib import Path
import sys

# 로컬 import 경로 추가
sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.io_utils import load_g2, zscore
from common.utils import get_rng, mvn_logpdf, regularize_cov, logsumexp
from common.metrics import (
    log_likelihood_gmm, purity_score, adjusted_rand_index, normalized_mutual_info
)
from common.viz import plot_scatter_with_gaussians

# --------------------------
# Wishart / Inv-Wishart 샘플러 (Bartlett)
# --------------------------

def _sample_wishart(rng: np.random.Generator, df: int, scale: np.ndarray) -> np.ndarray:
    """
    Wishart(df, scale) 샘플. (scale은 양정치 행렬)
    Bartlett 분해 사용.
    반환값 W는 D×D.
    """
    D = scale.shape[0]
    assert df >= D, "Wishart dof must be >= dimension"
    # Cholesky of scale
    L = np.linalg.cholesky(regularize_cov(scale))
    # Bartlett A (lower-triangular)
    A = np.zeros((D, D))
    for i in range(D):
        A[i, i] = np.sqrt(rng.chisquare(df - i))
        for j in range(i):
            A[i, j] = rng.standard_normal()
    C = L @ A
    W = C @ C.T
    return W

def _sample_invwishart(rng: np.random.Generator, df: int, scale: np.ndarray) -> np.ndarray:
    """
    Inv-Wishart(df, Psi) 샘플 = ( Wishart(df, Psi^{-1}) )^{-1}
    여기서는 precision Λ ~ Wishart(ν, W) 를 주로 쓰므로
    공분산 Σ ~ Inv-Wishart(ν, W^{-1}) 와 쌍대관계.
    """
    # scale^{-1}
    S_inv = np.linalg.inv(regularize_cov(scale))
    W = _sample_wishart(rng, df, S_inv)
    Sigma = np.linalg.inv(regularize_cov(W))
    return Sigma

# --------------------------
# NIW posterior 업데이트 유틸
# --------------------------

def _posterior_niw_stats(m0: np.ndarray, kappa0: float, nu0: int, W0: np.ndarray,
                         Nk: int, sum_x: np.ndarray, sum_xxT: np.ndarray):
    """
    NIW prior:  Λ ~ Wishart(nu0, W0),   μ|Λ ~ N(m0, (kappa0 Λ)^{-1})
    데이터 충분통계:
      Nk, sum_x = Σ x_n, sum_xxT = Σ x_n x_n^T
    posterior:
      kappa_n = kappa0 + Nk
      m_n     = (kappa0*m0 + sum_x) / kappa_n
      nu_n    = nu0 + Nk
      W_n^{-1}= W0^{-1} + Σ(x_n x_n^T) - Nk * xbar * xbar^T + (kappa0*Nk/kappa_n) (xbar - m0)(xbar - m0)^T
              = W0^{-1} + scatter + mean_adj
    여기서 scatter = Σ(x_n x_n^T) - Nk * xbar xbar^T
    """
    D = m0.shape[0]
    kappa_n = kappa0 + Nk
    if Nk == 0:
        m_n = m0.copy()
        nu_n = nu0
        W0_inv = np.linalg.inv(regularize_cov(W0))
        W_n_inv = W0_inv.copy()
        return m_n, kappa_n, nu_n, np.linalg.inv(regularize_cov(W_n_inv))

    xbar = sum_x / Nk
    scatter = sum_xxT - Nk * np.outer(xbar, xbar)
    mean_diff = (xbar - m0).reshape(D, 1)
    W0_inv = np.linalg.inv(regularize_cov(W0))
    W_n_inv = W0_inv + scatter + (kappa0 * Nk / kappa_n) * (mean_diff @ mean_diff.T)
    nu_n = nu0 + Nk
    m_n = (kappa0 * m0 + sum_x) / kappa_n
    W_n = np.linalg.inv(regularize_cov(W_n_inv))
    return m_n, kappa_n, nu_n, W_n

# --------------------------
# 메인 Gibbs
# --------------------------

def run_gibbs(
    X: np.ndarray, K: int, seed: int = 42,
    burnin: int = 500, iters: int = 2000, thin: int = 5,
    alpha: float | None = None,
    m0: np.ndarray | None = None, kappa0: float = 1e-3,
    nu0: int | None = None, W0: np.ndarray | None = None
):
    """
    반환: dict with 평균 파라미터(pi, mu, Sigma), 샘플 로그가능도 평균, y_pred 등
    """
    rng = get_rng(seed)
    N, D = X.shape
    if alpha is None:
        alpha = 1.0 / K  # symmetric Dirichlet
    if m0 is None:
        m0 = X.mean(axis=0)
    if nu0 is None:
        nu0 = D + 2
    if W0 is None:
        # 데이터 공분산의 역행렬 대략치 → Wishart scale 매트릭스로 사용
        S = np.cov(X.T)
        W0 = np.linalg.inv(regularize_cov(S))

    # 초기화
    # z 무작위, π 균등, (μ, Σ)는 대략적으로 데이터 통계 기반
    z = rng.integers(0, K, size=N)
    Nk = np.bincount(z, minlength=K)

    pi = np.ones(K) / K
    mus = np.stack([X[rng.choice(N)] for _ in range(K)], axis=0)  # 랜덤 픽
    Sigmas = np.stack([np.cov(X.T) for _ in range(K)], axis=0)
    for k in range(K):
        Sigmas[k] = regularize_cov(Sigmas[k])

    # 샘플 수집 버퍼
    kept = []
    ll_kept = []

    def _sufficient_stats_for_k(k: int):
        idx = (z == k)
        n = int(idx.sum())
        if n == 0:
            return 0, np.zeros(D), np.zeros((D, D))
        Xk = X[idx]
        sum_x = Xk.sum(axis=0)
        sum_xxT = Xk.T @ Xk
        return n, sum_x, sum_xxT

    for t in range(burnin + iters):
        # 1) z 업데이트
        # p(z_n=k) ∝ π_k N(x_n | μ_k, Σ_k)
        log_rho = np.zeros((N, K))
        for k in range(K):
            log_rho[:, k] = np.log(pi[k] + 1e-300) + mvn_logpdf(X, mus[k], Sigmas[k])
        log_norm = logsumexp(log_rho, axis=1, keepdims=True)
        prob = np.exp(log_rho - log_norm)
        # 카테고리 샘플
        cum = np.cumsum(prob, axis=1)
        u = rng.random(N)[:, None]
        z = (u > cum).sum(axis=1)

        # 2) π ~ Dir(α + N_k)
        Nk = np.bincount(z, minlength=K)
        pi = rng.dirichlet(alpha + Nk)

        # 3) (μ_k, Σ_k) posterior에서 샘플
        for k in range(K):
            nk, sum_x, sum_xxT = _sufficient_stats_for_k(k)
            m_n, kappa_n, nu_n, W_n = _posterior_niw_stats(m0, kappa0, nu0, W0, nk, sum_x, sum_xxT)
            # Λ_k ~ Wishart(ν_n, W_n),  Σ_k = Λ_k^{-1}
            # 여기서는 Σ를 바로 샘플(Inv-Wishart)로 얻어도 되지만, Wishart→역변환도 동일
            # 간단히 Inv-Wishart 샘플러 사용:
            Sigma_k = _sample_invwishart(rng, nu_n, np.linalg.inv(regularize_cov(W_n)))
            # μ_k | Σ_k ~ N(m_n, Σ_k / kappa_n)
            L = np.linalg.cholesky(regularize_cov(Sigma_k / kappa_n))
            mu_k = m_n + L @ rng.standard_normal(D)
            Sigmas[k] = regularize_cov(Sigma_k)
            mus[k] = mu_k

        # 수집
        if t >= burnin and ((t - burnin) % thin == 0):
            kept.append((pi.copy(), mus.copy(), Sigmas.copy()))
            ll = log_likelihood_gmm(X, pi, mus, Sigmas)
            ll_kept.append(ll)

    # 사후 평균 파라미터
    pis_mean = np.mean([k[0] for k in kept], axis=0)
    mus_mean = np.mean([k[1] for k in kept], axis=0)
    Sigmas_mean = np.mean([k[2] for k in kept], axis=0)

    # 최종 예측 라벨: 평균 파라미터로 책임도 계산해서 argmax
    log_rho = np.zeros((N, K))
    for k in range(K):
        log_rho[:, k] = np.log(pis_mean[k] + 1e-300) + mvn_logpdf(X, mus_mean[k], Sigmas_mean[k])
    y_pred = np.argmax(log_rho, axis=1)
    ll_pp_mean = float(np.mean(ll_kept)) if ll_kept else None

    return {
        "pis": pis_mean, "mus": mus_mean, "Sigmas": Sigmas_mean,
        "y_pred": y_pred, "ll_pp_mean": ll_pp_mean, "samples": len(kept)
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--burnin", type=int, default=500)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--thin", type=int, default=5)
    ap.add_argument("--standardize", type=int, default=0)
    ap.add_argument("--plot_out", type=str, default="")
    ap.add_argument("--save_params", type=str, default="")
    args = ap.parse_args()

    X, y_true = load_g2(args.data)
    if args.standardize == 1:
        X = zscore(X)

    out = run_gibbs(
        X, args.K, seed=args.seed,
        burnin=args.burnin, iters=args.iters, thin=args.thin
    )

    pis, mus, Sigmas, y_pred = out["pis"], out["mus"], out["Sigmas"], out["y_pred"]
    N = X.shape[0]
    print(f"[Gibbs] K={args.K} samples_collected={out['samples']}  LL_mean={out['ll_pp_mean']}")

    if y_true is not None and y_true.shape[0] == N:
        purity = purity_score(y_true, y_pred)
        ari = adjusted_rand_index(y_true, y_pred)
        nmi = normalized_mutual_info(y_true, y_pred)
        print(f"[Gibbs] Purity={purity:.4f}  ARI={ari:.4f}  NMI={nmi:.4f}")

    if X.shape[1] == 2 and args.plot_out:
        Path(args.plot_out).parent.mkdir(parents=True, exist_ok=True)
        plot_scatter_with_gaussians(X, y_pred, mus, Sigmas,
                                    title=f"Gibbs (K={args.K})",
                                    save_path=args.plot_out)

    if args.save_params:
        Path(args.save_params).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.save_params, pis=pis, mus=mus, Sigmas=Sigmas,
                 y_pred=y_pred, ll_pp_mean=out["ll_pp_mean"], samples=out["samples"],
                 K=args.K, seed=args.seed)

if __name__ == "__main__":
    main()
