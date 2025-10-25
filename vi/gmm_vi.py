#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gmm_vi.py
- NumPy만으로 구현한 Bayesian GMM Mean-field VI
  q(π)=Dir(α̂),  q(μ_k, Λ_k)=Normal-Wishart(m̂_k, κ̂_k, ν̂_k, Ŵ_k),  q(z_n)=Cat(r_{nk})

주요 기대식(차원 D):
  E[log π_k] = ψ(α̂_k) - ψ(Σ_j α̂_j)
  E[log |Λ_k|] = Σ_{i=1}^D ψ((ν̂_k + 1 - i)/2) + D*log 2 + log|Ŵ_k|
  E[(x - m̂_k)^T Λ_k (x - m̂_k)] = D/κ̂_k + ν̂_k (x - m̂_k)^T Ŵ_k (x - m̂_k)

플롯용 공분산:
  E[Σ_k] = (Ŵ_k^{-1}) / (ν̂_k - D - 1)   (ν̂_k > D+1)

실행 예:
  python vi/gmm_vi.py --data data/G2.txt --K 2 --max_iter 500 --tol 1e-6 --seed 42 \
    --plot_out runs/vi_K2.png --curve_out runs/vi_elbo.png --save_params runs/vi_K2.npz
"""

from __future__ import annotations
import argparse, math
import numpy as np
from pathlib import Path
import sys

# 로컬 import
sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.io_utils import load_g2, zscore
from common.utils import get_rng, regularize_cov, logsumexp
from common.metrics import purity_score, adjusted_rand_index, normalized_mutual_info
from common.viz import plot_scatter_with_gaussians, plot_curve

# --------------------------
# 특수함수 (digamma, ln multivariate gamma)
# --------------------------

def digamma(x: float) -> float:
    """ψ(x) 근사: 작은 구간 재귀 + 비대칭 전개(아심프틱)"""
    result = 0.0
    while x < 8.0:
        result -= 1.0 / x
        x += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    # Bernoulli 다항 전개 (간단 버전)
    result += math.log(x) - 0.5 * inv - inv2 * (1.0/12.0 - inv2 * (1.0/120.0 - inv2 * (1.0/252.0)))
    return result

def psi_vec(v: np.ndarray) -> np.ndarray:
    return np.vectorize(digamma, otypes=[float])(v)

def log_multigamma(a: float, d: int) -> float:
    """ln Γ_d(a) = (d(d-1)/4) ln π + Σ_{i=1}^d ln Γ(a + (1-i)/2)"""
    term = 0.25 * d * (d - 1) * math.log(math.pi)
    s = sum(math.lgamma(a + (1.0 - i) * 0.5) for i in range(1, d + 1))
    return term + s

def log_dirichlet_norm(a: np.ndarray) -> float:
    """ln B(a)"""
    return float(np.sum(np.log(np.vectorize(math.lgamma)(a))) - math.lgamma(float(np.sum(a))))

def wishart_log_norm(nu: float, W: np.ndarray, d: int) -> float:
    """log Z(Wishart) = (ν/2) log|W| - (ν d /2) log 2 - ln Γ_d(ν/2)"""
    W = regularize_cov(W)
    return (nu/2.0) * float(np.log(np.linalg.det(W))) - (nu * d / 2.0) * math.log(2.0) - log_multigamma(nu/2.0, d)

# --------------------------
# VI 업데이트
# --------------------------

def run_vi(
    X: np.ndarray, K: int, seed: int = 42, max_iter: int = 500, tol: float = 1e-6,
    alpha0: float | None = None, m0: np.ndarray | None = None, kappa0: float = 1e-3,
    nu0: int | None = None, W0: np.ndarray | None = None, eps_cov: float = 1e-6
):
    rng = get_rng(seed)
    N, D = X.shape
    if alpha0 is None:
        alpha0 = 1.0 / K
    if m0 is None:
        m0 = X.mean(axis=0)
    if nu0 is None:
        nu0 = D + 2  # ν0 > D-1 필요
    if W0 is None:
        S = np.cov(X.T)
        W0 = np.linalg.inv(regularize_cov(S))  # Wishart scale(정밀도 쪽)

    # 초기 책임도 r (무작위)
    r = rng.random((N, K))
    r /= r.sum(axis=1, keepdims=True)

    # 사전 하이퍼
    alpha0_vec = np.full(K, alpha0, dtype=float)  # Dirichlet prior
    kappa0 = float(kappa0)
    nu0 = float(nu0)
    m0 = m0.astype(float)
    W0 = regularize_cov(W0)

    # 변분 파라미터 초기화
    alpha_hat = alpha0_vec.copy()
    kappa_hat = np.full(K, kappa0, dtype=float)
    nu_hat = np.full(K, nu0, dtype=float)
    W_hat = np.stack([W0.copy() for _ in range(K)], axis=0)  # Wishart scale
    m_hat = np.stack([m0.copy() for _ in range(K)], axis=0)

    elbo_hist: list[float] = []
    prev_elbo = -np.inf

    for it in range(1, max_iter + 1):
        # ---------- 충분통계 ----------
        Nk = r.sum(axis=0) + 1e-300       # (K,)
        xbar = (r.T @ X) / Nk[:, None]    # (K,D)

        # sum_x, sum_xxT
        sum_x = r.T @ X                   # (K,D)
        sum_xxT = np.zeros((K, D, D))
        for n in range(N):
            xn = X[n]
            outer = np.outer(xn, xn)
            sum_xxT += r[n, :, None, None] * outer[None, :, :]

        Sk = sum_xxT - Nk[:, None, None] * np.einsum('ki,kj->kij', xbar, xbar)  # (K,D,D)

        # ---------- posterior params ----------
        alpha_hat = alpha0_vec + Nk
        kappa_hat = kappa0 + Nk
        nu_hat = nu0 + Nk

        # W_hat: posterior Wishart scale
        # W_n^{-1} = W0^{-1} + Sk + (kappa0 * Nk / kappa_hat) (xbar - m0)(xbar - m0)^T
        W0_inv = np.linalg.inv(regularize_cov(W0))
        W_hat_inv = np.empty_like(W_hat)
        for k in range(K):
            diff = (xbar[k] - m0).reshape(D, 1)
            W_hat_inv[k] = W0_inv + Sk[k] + (kappa0 * Nk[k] / kappa_hat[k]) * (diff @ diff.T)
        for k in range(K):
            W_hat[k] = np.linalg.inv(regularize_cov(W_hat_inv[k]))

        # m_hat
        for k in range(K):
            m_hat[k] = (kappa0 * m0 + sum_x[k]) / kappa_hat[k]

        # ---------- 기대값 ----------
        # E[log π_k]
        E_log_pi = psi_vec(alpha_hat) - digamma(float(alpha_hat.sum()))

        # E[log |Λ_k|]
        E_log_det_Lambda = np.zeros(K, dtype=float)
        logdetW = np.zeros(K, dtype=float)
        for k in range(K):
            logdetW[k] = float(np.log(np.linalg.det(regularize_cov(W_hat[k]))))
            s = 0.0
            for i in range(1, D + 1):
                s += digamma(0.5 * (nu_hat[k] + 1.0 - i))
            E_log_det_Lambda[k] = s + D * math.log(2.0) + logdetW[k]

        # ---------- 책임도 업데이트 ----------
        # log ρ_nk = E[log π_k] + 0.5*(E[log|Λ_k|] - D*log(2π) - E[(x-m)^T Λ (x-m)])
        log_rho = np.zeros((N, K), dtype=float)
        for k in range(K):
            diff = X - m_hat[k][None, :]
            quad = nu_hat[k] * np.sum(diff @ regularize_cov(W_hat[k]) * diff, axis=1) + (D / kappa_hat[k])
            log_rho[:, k] = E_log_pi[k] + 0.5 * (E_log_det_Lambda[k] - D * math.log(2.0 * math.pi) - quad)

        # 정규화
        log_norm = logsumexp(log_rho, axis=1, keepdims=True)
        r = np.exp(log_rho - log_norm)

        # ---------- ELBO ----------
        # 1) E[log p(X|Z,μ,Λ)]
        term1 = 0.0
        for k in range(K):
            diff = X - m_hat[k]
            Q = nu_hat[k] * np.sum((diff @ regularize_cov(W_hat[k])) * diff, axis=1) + D / kappa_hat[k]
            term1 += 0.5 * np.sum(r[:, k] * (E_log_det_Lambda[k] - D * math.log(2.0 * math.pi) - Q))

        # 2) E[log p(Z|π)]
        term2 = float(np.sum(r * E_log_pi[None, :]))

        # 3) E[log p(π)] - E[log q(π)]
        term3 = log_dirichlet_norm(alpha0_vec) - log_dirichlet_norm(alpha_hat) \
                + float(np.sum((alpha_hat - alpha0_vec) * E_log_pi))

        # 4) Σ_k [E[log p(μ_k,Λ_k)] - E[log q(μ_k,Λ_k)]]
        #    (Bishop PRML 10.51, 10.60 등 식 구성)
        term4 = 0.0
        for k in range(K):
            # (a) Λ 부분
            # E[log p(Λ_k)]
            E_log_p_L = 0.5 * (nu0 - D - 1.0) * E_log_det_Lambda[k] \
                        - 0.5 * float(np.trace(np.linalg.inv(regularize_cov(W0)) @ (nu_hat[k] * regularize_cov(W_hat[k])))) \
                        + 0.5 * nu0 * float(np.log(np.linalg.det(regularize_cov(W0)))) \
                        - (nu0 * D / 2.0) * math.log(2.0) \
                        - log_multigamma(nu0/2.0, D)
            # E[log q(Λ_k)]  (Wishart 엔트로피 항 포함)
            # logZ(ν̂, Ŵ) + (ν̂ - D - 1)/2 E[log|Λ|] - 1/2 Tr(Ŵ^{-1} E[Λ])
            # Tr(Ŵ^{-1} E[Λ]) = Tr(Ŵ^{-1} * (ν̂ Ŵ)) = ν̂ * D
            E_log_q_L = wishart_log_norm(nu_hat[k], regularize_cov(W_hat[k]), D) \
                        + 0.5 * (nu_hat[k] - D - 1.0) * E_log_det_Lambda[k] \
                        - 0.5 * nu_hat[k] * D
            # (b) μ|Λ 부분
            # E[log p(μ_k|Λ_k)]  = 0.5*D*log(kappa0/(2π)) + 0.5*E[log|Λ_k|]
            #                        - 0.5*kappa0 * E[(μ_k-m0)^T Λ_k (μ_k-m0)]
            diffm = (m_hat[k] - m0).reshape(D, 1)
            quad_m = float(diffm.T @ regularize_cov(W_hat[k]) @ diffm)
            E_log_p_M_given_L = 0.5 * D * (math.log(kappa0) - math.log(2.0 * math.pi)) \
                                + 0.5 * E_log_det_Lambda[k] \
                                - 0.5 * kappa0 * (D / kappa_hat[k] + nu_hat[k] * quad_m)
            # E[log q(μ_k|Λ_k)] = 0.5*D*log(kappa_hat/(2π)) + 0.5*E[log|Λ_k|] - 0.5*D
            E_log_q_M_given_L = 0.5 * D * (math.log(kappa_hat[k]) - math.log(2.0 * math.pi)) \
                                + 0.5 * E_log_det_Lambda[k] \
                                - 0.5 * D

            term4 += (E_log_p_L - E_log_q_L) + (E_log_p_M_given_L - E_log_q_M_given_L)

        # 5) -E[log q(Z)]
        term5 = -float(np.sum(r * (np.log(r + 1e-300))))

        elbo = float(term1 + term2 + term3 + term4 + term5)
        elbo_hist.append(elbo)

        # 수렴 체크
        if abs(elbo - prev_elbo) < tol:
            break
        prev_elbo = elbo

    # 최종 라벨 (r argmax)
    y_pred = np.argmax(r, axis=1)

    # 모수의 점추정 (VI 기대값 기반)
    E_pi = alpha_hat / alpha_hat.sum()
    # E[Σ_k] = (Ŵ^{-1}) / (ν̂ - D - 1)   (ν̂ > D+1)
    Sigmas = np.zeros((K, D, D), dtype=float)
    mus = m_hat.copy()
    for k in range(K):
        denom = max(nu_hat[k] - D - 1.0, 1.0)  # 안정화
        Sigmas[k] = regularize_cov(np.linalg.inv(regularize_cov(W_hat[k])) / denom)

    return E_pi, mus, Sigmas, y_pred, elbo_hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_iter", type=int, default=500)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--standardize", type=int, default=0)
    ap.add_argument("--plot_out", type=str, default="")
    ap.add_argument("--curve_out", type=str, default="")
    ap.add_argument("--save_params", type=str, default="")
    args = ap.parse_args()

    X, y_true = load_g2(args.data)
    if args.standardize == 1:
        X = zscore(X)

    pis, mus, Sigmas, y_pred, elbo_hist = run_vi(
        X, args.K, seed=args.seed, max_iter=args.max_iter, tol=args.tol
    )

    print(f"[VI] K={args.K}  iters={len(elbo_hist)}  ELBO={elbo_hist[-1]:.4f}")

    if y_true is not None and y_true.shape[0] == X.shape[0]:
        purity = purity_score(y_true, y_pred)
        ari = adjusted_rand_index(y_true, y_pred)
        nmi = normalized_mutual_info(y_true, y_pred)
        print(f"[VI] Purity={purity:.4f}  ARI={ari:.4f}  NMI={nmi:.4f}")

    if X.shape[1] == 2 and args.plot_out:
        Path(args.plot_out).parent.mkdir(parents=True, exist_ok=True)
        # 플롯은 공분산을 직접 사용
        plot_scatter_with_gaussians(
            X, y_pred, mus, Sigmas,
            title=f"VI (K={args.K})", save_path=args.plot_out, mat_type="cov"
        )

    if args.curve_out:
        Path(args.curve_out).parent.mkdir(parents=True, exist_ok=True)
        plot_curve(elbo_hist, title="VI ELBO", ylabel="ELBO", save_path=args.curve_out)

    if args.save_params:
        Path(args.save_params).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.save_params,
            pis=pis, mus=mus, Sigmas=Sigmas, y_pred=y_pred,
            elbo_hist=np.asarray(elbo_hist, dtype=float),
            K=args.K, seed=args.seed
        )

if __name__ == "__main__":
    main()
