#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
viz.py
- 2D 산점도 + 가우시안 타원(1/2/3σ)
- 학습 곡선 플롯(감소 구간 표시 옵션)
- 공분산/정밀도 행렬 구분 지원 및 수치안정 처리

의존성: numpy, matplotlib (conda env.yml의 최소 구성과 호환)
"""

from __future__ import annotations
from typing import Iterable, Optional, Literal

import numpy as np
import matplotlib.pyplot as plt


# -------------------------------
# 내부 유틸
# -------------------------------
def _to_covariance(M: np.ndarray, mat_type: Literal["cov", "precision", "auto"] = "cov") -> np.ndarray:
    """
    입력 행렬 M을 공분산(Σ)으로 변환해서 반환.
    - mat_type="cov"      : 그대로 사용
    - mat_type="precision": Σ = M^{-1}
    - mat_type="auto"     : 간단 휴리스틱으로 결정(정밀도로 보이면 역행렬)

    수치안정 처리:
      - 대칭화: (M + M.T)/2
      - 역행렬 실패 시: 작은 εI 더해 재시도
    """
    assert M.shape[0] == M.shape[1], "정사각 행렬이어야 합니다."
    A = (M + M.T) * 0.5  # 대칭화

    decide = mat_type
    if mat_type == "auto":
        # 휴리스틱: trace(A^{-1}) < trace(A) 이면 A가 '정밀도'일 확률이 큼.
        # (역행렬 가능해야 하므로 try/except)
        try:
            tr = np.trace(A)
            invA = np.linalg.inv(A)
            tri = np.trace(invA)
            decide = "precision" if tri < tr else "cov"
        except np.linalg.LinAlgError:
            # 역행렬이 안 되면 공분산으로 가정
            decide = "cov"

    if decide == "precision":
        eps = 1e-8
        # 역행렬의 수치안정: (A + eps I)^{-1}
        eye = np.eye(A.shape[0])
        try:
            Sigma = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            Sigma = np.linalg.inv(A + eps * eye)
    else:
        Sigma = A

    # PSD 보정: 음수 고유값을 0으로 클램프
    w, V = np.linalg.eigh(Sigma)
    w = np.maximum(w, 0.0)
    Sigma_psd = (V * w) @ V.T

    # 아주 작은 εI 추가 (Cholesky/고유값 안정화를 위해)
    Sigma_psd += 1e-8 * np.eye(Sigma_psd.shape[0])
    return Sigma_psd


def _cov_ellipse_params(Sigma: np.ndarray, nsig: float = 1.0) -> tuple[float, float, float]:
    """
    공분산 Σ의 고유분해로 타원 파라미터(가로/세로 길이, 회전각(rad)) 계산.
    - width, height는 전체 길이(지름). nsig*표준편차 기준.
    """
    # 대칭/PSD 가정. 수치 잡음을 다시 한 번 정리
    Sigma = (Sigma + Sigma.T) * 0.5
    w, V = np.linalg.eigh(Sigma)
    # 큰 고유값이 첫 번째가 되도록 정렬
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]

    # 음수 고유값이 있으면 0으로 조정(수치 안전)
    w = np.maximum(w, 0.0)

    # nsig 표준편차 반영 (지름 = 2 * nsig * sqrt(eigval))
    width = 2.0 * nsig * np.sqrt(w[0])
    height = 2.0 * nsig * np.sqrt(w[1])

    # 첫 번째 고유벡터의 각도
    angle = np.arctan2(V[1, 0], V[0, 0])
    return float(width), float(height), float(angle)


# -------------------------------
# 공개 함수
# -------------------------------
def plot_scatter_with_gaussians(
    X: np.ndarray,
    labels: Optional[np.ndarray],
    mus: np.ndarray,
    Sigmas_or_precisions: np.ndarray,
    title: str = "",
    save_path: Optional[str] = None,
    mat_type: Literal["cov", "precision", "auto"] = "cov",
    show_legend: bool = True,
    equal_axis: bool = True,
) -> None:
    """
    2D 전용 산점도 + 가우시안 타원(1,2,3σ)을 그립니다.

    Parameters
    ----------
    X : (N, 2) ndarray
        데이터 포인트
    labels : (N,) ndarray or None
        클러스터 라벨(없으면 단일 색)
    mus : (K, 2) ndarray
        각 클러스터 평균
    Sigmas_or_precisions : (K, 2, 2) ndarray
        각 클러스터 공분산 또는 정밀도 행렬
    title : str
        플롯 제목
    save_path : str or None
        저장 경로(확장자에 따라 png/pdf 등)
    mat_type : {"cov","precision","auto"}
        입력 행렬의 종류. "auto"는 간단 휴리스틱으로 판단
    show_legend : bool
        범례 표시 여부
    equal_axis : bool
        x/y 축 비율을 동일하게 맞출지 여부
    """
    assert X.ndim == 2 and X.shape[1] == 2, "2D 데이터만 지원합니다."
    K = mus.shape[0]
    assert Sigmas_or_precisions.shape == (K, 2, 2), "행렬 크기가 (K,2,2) 이어야 합니다."

    plt.figure(figsize=(6, 6))

    if labels is None:
        plt.scatter(X[:, 0], X[:, 1], s=8, alpha=0.6)
    else:
        cmap = plt.get_cmap("tab10")
        for k in range(K):
            mask = (labels == k)
            if np.any(mask):
                plt.scatter(X[mask, 0], X[mask, 1], s=8, alpha=0.7, label=f"cluster {k}", color=cmap(k))

    # 가우시안 타원
    from matplotlib.patches import Ellipse
    ax = plt.gca()
    for k in range(K):
        Sigma_k = _to_covariance(Sigmas_or_precisions[k], mat_type=mat_type)
        for ns in (1.0, 2.0, 3.0):
            w, h, ang = _cov_ellipse_params(Sigma_k, nsig=ns)
            e = Ellipse(
                xy=mus[k],
                width=w,
                height=h,
                angle=np.degrees(ang),
                fill=False,
                linestyle="--",
                linewidth=1.2,
                alpha=0.9,
                color="k",
            )
            ax.add_patch(e)

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    if equal_axis:
        plt.axis("equal")
    if show_legend and labels is not None:
        plt.legend(loc="best")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_curve(
    values: Iterable[float],
    title: str = "",
    ylabel: str = "",
    save_path: Optional[str] = None,
    mark_decrease: bool = True,
) -> None:
    """
    단순 학습 곡선 플롯.
    - mark_decrease=True면 이전 값보다 감소한 지점을 빨간색으로 표시(수렴 검증에 도움).

    Parameters
    ----------
    values : Iterable[float]
        y 값 시퀀스
    title : str
        플롯 제목
    ylabel : str
        y축 레이블
    save_path : str or None
        저장 경로
    mark_decrease : bool
        감소 구간을 강조 표시할지 여부
    """
    vals = np.asarray(list(values), dtype=float)
    xs = np.arange(1, len(vals) + 1, dtype=int)

    plt.figure(figsize=(6, 4))
    plt.plot(xs, vals, marker="o")

    if mark_decrease and len(vals) >= 2:
        dec_idx = np.where(vals[1:] < vals[:-1] - 1e-10)[0] + 1  # 감소한 지점 인덱스(1-based x좌표)
        if dec_idx.size > 0:
            plt.scatter(xs[dec_idx], vals[dec_idx], s=36, facecolors="none", edgecolors="r", linewidths=1.5)
            # 그래프 상단에 간단 경고 문구
            ymin, ymax = plt.ylim()
            plt.ylim(ymin, ymax)  # 축 고정
            plt.text(
                0.02,
                0.98,
                "Warning: non-monotonic segments detected",
                transform=plt.gca().transAxes,
                ha="left",
                va="top",
                fontsize=9,
                color="r",
            )

    plt.title(title)
    plt.xlabel("iteration")
    plt.ylabel(ylabel if ylabel else "value")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()
