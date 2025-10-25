#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
io_utils.py
- G2.txt 로드(헤더/설명/괄호/문자 포함 줄 전부 무시)
- 공백/쉼표 구분자 지원
- Z-score 표준화
"""

from __future__ import annotations
import re
import numpy as np
from pathlib import Path
from typing import Tuple, List

# 숫자 토큰 검출
_NUM_TOKEN = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
# 토큰 분리(쉼표/공백 혼용)
_SPLIT = re.compile(r"[,\s]+")

def _looks_like_data_line(line: str) -> bool:
    """
    '순수 데이터 행'인지 판정:
    - 알파벳/콜론/대괄호/괄호 등이 있으면 스킵
    - 빈 줄/주석(#,%,;) 스킵
    """
    if not line:
        return False
    if line[0] in "#%;":
        return False
    # 알파벳/괄호류/콜론 등 포함 시 데이터 아님
    if re.search(r"[A-Za-z\[\]\(\)\:\=]", line):
        return False
    # 숫자 하나도 없으면 데이터 아님
    if not _NUM_TOKEN.search(line):
        return False
    return True

def _line_to_numeric_tokens(line: str) -> List[float]:
    parts = [p for p in _SPLIT.split(line.strip()) if p]
    vals: List[float] = []
    for p in parts:
        try:
            vals.append(float(p))
        except ValueError:
            # 숫자 아님 → 무시
            pass
    return vals

def load_txt(path: str | Path, dtype=float) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    # 1차 필터: 데이터처럼 보이는 줄만 수집
    candidates: List[List[float]] = []
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not _looks_like_data_line(line):
                continue
            vals = _line_to_numeric_tokens(line)
            if len(vals) == 0:
                continue
            candidates.append(vals)

    if len(candidates) == 0:
        raise ValueError(f"No numeric data found in: {p}")

    # 2차 정제: 가장 흔한 컬럼 수(모달 값)에 맞는 행만 사용
    lengths = [len(r) for r in candidates]
    # 모달 컬럼 수
    uniq, counts = np.unique(lengths, return_counts=True)
    modal_cols = int(uniq[np.argmax(counts)])

    # 컬럼 수가 modal_cols인 행만 모음
    rows = [r for r in candidates if len(r) == modal_cols]

    # 데이터가 너무 적으면 실패 처리
    if len(rows) < 10:
        raise ValueError(
            f"Too few data rows with {modal_cols} columns in {p}. Found {len(rows)} rows."
        )

    arr = np.asarray(rows, dtype=dtype)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr

def load_g2(path: str | Path) -> Tuple[np.ndarray, np.ndarray | None]:
    """
    G2.txt 로드.
    - 표준 G2라면 (2048, 2). 이 경우 정답 라벨을 0/1로 자동 생성.
    - 행 수가 다르면 라벨은 None.
    """
    X = load_txt(path).astype(float)
    N = X.shape[0]
    y_true = None
    if N == 2048:
        y_true = np.zeros(N, dtype=int)
        y_true[1024:] = 1
    return X, y_true

def zscore(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    return (X - mu) / (sd + eps)

if __name__ == "__main__":
    # 사용 예: python common/io_utils.py data/G2.txt
    import sys
    if len(sys.argv) > 1:
        X, y = load_g2(sys.argv[1])
        print("X shape:", X.shape)
        if y is None:
            print("y_true: None")
        else:
            print("y_true:", y.shape, np.bincount(y))
        print("head:\n", X[:5])
    else:
        print("Usage: python common/io_utils.py data/G2.txt")
