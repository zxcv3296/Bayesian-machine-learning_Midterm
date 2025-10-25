#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scripts/quick_look.py
- G2 데이터 빠른 확인용 산점도/히스토그램
사용:
  python scripts/quick_look.py --data data/G2.txt --out runs/quick_look.png
"""

from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, default="runs/quick_look.png")
    args = ap.parse_args()

    X = np.loadtxt(args.data)
    assert X.shape[1] == 2, "2D 데이터만 지원"
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.scatter(X[:,0], X[:,1], s=6, alpha=0.6)
    plt.title("Scatter (G2)"); plt.xlabel("x1"); plt.ylabel("x2")

    plt.subplot(1,2,2)
    plt.hist(X[:,0], bins=40, alpha=0.7, label="x1")
    plt.hist(X[:,1], bins=40, alpha=0.7, label="x2")
    plt.title("Histograms"); plt.legend()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    plt.close()
    print("[INFO] Saved:", args.out)

if __name__ == "__main__":
    main()
