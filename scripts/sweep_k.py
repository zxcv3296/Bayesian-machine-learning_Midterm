#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scripts/sweep_k.py
- EM, Gibbs, VI를 K∈{1,2,3,4,5}에 대해 일괄 실행
- 결과 요약 CSV와 플롯을 runs/ 에 저장
- NumPy/Matplotlib만 사용

사용:
  python scripts/sweep_k.py --data data/G2.txt --methods EM,Gibbs,VI --Ks 1,2,3,4,5 --seed 42 --standardize 0
"""

from __future__ import annotations
import argparse, subprocess, sys, os, json, time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def run_cmd(cmd: list[str]) -> int:
    print("[RUN]", " ".join(cmd))
    return subprocess.call(cmd)

def parse_npz_metrics(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    out = dict()
    for k in d.files:
        out[k] = d[k].item() if d[k].shape == () else d[k]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--methods", type=str, default="EM,Gibbs,VI", help="쉼표구분: EM,Gibbs,VI")
    ap.add_argument("--Ks", type=str, default="1,2,3,4,5")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--standardize", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="runs")
    ap.add_argument("--burnin", type=int, default=500)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--thin", type=int, default=5)
    ap.add_argument("--max_iter", type=int, default=500)
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    Ks = [int(x) for x in args.Ks.split(",")]
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for K in Ks:
        if "EM" in methods:
            save_npz = outdir / f"em_K{K}.npz"
            plot_png = outdir / f"em_K{K}.png"
            curve_png = outdir / f"em_ll_K{K}.png"
            cmd = [
                sys.executable, "em/gmm_em.py",
                "--data", args.data,
                "--K", str(K),
                "--seed", str(args.seed),
                "--max_iter", str(args.max_iter),
                "--tol", str(args.tol),
                "--standardize", str(args.standardize),
                "--plot_out", str(plot_png),
                "--curve_out", str(curve_png),
                "--save_params", str(save_npz)
            ]
            rc = run_cmd(cmd)
            if rc == 0 and save_npz.exists():
                m = np.load(save_npz)
                ll = float(m["ll"]) if "ll" in m else np.nan
                aic = float(m["aic"]) if "aic" in m else np.nan
                bic = float(m["bic"]) if "bic" in m else np.nan
                purity = float(m["purity"]) if "purity" in m and m["purity"].shape==() else np.nan
                ari = float(m["ari"]) if "ari" in m and m["ari"].shape==() else np.nan
                nmi = float(m["nmi"]) if "nmi" in m and m["nmi"].shape==() else np.nan
                rows.append(["EM", K, ll, aic, bic, purity, ari, nmi])

        if "Gibbs" in methods:
            save_npz = outdir / f"gibbs_K{K}.npz"
            plot_png = outdir / f"gibbs_K{K}.png"
            cmd = [
                sys.executable, "gibbs/gmm_gibbs.py",
                "--data", args.data,
                "--K", str(K),
                "--seed", str(args.seed),
                "--burnin", str(args.burnin),
                "--iters", str(args.iters),
                "--thin", str(args.thin),
                "--standardize", str(args.standardize),
                "--plot_out", str(plot_png),
                "--save_params", str(save_npz)
            ]
            rc = run_cmd(cmd)
            # Gibbs 요약: ll_pp_mean만 npz에 존재
            if rc == 0 and save_npz.exists():
                m = np.load(save_npz, allow_pickle=True)
                ll_pp = float(m["ll_pp_mean"]) if "ll_pp_mean" in m else np.nan
                # Purity/ARI/NMI는 콘솔에만 찍으므로, 여기서는 생략(원하면 Gibbs 코드에도 저장 가능)
                rows.append(["Gibbs", K, ll_pp, np.nan, np.nan, np.nan, np.nan, np.nan])

        if "VI" in methods:
            save_npz = outdir / f"vi_K{K}.npz"
            plot_png = outdir / f"vi_K{K}.png"
            curve_png = outdir / f"vi_elbo_K{K}.png"
            cmd = [
                sys.executable, "vi/gmm_vi.py",
                "--data", args.data,
                "--K", str(K),
                "--seed", str(args.seed),
                "--max_iter", str(args.max_iter),
                "--tol", str(args.tol),
                "--standardize", str(args.standardize),
                "--plot_out", str(plot_png),
                "--curve_out", str(curve_png),
                "--save_params", str(save_npz)
            ]
            rc = run_cmd(cmd)
            if rc == 0 and save_npz.exists():
                m = np.load(save_npz, allow_pickle=True)
                elbo = float(m["elbo_hist"][-1]) if "elbo_hist" in m else np.nan
                rows.append(["VI", K, elbo, np.nan, np.nan, np.nan, np.nan, np.nan])

    # CSV 저장
    import csv
    csv_path = outdir / "summary_k.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "K", "score", "AIC", "BIC", "Purity", "ARI", "NMI"])
        for r in rows:
            w.writerow(r)
    print("[INFO] Saved:", csv_path)

    # 간단 플롯(EM: BIC vs K, VI: ELBO vs K, Gibbs: LL_mean vs K)
    # 데이터 정리
    rows_arr = np.array([
        [r[0], int(r[1]), float(r[2]),
         (np.nan if r[3] is np.nan else float(r[3])),
         (np.nan if r[4] is np.nan else float(r[4]))]
        for r in rows
    ], dtype=object)

    # EM
    K_list = sorted(set(int(k) for m,k,_,_,_ in rows_arr if m=="EM"))
    if K_list:
        bic_vals = []
        for K in K_list:
            v = [float(s[4]) for s in rows_arr if s[0]=="EM" and int(s[1])==K]
            bic_vals.append(v[0] if v else np.nan)
        plt.figure(figsize=(5,4))
        plt.plot(K_list, bic_vals, marker="o")
        plt.xlabel("K"); plt.ylabel("BIC (EM)"); plt.title("EM: BIC vs K")
        plt.tight_layout()
        plt.savefig(outdir / "em_bic_vs_k.png", dpi=150); plt.close()

    # VI
    K_list = sorted(set(int(k) for m,k,_,_,_ in rows_arr if m=="VI"))
    if K_list:
        elbo_vals = []
        for K in K_list:
            v = [float(s[2]) for s in rows_arr if s[0]=="VI" and int(s[1])==K]
            elbo_vals.append(v[0] if v else np.nan)
        plt.figure(figsize=(5,4))
        plt.plot(K_list, elbo_vals, marker="o")
        plt.xlabel("K"); plt.ylabel("ELBO (VI)"); plt.title("VI: ELBO vs K")
        plt.tight_layout()
        plt.savefig(outdir / "vi_elbo_vs_k.png", dpi=150); plt.close()

    # Gibbs
    K_list = sorted(set(int(k) for m,k,_,_,_ in rows_arr if m=="Gibbs"))
    if K_list:
        ll_vals = []
        for K in K_list:
            v = [float(s[2]) for s in rows_arr if s[0]=="Gibbs" and int(s[1])==K]
            ll_vals.append(v[0] if v else np.nan)
        plt.figure(figsize=(5,4))
        plt.plot(K_list, ll_vals, marker="o")
        plt.xlabel("K"); plt.ylabel("Posterior Predictive LL (mean)"); plt.title("Gibbs: LL vs K")
        plt.tight_layout()
        plt.savefig(outdir / "gibbs_ll_vs_k.png", dpi=150); plt.close()

if __name__ == "__main__":
    main()
