Gaussian Mixture Model (GMM) - EM / Gibbs / Variational Inference
==================================================================

This repository contains from-scratch NumPy implementations of Gaussian Mixture Model (GMM) learning using:
1. Expectation-Maximization (EM)
2. Gibbs Sampling
3. Variational Inference (VI)

------------------------------------------------------------------
1. Environment Setup
------------------------------------------------------------------
Required Packages:
- Python >= 3.10
- NumPy
- Matplotlib

Install via conda:
    conda env create -f env.yml
    conda activate gmm-g2

------------------------------------------------------------------
2. Folder Structure
------------------------------------------------------------------
project/
├─ data/
│  └─ G2.txt
├─ common/
│  ├─ io_utils.py
│  ├─ metrics.py
│  ├─ utils.py
│  └─ viz.py
├─ em/
│  └─ gmm_em.py
├─ gibbs/
│  └─ gmm_gibbs.py
├─ vi/
│  └─ gmm_vi.py
├─ runs/
│  └─ output images and npz results
└─ notebooks/
   ├─ 00_quick_look.ipynb
   └─ 99_compare_K.ipynb

------------------------------------------------------------------
3. Execution Examples
------------------------------------------------------------------

EM Algorithm:
--------------
python em/gmm_em.py --data data/G2.txt --K 2 \
  --max_iter 200 --tol 1e-6 --seed 42 \
  --plot_out runs/em_K2.png --curve_out runs/em_ll_K2.png \
  --save_params runs/em_K2.npz --standardize 0

Gibbs Sampling:
---------------
python gibbs/gmm_gibbs.py --data data/G2.txt --K 2 \
  --burnin 500 --iters 2000 --thin 5 --seed 42 --standardize 0 \
  --plot_out runs/gibbs_K2.png --save_params runs/gibbs_K2.npz

Variational Inference:
----------------------
python vi/gmm_vi.py --data data/G2.txt --K 2 \
  --max_iter 500 --tol 1e-6 --seed 42 --standardize 0 \
  --plot_out runs/vi_K2.png --curve_out runs/vi_elbo_K2.png \
  --save_params runs/vi_K2.npz

------------------------------------------------------------------
4. Results
------------------------------------------------------------------
K values tested: 1~4  
Metrics (for K=2 example):
- EM  : LL=-22812.15  Purity=0.917  ARI=0.695  NMI=0.587
- Gibbs: LL_mean=-28610.33  Purity=0.917  ARI=0.695  NMI=0.587
- VI  : ELBO=41068.09  Purity=0.917  ARI=0.695  NMI=0.587

Visualizations saved in `runs/`:
- em_K*.png, gibbs_K*.png, vi_K*.png  → Cluster plots
- em_ll_K*.png, vi_elbo_K*.png        → Convergence curves

------------------------------------------------------------------
5. Notes
------------------------------------------------------------------
- All implementations use only NumPy (no scikit-learn or scipy).
- Random seeds are fixed for reproducibility.
- Results verified on dataset G2 (two Gaussian clusters).
- Figures show convergence and posterior separation quality for each method.

------------------------------------------------------------------
6. Quick Git Commands
------------------------------------------------------------------
git add README.txt
git commit -m "Add README.txt"
git push

------------------------------------------------------------------
Author
------------------------------------------------------------------
건호 (Ewha Womans University)
2025
