# Bayesian-machine-learning_Midterm


# Conda
conda env create -f env.yml
conda activate gmm-g2

# EM
python em/gmm_em.py --data data/G2.txt --K 2 --max_iter 200 --tol 1e-6 --seed 42 \
  --plot_out runs/em_K2.png --save_params runs/em_K2.npz

# Gibbs
python gibbs/gmm_gibbs.py --data data/G2.txt --K 2 --burnin 500 --iters 2000 --thin 5 --seed 42 \
  --plot_out runs/gibbs_K2.png --save_params runs/gibbs_K2.npz

# VI
python vi/gmm_vi.py --data data/G2.txt --K 2 --max_iter 500 --tol 1e-6 --seed 42 \
  --plot_out runs/vi_K2.png --save_params runs/vi_K2.npz

# 비교 노트북
jupyter notebook notebooks/99_compare_K.ipynb
