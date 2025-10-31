from pathlib import Path
import pandas as pd
import numpy as np
import scikit_posthocs as sp
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare

CSV_PATH = Path("src/results/adult_shortlist.csv")  # <- change if different

if not CSV_PATH.exists():
    raise FileNotFoundError(f"Per-fold CSV not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

# Build the pivot matrix
pivot = df.pivot_table(index="fold_id", columns="config_id", values="f1_macro", aggfunc="mean")

# rank within each fold
ranks = pivot.rank(axis=1, ascending=False, method="average")  # ties get average rank

# aggregate across folds
avg_rank = ranks.mean(axis=0)                 # average rank per config
mean_f1  = pivot.mean(axis=0)                 # mean F1 across folds 
std_f1   = pivot.std(axis=0, ddof=1)          # spread across folds
rank1_ct = (ranks == 1.0).sum(axis=0)         # how many times each config was rank 1

# create rank table with mean std
rank_table = (
    pd.DataFrame({
        "avg_rank": avg_rank,
        "mean_f1":  mean_f1,
        "std_f1":   std_f1,
        "rank1_count": rank1_ct,
    })
    .sort_values(["avg_rank","mean_f1"], ascending=[True, False])
)

pivot_f = pivot.dropna(axis=1, how="any")
alg_vectors = [pivot_f[c].values for c in pivot_f.columns]

stat, pval = friedmanchisquare(*alg_vectors)
print(f"Friedman test = {stat:.4f}, p = {pval:.6g}  (df = {pivot_f.shape[1]-1})")

# Nemenyi posthoc on Friedman ranks
nemenyi_p = sp.posthoc_nemenyi_friedman(pivot.values)

# Add config labels to the matrix 
nemenyi_p.index = pivot.columns
nemenyi_p.columns = pivot.columns

# 1) Average ranks for all surviving configs (higher metric -> lower rank is better)
avg_ranks_all = pivot.rank(axis=1, ascending=False, method="average").mean()

# 2) Sort by average rank for display (best → worst)
order = avg_ranks_all.sort_values().index
avg_ranks = avg_ranks_all.loc[order]

# 3) Align the precomputed Nemenyi p-matrix to the same order
nemenyi_ordered = nemenyi_p.loc[order, order]

# 4) Plot
width = max(20, 0.45 * len(order))  # widen for many labels
plt.figure(figsize=(width, 2.8), dpi=180)
sp.critical_difference_diagram(avg_ranks, nemenyi_ordered)
plt.title(f"Critical Difference Diagram — all configs (n={len(order)})")
plt.tight_layout()
plt.show()