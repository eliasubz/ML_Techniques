# shortlist.py — condense a 72-config per-fold CSV to a diverse subset (~20 configs)

from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

try:
    import scikit_posthocs as sp
except Exception:
    sp = None

# Settings
INPUT_CSV  = Path("src/results/pen-based-k_ibl.csv")
OUTPUT_CSV = Path("src/results/pen-based-k_ibl_shortlist.csv")
METRIC_COL = "f1_macro"   # higher is better
TOP_N      = 12          # target number of configuration
ALPHA      = 0.05         # significance level for contrastive picks

# picks the best-ranked config for every level of a factor
def best_by_level(avg_ranks_sorted: pd.Series, meta: pd.DataFrame, colname: str) -> list[str]:
    picks = (
        pd.concat([avg_ranks_sorted, meta[colname]], axis=1)
          .groupby(colname, sort=False)
          .apply(lambda g: g.iloc[0:1], include_groups=False)
    )
    return list(picks.index.get_level_values(1))

# adds configs most significantly different from current selection
def add_contrastive(S: list[str], p_mat: pd.DataFrame, pool: list[str], n_to_add: int, alpha: float) -> None:
    for _ in range(max(0, n_to_add)):
        best_cand, best_score = None, None
        for c in pool:
            if c in S:
                continue
            p_vs_S = p_mat.loc[c, S]
            sig_count = int((p_vs_S.values < alpha).sum())
            min_p = float(p_vs_S.min())
            score = (sig_count, -min_p)  # more significant pairs, then smaller min p
            if (best_score is None) or (score > best_score):
                best_score, best_cand = score, c
        if best_cand is None:
            break
        S.append(best_cand)

def main():

    if not INPUT_CSV.exists():
        print(f"[ERROR] Input not found: {INPUT_CSV}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)

    if "config_id" not in df.columns:
        print("[ERROR] 'config_id' column is missing in the CSV.", file=sys.stderr)
        sys.exit(1)
    if METRIC_COL not in df.columns:
        print(f"[ERROR] Metric column '{METRIC_COL}' not found.", file=sys.stderr)
        sys.exit(1)

    # pivot table (folds × configs on the metric)
    pivot = df.pivot_table(index="fold_id", columns="config_id", values=METRIC_COL, aggfunc="mean")

    # Average ranks (higher metric -> better -> lower rank), sorted best→worst
    avg_ranks_all = pivot.rank(axis=1, ascending=False, method="average").mean().sort_values()

    # Metadata per surviving config
    meta = (
        df[df["config_id"].isin(pivot.columns)]
          .sort_values("fold_id")
          .groupby("config_id", as_index=False)
          .agg({"k":"first","metric":"first","vote":"first","retention":"first"})
          .set_index("config_id")
    )

    # Full Nemenyi p-matrix (computed on the full surviving set)
    p_all = sp.posthoc_nemenyi_friedman(pivot.values)
    p_all.index = pivot.columns
    p_all.columns = pivot.columns

    # extremes → coverage → contrast 
    S: list[str] = []
    S.append(avg_ranks_all.index[0])      # best
    S.append(avg_ranks_all.index[-1])     # worst

    # ensure each factor level appears at least once (cover)
    for col in ["retention", "metric", "k", "vote"]:
        for cid in best_by_level(avg_ranks_all, meta, col):
            if cid not in S:
                S.append(cid)
            if len(S) >= TOP_N:
                break
        if len(S) >= TOP_N:
            break

    # contrastive configs that are significantly different from S
    if len(S) < TOP_N:
        pool = list(avg_ranks_all.index)
        add_contrastive(S, p_all, pool, n_to_add=min(6, TOP_N - len(S)), alpha=ALPHA)

    # Keep output order by average rank for convenience
    S = list(avg_ranks_all.loc[S].sort_values().index)

    # Write condensed CSV with all original columns, only selected config_ids
    out_df = df[df["config_id"].isin(S)].copy()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[ok] wrote {len(out_df)} rows ({len(S)} configs) -> {OUTPUT_CSV}")
    for i, cid in enumerate(S, 1):
        print(f"{i:2d}. {cid}  avg_rank={avg_ranks_all.loc[cid]:.3f}")

if __name__ == "__main__":
    main()
