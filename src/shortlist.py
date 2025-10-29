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

# --------- settings ---------
INPUT_CSV  = Path("src/results/results_adult.csv")
OUTPUT_CSV = Path("src/results/adult_shortlist.csv")
METRIC_COL = "f1_macro"   # higher is better
TOP_N      = 20           # target number of configurations
ALPHA      = 0.05         # significance level for contrastive picks
# -----------------------------------------

def best_by_level(avg_ranks_sorted: pd.Series, meta: pd.DataFrame, colname: str) -> list[str]:
    """Pick the best-ranked config for every level of a factor."""
    picks = (
        pd.concat([avg_ranks_sorted, meta[colname]], axis=1)
          .groupby(colname, sort=False)
          .apply(lambda g: g.iloc[0:1])
    )
    return list(picks.index.get_level_values(1))

def add_contrastive(S: list[str], p_mat: pd.DataFrame, pool: list[str], n_to_add: int, alpha: float) -> None:
    """Greedily add configs most significantly different from current selection."""
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
    if sp is None:
        print("[ERROR] scikit-posthocs is required. Install: pip install scikit-posthocs", file=sys.stderr)
        sys.exit(1)

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

    # Pivot folds × configs on the metric
    pivot = df.pivot_table(index="fold_id", columns="config_id", values=METRIC_COL, aggfunc="mean")
    # Drop configs with any missing folds (ensures consistent set for ranks/post-hoc)
    pivot_n = pivot.dropna(axis=1, how="any")
    if pivot_n.shape[1] < 2 or pivot_n.shape[0] < 2:
        print("[ERROR] Need at least 2 configs and 2 folds after dropping NaNs.", file=sys.stderr)
        sys.exit(1)

    # Average ranks (higher metric -> better -> lower rank), sorted best→worst
    avg_ranks_all = pivot_n.rank(axis=1, ascending=False, method="average").mean().sort_values()

    # Metadata per surviving config
    meta = (
        df[df["config_id"].isin(pivot_n.columns)]
          .sort_values("fold_id")
          .groupby("config_id", as_index=False)
          .agg({"k":"first","metric":"first","vote":"first","retention":"first"})
          .set_index("config_id")
    )

    # Full Nemenyi p-matrix (computed on the full surviving set)
    p_all = sp.posthoc_nemenyi_friedman(pivot_n.values)
    p_all.index = pivot_n.columns
    p_all.columns = pivot_n.columns

    # ---- selection: extremes → coverage → contrast → quantile fill ----
    S: list[str] = []
    S.append(avg_ranks_all.index[0])      # best
    S.append(avg_ranks_all.index[-1])     # worst

    # coverage: ensure each factor level appears at least once
    for col in ["retention", "metric", "k", "vote"]:
        for cid in best_by_level(avg_ranks_all, meta, col):
            if cid not in S:
                S.append(cid)
            if len(S) >= TOP_N:
                break
        if len(S) >= TOP_N:
            break

    # contrastive additions: push in configs that are significantly different from S
    if len(S) < TOP_N:
        pool = list(avg_ranks_all.index)
        add_contrastive(S, p_all, pool, n_to_add=min(6, TOP_N - len(S)), alpha=ALPHA)

    # rank-quantile fill: spread across the spectrum
    if len(S) < TOP_N:
        want = TOP_N - len(S)
        idxs = np.linspace(0, len(avg_ranks_all)-1, want+2, dtype=int)[1:-1]
        for i in idxs:
            cid = avg_ranks_all.index[i]
            if cid not in S:
                S.append(cid)
            if len(S) >= TOP_N:
                break

    # If we somehow exceeded, trim deterministically by rank order
    if len(S) > TOP_N:
        seen = set()
        S = [c for c in avg_ranks_all.index if (not (c in seen or seen.add(c))) and c in S][:TOP_N]

    # Keep output order by average rank for convenience
    S = list(avg_ranks_all.loc[S].sort_values().index)

    # Write condensed CSV with all original columns, only selected config_ids
    out_df = df[df["config_id"].isin(S)].copy()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)

    # Brief summary
    print(f"[ok] wrote {len(out_df)} rows ({len(S)} configs) -> {OUTPUT_CSV}")
    for i, cid in enumerate(S, 1):
        print(f"{i:2d}. {cid}  avg_rank={avg_ranks_all.loc[cid]:.3f}")

if __name__ == "__main__":
    main()
