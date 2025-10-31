from __future__ import annotations
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

# ========= helpers =========

def require_cols(df: pd.DataFrame, required: set[str], label: str) -> None:
    """
    description (this function ensures a dataframe has required columns, else raises)
    @Input: df (pd.DataFrame), required (set of column names), label (file label)
    @Output: None (raises ValueError if missing)
    """
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[{label}] missing required columns: {sorted(missing)}")

def build_pivot(df_all: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    description (this function builds a fold x config pivot on the chosen metric)
    @Input: df_all (concatenated data), metric (str)
    @Output: pivot (pd.DataFrame: rows=fold_id, cols=config_id)
    """
    pivot = df_all.pivot_table(index="fold_id", columns="config_id", values=metric, aggfunc="mean")
    pivot = pivot.dropna(axis=1, how="any")
    return pivot

def rank_table_from_pivot(pivot: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    description (this function computes rank summary across folds)
    @Input: pivot (fold x config), metric (str)
    @Output: table with avg_rank, mean_metric, std_metric, rank1_count
    """
    ranks = pivot.rank(axis=1, ascending=False, method="average")
    avg_rank = ranks.mean(axis=0)
    mean_metric = pivot.mean(axis=0)
    std_metric = pivot.std(axis=0, ddof=1)
    rank1_ct = (ranks == 1.0).sum(axis=0)

    tbl = (
        pd.DataFrame({
            "avg_rank": avg_rank,
            f"mean_{metric}": mean_metric,
            f"std_{metric}": std_metric,
            "rank1_count": rank1_ct,
        })
        .sort_values(["avg_rank", f"mean_{metric}"], ascending=[True, False])
    )
    return tbl

def run_friedman_nemenyi(pivot: pd.DataFrame):
    """
    description (this function runs Friedman test and Nemenyi posthoc)
    @Input: pivot (fold x config)
    @Output: (friedman_stat, friedman_p, avg_ranks (Series), nemenyi_p (DataFrame))
    """
    alg_vectors = [pivot[c].values for c in pivot.columns]
    stat, pval = friedmanchisquare(*alg_vectors)

    avg_ranks = pivot.rank(axis=1, ascending=False, method="average").mean()
    nemenyi_p = sp.posthoc_nemenyi_friedman(pivot.values)
    nemenyi_p.index = pivot.columns
    nemenyi_p.columns = pivot.columns
    return stat, pval, avg_ranks, nemenyi_p

def plot_cd(avg_ranks: pd.Series, nemenyi_p: pd.DataFrame, title: str,
            save_path: str | Path | None = None, dpi: int = 300) -> None:
    """
    description (this function plots a critical difference diagram and optionally saves it)
    @Input: avg_ranks (Series), nemenyi_p (DataFrame), title (str), save_path (str|Path|None), dpi (int)
    @Output: None (saves and shows the plot)
    """
    order = avg_ranks.sort_values().index
    avg_ranks_ord = avg_ranks.loc[order]
    nemenyi_ord = nemenyi_p.loc[order, order]

    width = max(20, 0.45 * len(order))
    plt.figure(figsize=(width, 2.8), dpi=180)
    sp.critical_difference_diagram(avg_ranks_ord, nemenyi_ord)
    plt.title(title)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"saved plot -> {save_path}")

    plt.show()

# ========= main =========

def main():
    parser = argparse.ArgumentParser(description="Minimal Friedman/Nemenyi across multiple per-fold CSVs.")
    parser.add_argument("csvs", nargs="+", type=Path, help="Input CSV files (must have config_id, fold_id, metric).")
    parser.add_argument("--metric", default="f1_macro", help="Metric column to analyze (default: f1_macro)")
    parser.add_argument("--export-csv", type=Path, help="Path prefix to export rank_table and nemenyi_p as CSVs")
    parser.add_argument("--savefig", type=Path, help="Path to save the CD diagram image (e.g., plots/cd.png)")
    args = parser.parse_args()

    metric = args.metric
    required = {"config_id", "fold_id", metric}

    frames = []
    for p in args.csvs:
        df = pd.read_csv(p)
        require_cols(df, required, p.name)
        frames.append(df[["config_id", "fold_id", metric]].copy())

    df_all = pd.concat(frames, ignore_index=True)

    pivot = build_pivot(df_all, metric)
    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        raise RuntimeError(f"Not enough data for Friedman/Nemenyi (folds={pivot.shape[0]}, configs={pivot.shape[1]}).")

    ranks = rank_table_from_pivot(pivot, metric)
    print("\n=== Rank table ===")
    print(ranks.to_string())

    stat, pval, avg_ranks, nemenyi_p = run_friedman_nemenyi(pivot)
    print(f"\nFriedman test = {stat:.4f}, p = {pval:.6g}  (df = {pivot.shape[1]-1}) "
          f"[folds={pivot.shape[0]}, configs={pivot.shape[1]}]")

    if args.export_csv:
        ranks.to_csv(args.export_csv.with_suffix(".rank_table.csv"))
        nemenyi_p.to_csv(args.export_csv.with_suffix(".nemenyi_p.csv"))
        print(f"exported: {args.export_csv.with_suffix('.rank_table.csv')} and {args.export_csv.with_suffix('.nemenyi_p.csv')}")

    plot_cd(avg_ranks, nemenyi_p,
            title=f"Critical Difference Diagram",
            save_path=args.savefig,
            dpi=300)

if __name__ == "__main__":
    main()
