import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ========== SETTINGS ==========
# Choose what to color/group by in the scatter: "metric" or "retention"
COLOR_BY = "metric"   # <- switch to "retention" if you prefer


# file_path = Path("src/results/pen-based-k_ibl.csv")
file_path = Path("src/results/adult-k_ibl.csv")
# ==============================

# === Load ===
df = pd.read_csv(file_path)

# === Basic sanity checks / typing ===
needed_cols = {
    "config_id","metric","k","vote","retention","fold_id","num_folds",
    "n_train","n_test","fit_time_s","predict_time_s","total_time_s",
    "accuracy","precision_macro","recall_macro","f1_macro",
    "precision_weighted","recall_weighted","f1_weighted","confusion_matrix_json"
}
missing = [c for c in needed_cols if c not in df.columns]
if missing:
    raise ValueError(f"CSV missing columns: {missing}")

if COLOR_BY not in {"metric","retention"}:
    raise ValueError("COLOR_BY must be 'metric' or 'retention'")

df["k"] = pd.to_numeric(df["k"], errors="coerce")
df["vote"] = df["vote"].astype(str)
df["metric"] = df["metric"].astype(str)

# === Retention mapping to short codes ===
retention_map = {
    "RetentionPolicy.ALWAYS_RETAIN": "AR",
    "RetentionPolicy.NEVER_RETAIN": "NR",
    "RetentionPolicy.DIFFERENT_CLASS_RETENTION": "DC",
    "RetentionPolicy.DD_RETENTION": "DD",
}
df["retention"] = df["retention"].astype(str).str.strip().replace(retention_map)
RETENTION_ORDER = ["AR", "NR", "DC", "DD"]  # optional consistent order

# === Aggregate per config (mean/std across folds) ===
metrics = [
    "accuracy", "precision_macro", "recall_macro", "f1_macro",
    "fit_time_s", "predict_time_s", "total_time_s"
]

agg = (
    df.groupby(["config_id", "k", "vote", COLOR_BY])[metrics]
      .agg(['mean', 'std'])
      .reset_index()
)
agg.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in agg.columns]
rename_map = {
    "config_id_": "config_id",
    f"{COLOR_BY}_": COLOR_BY,
    "k_": "k",
    "vote_": "vote"
}
agg = agg.rename(columns={k: v for k, v in rename_map.items() if k in agg.columns})

print(agg.head())

# === Viz style ===
sns.set(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (8, 5)

# === Helper: bar by group ===
def plot_grouped(df_in, group_col, metric_name, title, ylabel):
    order = None
    if group_col == "retention":
        order = RETENTION_ORDER
    plt.figure()
    sns.barplot(data=df_in, x=group_col, y=f"{metric_name}_mean", ci=None, order=order)
    # error bars (mean of group means/stds)
    x_categories = order if order is not None else list(df_in[group_col].unique())
    group_means = df_in.groupby(group_col)[f"{metric_name}_mean"].mean().reindex(x_categories)
    group_stds  = df_in.groupby(group_col)[f"{metric_name}_std"].mean().reindex(x_categories)
    plt.errorbar(
        x=range(len(x_categories)),
        y=group_means.values,
        yerr=group_stds.values,
        fmt="none", ecolor="gray", capsize=4
    )
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(group_col.capitalize())
    plt.tight_layout()
    plt.show()

# === 1) Accuracy by k ===
plot_grouped(
    agg,
    "k",
    "accuracy",
    "Average Accuracy by k",
    "Accuracy"
)

# === 2) Trade-off: F1 Macro vs Runtime (color by COLOR_BY, style by vote, size by k) ===
plt.figure(figsize=(8, 6))
hue_order = RETENTION_ORDER if COLOR_BY == "retention" else None
sns.scatterplot(
    data=agg,
    x="total_time_s_mean",
    y="f1_macro_mean",
    hue=COLOR_BY,          
    hue_order=hue_order,   
    style="vote",
    size="k",
    alpha=0.75
)

plt.title(f"F1 Macro vs Runtime Trade-off ({file_path.stem})")
plt.xlabel("Total Time (s)")
plt.ylabel("F1 Macro")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

