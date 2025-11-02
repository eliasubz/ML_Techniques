import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path

# === Load data ===
file_path = Path("src/results/pen-based-k_ibl.csv")
file_path = Path("src/results/adult-ir_k_ibl.csv")
df = pd.read_csv(file_path)

# === Parse config_id into meaningful columns ===
def parse_config(config_id):
    import re
    match = re.match(r"c-k(\d+)-([a-zA-Z]+)-([A-Z]+)", str(config_id))
    if match:
        return pd.Series({
            "k": int(match.group(1)),
            "vote": match.group(2),
            "retention": match.group(3),
        })
    else:
        return pd.Series({"k": None, "vote": None, "retention": None})

parsed = df["config_id"].apply(parse_config)
df["k"] = parsed["k"].astype(float)
df["vote"] = parsed["vote"].astype(str)
df["retention"] = parsed["retention"].astype(str)


# === Aggregate per config_id (mean across folds) ===
metrics = [
    "accuracy", "precision_macro", "recall_macro", "f1_macro",
    "fit_time_s", "predict_time_s", "total_time_s"
]

agg = (
    df.groupby(["config_id", "k", "vote", "retention"])[metrics]
    .agg(['mean', 'std'])
)
agg.columns = ['_'.join(col) for col in agg.columns]
agg = agg.reset_index()

print(agg.head())

sns.set(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (8, 5)

# === Helper function ===
def plot_grouped(df, group_col, metric, title, ylabel, filename):
    plt.figure()
    sns.barplot(data=df, x=group_col, y=f"{metric}_mean", ci=None, color="skyblue")
    plt.errorbar(
        x=range(len(df[group_col].unique())),
        y=df.groupby(group_col)[f"{metric}_mean"].mean(),
        yerr=df.groupby(group_col)[f"{metric}_std"].mean(),
        fmt="none", ecolor="gray", capsize=4
    )
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(group_col.capitalize())
    plt.tight_layout()
    plt.show()

# === Accuracy by k ===
plot_grouped(
    agg,
    "k",
    "accuracy",
    "Average Accuracy by k",
    "Accuracy",
    "accuracy_by_k.png"
)

# === Trade-off plot: F1 Macro vs Runtime (showing all hyperparameters) ===
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=agg,
    x="total_time_s_mean",
    y="f1_macro_mean",
    hue="retention",           # color → voting strategy
    style="vote",    # marker shape → retention method
    size="k",             # bubble size → neighborhood size
    alpha=0.75
)

plt.title("F1 Macro vs Runtime Trade-off for Pen Based Dataset")
plt.xlabel("Total Time (s)")
plt.ylabel("F1 Macro")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

