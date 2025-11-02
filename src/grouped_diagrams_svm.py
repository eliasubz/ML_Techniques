# === 1️⃣ Load & aggregate SVM results ===
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your CSV (adjust path if needed)
df = pd.read_csv("src/results/adult-svm.csv")

# df = pd.read_csv("src/results/pen-based-svm.csv")
# Aggregate over folds
metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "total_time_s"]
agg = (
    df.groupby(["config_id", "kernel", "c", "gamma", "degree"])[metrics]
    .mean()
    .reset_index()
    .rename(columns={
        "accuracy": "accuracy_mean",
        "precision_macro": "precision_macro_mean",
        "recall_macro": "recall_macro_mean",
        "f1_macro": "f1_macro_mean",
        "total_time_s": "total_time_s_mean"
    })
)

# === 2️⃣ F1 Macro vs Runtime Trade-off (all hyperparameters) ===
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=agg,
    x="total_time_s_mean",
    y="f1_macro_mean",
    hue="gamma",        # color = kernel type (rbf, poly, etc.)
    style="kernel",      # marker shape = degree (only relevant for poly)
    size="c",            # bubble size = regularization strength
    alpha=0.75,
)

plt.title("F1 Macro vs Runtime Trade-off for SVM on Adult Dataset")
plt.xlabel("Total Time (s)")
plt.ylabel("F1 Macro")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# === 3️⃣ Optional: Facet Grid to separate kernels ===
g = sns.FacetGrid(
    agg,
    col="kernel",
    height=5,
    sharex=False,
    sharey=False
)
g.map_dataframe(
    sns.scatterplot,
    x="total_time_s_mean",
    y="f1_macro_mean",
    size="c",
    hue="gamma",
    style="kernel",
    alpha=0.7
)
g.set_axis_labels("Total Time (s)", "F1 Macro")
g.add_legend()
g.fig.subplots_adjust(top=0.8)
g.fig.suptitle("F1 Macro vs Runtime per SVM Kernel")
plt.show()

print("✅ Done! SVM visualization now includes kernel (color/column), C (size), gamma (hue), and degree (marker).")
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=agg,
    x="total_time_s_mean",
    y="f1_macro_mean",
    hue="gamma",           # color encodes gamma
    style="kernel",        # marker shape encodes kernel type
    size="c",              # bubble size encodes regularization strength
    palette="viridis",     # better continuous colormap for numeric gamma
    sizes=(50, 400),       # make size scaling more visible
    alpha=0.85,
    edgecolor="black",
    linewidth=0.5
)

plt.title("F1 Macro vs Runtime Trade-off for SVM on Adult Dataset")
plt.xlabel("Total Time (s)")
plt.ylabel("F1 Macro")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Hyperparams")
plt.tight_layout()
plt.show()

# === F1 Macro vs Runtime Trade-off (better discrete sizing & colors) ===
plt.figure(figsize=(8, 6))

# Discrete size mapping for C
size_map = {"poly": 50, "rbf": 200}  # smaller gap between 1 and 10
size_map = {0.1: 50, 1: 150, 10: 250}  # smaller gap between 1 and 10

sns.scatterplot(
    data=agg,
    x="total_time_s_mean",
    y="f1_macro_mean",
    hue="gamma",        # color for gamma
    style="kernel",     # marker for kernel
    size="c",           # size for C (mapped manually)
    sizes=size_map,
    palette="viridis",
    alpha=0.85,
    edgecolor="black",
    linewidth=0.4,
)

plt.title("F1 Macro vs Runtime Trade-off for SVM on Pen-based Dataset")
plt.title("F1 Macro vs Runtime Trade-off for SVM on Adult Dataset")
plt.xlabel("Total Time (s)")
plt.ylabel("F1 Macro")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Hyperparams")
plt.tight_layout()
plt.show()
