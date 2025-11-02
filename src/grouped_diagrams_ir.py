import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import re

# === Load data ===
file_path = Path("src/results/adult-ir_k_ibl.csv")
file_path = Path("src/results/pen-based-ir_k_ibl.csv")
df = pd.read_csv(file_path)

# === Clean columns ===
df.columns = df.columns.str.strip().str.lower()

# === Aggregate per instance reduction method (mean/std over folds) ===
metrics = [
    "accuracy", "precision_macro", "recall_macro", "f1_macro",
    "precision_weighted", "recall_weighted", "f1_weighted",
    "fit_time_s", "predict_time_s", "total_time_s",
    "percentage_memory_reduction"
]

agg = (
    df.groupby(["instance_reduction_method"])[metrics]
    .agg(['mean', 'std'])
)
agg.columns = ['_'.join(col) for col in agg.columns]
agg = agg.reset_index()

print("\n=== Summary per IR method ===")
print(agg[["instance_reduction_method", "accuracy_mean", "f1_macro_mean", "total_time_s_mean"]])

sns.set(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (8, 5)

# === Accuracy and F1 Macro comparison ===
plt.figure(figsize=(8, 5))
sns.barplot(
    data=agg,
    x="instance_reduction_method",
    y="accuracy_mean",
    hue="instance_reduction_method",
    dodge=False,
    palette="Set2"
)
plt.title("Accuracy by Instance Reduction Method")
plt.ylabel("Accuracy")
plt.xlabel("Instance Reduction Method")
plt.legend([], [], frameon=False)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(
    data=agg,
    x="instance_reduction_method",
    y="f1_macro_mean",
    hue="instance_reduction_method",
    dodge=False,
    palette="Set2"
)
plt.title("F1 Macro by Instance Reduction Method")
plt.ylabel("F1 Macro")
plt.xlabel("Instance Reduction Method")
plt.legend([], [], frameon=False)
plt.tight_layout()
plt.show()

# === Runtime Comparison (Total Time) ===
plt.figure(figsize=(8, 5))
sns.barplot(
    data=agg,
    x="instance_reduction_method",
    y="total_time_s_mean",
    hue="instance_reduction_method",
    dodge=False,
    palette="Set1"
)
plt.title("Average Total Runtime per Instance Reduction Method")
plt.ylabel("Total Time (s)")
plt.xlabel("Instance Reduction Method")
plt.legend([], [], frameon=False)
plt.tight_layout()
plt.show()

# === Memory Efficiency ===
plt.figure(figsize=(8, 5))
sns.barplot(
    data=agg,
    x="instance_reduction_method",
    y="percentage_memory_reduction_mean",
    hue="instance_reduction_method",
    dodge=False,
    palette="coolwarm"
)
plt.title("Average Memory Reduction per IR Method")
plt.ylabel("Memory Reduction (%)")
plt.xlabel("Instance Reduction Method")
plt.legend([], [], frameon=False)
plt.tight_layout()
plt.show()

# === Trade-off: F1 vs Runtime ===
plt.figure(figsize=(7, 5))
sns.scatterplot(
    data=agg,
    x="total_time_s_mean",
    y="f1_macro_mean",
    hue="instance_reduction_method",
    style="instance_reduction_method",
    s=150,
    palette="tab10",
)
plt.title("F1 vs Runtime Trade-off by Instance Reduction Method")
plt.xlabel("Total Time (s)")
plt.ylabel("F1 Macro")
plt.tight_layout()
plt.show()

# === Summary table ===
print("\n=== PERFORMANCE SUMMARY (mean ± std) ===")
summary_cols = [
    "accuracy_mean", "f1_macro_mean", "total_time_s_mean", "percentage_memory_reduction_mean"
]
print(agg[["instance_reduction_method"] + summary_cols])
