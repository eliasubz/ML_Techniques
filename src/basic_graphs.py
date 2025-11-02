import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# === 1. Load and aggregate ===
file_path = Path("src/results/pen-based-k_ibl_shortlist.csv")
df = pd.read_csv(file_path)

# Average per configuration
metrics_to_avg = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "fit_time_s",
    "predict_time_s",
    "total_time_s"
]

agg_df = (
    df.groupby("config_id")[metrics_to_avg]
    .mean()
    .reset_index()
    .sort_values(by="accuracy", ascending=False)
)

print("✅ Aggregated metrics per config:")
print(agg_df.head())

# === 2. Visualization setup ===
sns.set(style="whitegrid", context="talk", palette="tab10")
plt.rcParams["figure.figsize"] = (10, 6)

# === 3. Scatter plot: Accuracy vs F1_macro ===
plt.figure()
sns.scatterplot(
    data=agg_df,
    x="accuracy",
    y="f1_macro",
    s=120,
    hue="config_id",
    legend=False,
)
plt.title("Accuracy vs F1_macro per Configuration")
plt.xlabel("Accuracy")
plt.ylabel("F1 Macro")
plt.tight_layout()
plt.show()

# === 4. Runtime vs Accuracy trade-off ===
agg_df["total_time_min"] = agg_df["total_time_s"] / 60
plt.figure()
sns.scatterplot(
    data=agg_df,
    x="total_time_min",
    y="accuracy",
    hue="config_id",
    s=120,
)
plt.title("Runtime vs Accuracy (Efficiency Trade-off)")
plt.xlabel("Total Time [min]")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.show()

# === 5. Bar plot of Accuracy per config ===
plt.figure(figsize=(12, 6))
sns.barplot(
    data=agg_df,
    x="config_id",
    y="accuracy",
    order=agg_df.sort_values("accuracy", ascending=False)["config_id"],
    palette="Blues_r"
)
plt.xticks(rotation=45, ha="right")
plt.title("Average Accuracy per Configuration")
plt.tight_layout()
plt.show()

# === 6. Correlation heatmap of averaged metrics ===
plt.figure(figsize=(10, 8))
sns.heatmap(agg_df[metrics_to_avg].corr(), annot=True, cmap="coolwarm", center=0)
plt.title("Metric Correlation Heatmap")
plt.tight_layout()
plt.show()

# === 7. Per-fold variability (robustness) ===
plt.figure(figsize=(12, 6))
sns.boxplot(
    data=df,
    x="config_id",
    y="f1_macro",
    order=agg_df.sort_values("f1_macro", ascending=False)["config_id"],
    palette="pastel"
)
plt.xticks(rotation=45, ha="right")
plt.title("F1 Macro Variability Across Folds")
plt.tight_layout()
plt.show()

# === 8. Optional: Export aggregated summary ===
agg_df.to_csv("results/adult-b_ibl_summary.csv", index=False)
print("\n📁 Saved aggregated summary to results/adult-b_ibl_summary.csv")
