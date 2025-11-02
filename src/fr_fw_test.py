import subprocess

# Common parameters
base_args_ir = [
    "--normalization", "mean_normalize",
    "--encoding", "one_hot_encode",
    "--missing-numeric-strategy", "median",
    "--missing-categorical-strategy", "mode",
]

base_args_fw = [
    "--normalization", "mean_normalize",
    "--encoding", "one_hot_encode",
    "--missing-numeric-strategy", "median",
    "--missing-categorical-strategy", "mode",
    "--retention-strategy", "always-retain",
]

# 1️⃣ IR model with instance reduction strategies
datasets_ir = ["adult", "pen-based"]
instance_red_strategies = ["IBL3","CNN","MCNN","enn","RENN"]

for dataset in datasets_ir:
    for strategy in instance_red_strategies:
        cmd = ["& 'C:/Users/elias/Documents/Ani/IML (Introduction to Machine Learning)/Classification with Lazy Learning and SVM/venv/Scripts/Activate.ps1'", "&&"
            "python", "src/main.py",
            "--model", "ir_k_ibl",
            "--dataset", dataset,
            "--k", "7" if dataset == "adult" else "5",
            "--distance-metric", "cosine",
            "--voting-strategy", "borda",
            "--instance-reduction-strategy", strategy
        ] + base_args_ir

        print(f"\nRunning IR model on {dataset} with {strategy}...")
        subprocess.run(cmd)

# 2️⃣ FW model with feature weighting strategies
datasets_fw = ["adult", "pen-based"]
feature_weighting_strategies = ["relieff", "information-gain"]

for dataset in datasets_fw:
    for fw_strategy in feature_weighting_strategies:
        cmd = ["& 'C:/Users/elias/Documents/Ani/IML (Introduction to Machine Learning)/Classification with Lazy Learning and SVM/venv/Scripts/Activate.ps1'", "&&",
            "python", "src/main.py",
            "--model", "fw_k_ibl",
            "--dataset", dataset,
            "--k", "7" if dataset == "adult" else "5",
            "--distance-metric", "cosine",
            "--voting-strategy", "borda",
            "--feature-weighting-strategy", fw_strategy
        ] + base_args_fw

        print(f"\nRunning FW model on {dataset} with {fw_strategy}...")
        subprocess.run(cmd)
