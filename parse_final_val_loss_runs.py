import pandas as pd
import numpy as np
import os

root_dir = "early_stopping_fixed_num_steps"
root_dir = "early_stopping_fixed_epoch"

# === Read run info ===
run_info = pd.read_csv(
    os.path.join(root_dir, "runs.txt"),
    sep=r"\s*\|\s*",
    header=None,
    names=["run_name", "seed", "n_layers", "n_neurons"],
    engine="python"
)

run_info["run_id"] = run_info["run_name"].str.replace(".yaml", "", regex=False)

# keep as strings for plotting compatibility
run_info["seed"] = run_info["seed"].str.replace("seed:", "", regex=False)
run_info["n_layers"] = run_info["n_layers"].str.strip()
run_info["n_neurons"] = run_info["n_neurons"].str.replace(" neurons", "", regex=False).str.strip()

# === Read loss info ===
loss_info = pd.read_csv(
    os.path.join(root_dir, "training_log.txt"),
    sep=r"\s*\|\s*",
    header=None,
    names=["name", "epochs", "time", "loss"],
    engine="python"
)

loss_info["name"] = loss_info["name"].str.strip()
loss_info["loss"] = loss_info["loss"].str.extract(r"([\d\.eE+-]+)$").astype(float)

# === Merge and compute stats ===
merged = pd.merge(run_info, loss_info, left_on="run_id", right_on="name", how="left")

grouped = merged.groupby(["n_layers", "n_neurons"], as_index=False)
stats = grouped["loss"].agg(["mean", "std", "count"]).reset_index()

# Compute 95% CI
stats["ci95"] = 1.96 * stats["std"] / np.sqrt(stats["count"])
stats["ci95_lower"] = stats["mean"] - stats["ci95"]
stats["ci95_upper"] = stats["mean"] + stats["ci95"]

stats = stats.rename(columns={
    "mean": "mean_loss",
    "std": "std_loss",
    "count": "num_runs",
    "ci95": "95%_CI"
})

# === Save summary ===
stats.to_csv(os.path.join(root_dir, "architecture_loss_summary.csv"), index=False)

print("✅ architecture_loss_summary.csv created successfully!")
print(stats.head())
