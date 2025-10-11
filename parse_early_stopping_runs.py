import pandas as pd
import numpy as np
import os

root_dir = "early_stopping"

run_info = pd.read_csv(
    os.path.join(root_dir, "runs.txt"),
    sep=r"\s*\|\s*",
    header=None,
    names=["run_name", "seed", "n_layers", "n_neurons"],
    engine="python"
)

run_info["run_id"] = run_info["run_name"].str.replace(".yaml", "", regex=False)
checkpoint_info = pd.read_csv(
    os.path.join(root_dir, "checkpoints", "best_model_info.txt"),
    sep=r"\s*\|\s*",
    header=None,
    names=["name", "loss", "epochs"],
    engine="python"
)

checkpoint_info["name"] = checkpoint_info["name"].str.strip()
checkpoint_info["loss"] = pd.to_numeric(checkpoint_info["loss"], errors="coerce")

merged = pd.merge(run_info, checkpoint_info, left_on="run_id", right_on="name", how="left")

grouped = merged.groupby(["n_layers", "n_neurons"])

stats = grouped["loss"].agg(["mean", "std", "count"]).reset_index()
stats["ci95"] = 1.96 * stats["std"] / np.sqrt(stats["count"])
margin = 1.96 * stats["std"] / np.sqrt(stats["count"])

stats["ci95_lower"] = stats["mean"] - margin
stats["ci95_upper"] = stats["mean"] + margin
stats = stats.rename(columns={
    "mean": "mean_loss",
    "std": "std_loss",
    "count": "num_runs",
    "ci95": "95%_CI"
})

print("=== Architecture Loss Summary ===")
print(stats)

stats.to_csv(os.path.join(root_dir, "architecture_loss_summary.csv"), index=False)