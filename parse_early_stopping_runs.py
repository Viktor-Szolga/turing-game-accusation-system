import pandas as pd
import numpy as np

# --- 1. Read run info (architectures, seeds, etc.) ---
run_info = pd.read_csv(
    "runs.txt",
    sep=r"\s*\|\s*",
    header=None,
    names=["run_name", "seed", "n_layers", "n_neurons"],
    engine="python"
)

# Clean up run names for merging (remove .yaml)
run_info["run_id"] = run_info["run_name"].str.replace(".yaml", "", regex=False)

# --- 2. Read checkpoint info (final loss per run) ---
checkpoint_info = pd.read_csv(
    "checkpoints/best_model_info.txt",
    sep=r"\s*\|\s*",
    header=None,
    names=["name", "loss", "epochs"],
    engine="python"
)

# Clean whitespace and ensure numeric loss
checkpoint_info["name"] = checkpoint_info["name"].str.strip()
checkpoint_info["loss"] = pd.to_numeric(checkpoint_info["loss"], errors="coerce")

# --- 3. Merge both on run ID ---
merged = pd.merge(run_info, checkpoint_info, left_on="run_id", right_on="name", how="left")

# --- 4. Group by architecture (n_layers + n_neurons) ---
grouped = merged.groupby(["n_layers", "n_neurons"])

# --- 5. Compute mean, std, count, and 95% confidence interval ---
stats = grouped["loss"].agg(["mean", "std", "count"]).reset_index()
stats["ci95"] = 1.96 * stats["std"] / np.sqrt(stats["count"])
margin = 1.96 * stats["std"] / np.sqrt(stats["count"])

# Lower and upper bounds
stats["ci95_lower"] = stats["mean"] - margin
stats["ci95_upper"] = stats["mean"] + margin
# --- 6. Rename columns for clarity ---
stats = stats.rename(columns={
    "mean": "mean_loss",
    "std": "std_loss",
    "count": "num_runs",
    "ci95": "95%_CI"
})

# --- 7. Display results ---
print("=== Architecture Loss Summary ===")
print(stats)

stats.to_csv("architecture_loss_summary.csv", index=False)