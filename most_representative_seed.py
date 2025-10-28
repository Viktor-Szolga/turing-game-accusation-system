import pandas as pd
import os
import matplotlib.pyplot as plt

# ======================
# CONFIG
# ======================
root_dir = "early_stopping_fixed_epoch"  # Change if needed

# Specify the four architectures you want to compare exactly as they appear in n_layers/n_neurons
# Example: ("3-layer", "128-64-32")
arch_filter = [
    ("1-layer", "256"),
    ("2-layer", "256-128"),
    ("3-layer", "128-64-32"),
    ("1-layer", "48-24")
]

# ======================
# READ RUN INFO
# ======================
run_info = pd.read_csv(
    os.path.join(root_dir, "runs.txt"),
    sep=r"\s*\|\s*",
    header=None,
    names=["run_name", "seed", "n_layers", "n_neurons"],
    engine="python"
)

# Ensure string
for col in ["run_name", "seed", "n_layers", "n_neurons"]:
    run_info[col] = run_info[col].astype(str)

# Clean columns
run_info["run_id"] = run_info["run_name"].str.replace(".yaml", "", regex=False)
run_info["seed"] = run_info["seed"].str.replace("seed:", "", regex=False).str.strip()
run_info["n_layers"] = run_info["n_layers"].str.strip().str.lower()
run_info["n_neurons"] = run_info["n_neurons"].str.replace(" neurons", "", regex=False).str.strip()

# ======================
# READ LOSS INFO
# ======================
loss_info = pd.read_csv(
    os.path.join(root_dir, "training_log.txt"),
    sep=r"\s*\|\s*",
    header=None,
    names=["name", "epochs", "time", "loss"],
    engine="python"
)

loss_info["name"] = loss_info["name"].astype(str).str.strip()
loss_info["loss"] = loss_info["loss"].astype(str).str.extract(r"([\d\.eE+-]+)$").astype(float)

# ======================
# MERGE
# ======================
merged = pd.merge(run_info, loss_info, left_on="run_id", right_on="name", how="left")

# ======================
# FILTER ARCHITECTURES
# ======================
filtered = merged[
    merged.set_index(["n_layers", "n_neurons"]).index.isin(arch_filter)
].copy()

if filtered.empty:
    print("\n❌ No matching architectures found. Check your 'arch_filter' values and dataset formatting.")
    print("Available architectures:")
    print(merged[["n_layers", "n_neurons"]].drop_duplicates().sort_values(by=["n_layers", "n_neurons"]))
    exit()

filtered["seed"] = filtered["seed"].astype(int)

# ======================
# COMPUTE DISTANCE TO ARCH MEAN
# ======================
arch_means = filtered.groupby(["n_layers", "n_neurons"])["loss"].mean().rename("arch_mean")
filtered = filtered.merge(arch_means, on=["n_layers", "n_neurons"], how="left")
filtered["dist_to_mean"] = (filtered["loss"] - filtered["arch_mean"]).abs()

# ======================
# COMPUTE AVG DEVIATION PER SEED
# ======================
seed_deviation = (
    filtered.groupby("seed")["dist_to_mean"]
    .mean()
    .reset_index()
    .rename(columns={"dist_to_mean": "avg_deviation"})
    .sort_values("avg_deviation")
)

print("\n📉 Seedwise Mean Deviation from Architecture Mean:")
print(seed_deviation.to_string(index=False))

best_seed = seed_deviation.iloc[0]["seed"]
print(f"\n🎯 Most representative seed: {best_seed}")