import pandas as pd
import numpy as np
import os

root_dir = "early_stopping_fixed_epoch_average"

# === Read run info ===
run_info = pd.read_csv(
    os.path.join(root_dir, "runs.txt"),
    sep=r"\s*\|\s*",
    header=None,
    names=["run_name", "seed", "n_layers", "n_neurons"],
    engine="python"
)

run_info["run_id"] = run_info["run_name"].str.replace(".yaml", "", regex=False)
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

# === Merge ===
merged = pd.merge(run_info, loss_info, left_on="run_id", right_on="name", how="left")

# === Find the model with the lowest loss per architecture ===
best_models = merged.loc[
    merged.groupby(["n_layers", "n_neurons"])["loss"].idxmin(),
    ["run_id", "seed", "n_layers", "n_neurons", "loss"]
]

# === Sort by loss ===
best_models = best_models.sort_values(by="loss", ascending=True)

# === Print results ===
print("\n🏆 Model with Lowest Loss per Architecture:\n")
print(best_models.to_string(index=False))
