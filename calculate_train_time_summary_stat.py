import re
import os
import yaml
import pandas as pd
from omegaconf import OmegaConf

# -----------------------------
# CONFIG
# -----------------------------
LOG_FILE = "time_for_training.txt"
RUNS_DIR = "experiments/"              
pattern = r"run(\d+)\.yaml.*\[(\d+):(\d+)<.*?,\s*([\d.]+)\s*(?:it/s|s/it)\]"
records = []

with open(LOG_FILE, "r") as f:
    for line in f:
        match = re.search(pattern, line)
        if match:
            run_id = int(match.group(1))
            time_s = float(match.group(2))
            iters_per_s = float(match.group(3))
            records.append({"run_id": run_id, "time_s": time_s, "iters_per_s": iters_per_s})


df = pd.DataFrame(records)
print(f"Parsed {len(df)} runs from log.")
def classify_model(run_id):
    yaml_path = os.path.join(RUNS_DIR, f"run{run_id:03d}.yaml")
    if not os.path.exists(yaml_path):
        return "unknown"

    config = OmegaConf.load(yaml_path)
    n = len(config.model.hidden_sizes)
    #print(f"{yaml_path}: {config.model.hidden_sizes}")
    if n == 1:
        return "A"
    elif n == 2:
        return "B"
    elif n == 3:
        return "C"
    return "unknown"

df["model"] = df["run_id"].apply(classify_model)
summary = df.groupby("model").agg(
    mean_time=("time_s", "mean"),
    std_time=("time_s", "std"),
    median_time=("time_s", "median"),
    min_time=("time_s", "min"),
    max_time=("time_s", "max"),
    mean_iter_s=("iters_per_s", "mean"),
    std_iter_s=("iters_per_s", "std")
).reset_index()

print("\n=== Summary Statistics by Model ===")
print(summary.to_string(index=False))
