import re
import os
import pandas as pd
from omegaconf import OmegaConf

root_dir = "gridsearch"
LOG_FILE = os.path.join(root_dir, "training_log.txt")
RUNS_DIR = os.path.join(root_dir, "experiments")              

pattern = r"run(\d+)\s*\|\s*epochs:\s*(\d+)\s*\|\s*time \(s\):\s*([\d.]+)\s*\|\s*final validation loss:\s*([\d.]+)"

records = []
with open(LOG_FILE, "r") as f:
    for line in f:
        match = re.search(pattern, line)
        if match:
            run_id = int(match.group(1))
            epochs = int(match.group(2))
            time_s = float(match.group(3))
            val_loss = float(match.group(4))
            records.append({
                "run_id": run_id,
                "epochs": epochs,
                "time_s": time_s,
                "final_val_loss": val_loss
            })

def classify_model(run_id):
    yaml_path = os.path.join(RUNS_DIR, f"run{run_id:03d}.yaml")
    if not os.path.exists(yaml_path):
        return "unknown"

    config = OmegaConf.load(yaml_path)
    if config.model.type == "LinearClassifier":
        return "unknown"
    n = len(config.model.hidden_sizes)
    return n
    if n == 1:
        return "A"
    elif n == 2:
        return "B"
    elif n == 3:
        return "C"
    return "unknown"

df = pd.DataFrame(records)
df["n_hidden"] = df["run_id"].apply(classify_model)

summary = df.groupby("n_hidden").agg(
    mean_time=("time_s", "mean"),
    std_time=("time_s", "std"),
    median_time=("time_s", "median"),
    min_time=("time_s", "min"),
    max_time=("time_s", "max"),
).reset_index()

print("\n=== Summary Statistics by Model ===")
print(summary.to_string(index=False))
