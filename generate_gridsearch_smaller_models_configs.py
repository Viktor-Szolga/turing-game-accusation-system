import itertools
import yaml
from pathlib import Path
import os
# Base config template
base_config = {
    "data": {"split_by": "userID"},
    "model": {
        "type": "MessageClassifier",
        "input_size": 1024,
        "hidden_sizes": [],  # will be filled dynamically
        "output_size": 2,
        "dropout": 0.75,
    },
    "training": {"batch_size": 64, "epochs": 300, "label_smoothing": 0.05},
    "validation": {"batch_size": 64},
    "optimizer": {"type": "adamW", "lr": 5e-4, "weight_decay": 5e-4},
}

# Model variants
model_variants = {
    #"Model D": [1],
    "Model E": [512]
}

# Parameter grids
learning_rates = [0.0001]
dropouts = [0]
weight_decays = [0]
smoothing_factors = [0, 0.1, 0.2]

# Output folder
out_dir = Path(os.path.join("gridsearch_smaller_models", "experiments"))
out_dir.mkdir(exist_ok=True, parents=True)

# Verzeichnis file (index)
verzeichnis_file = Path("gridsearch_smaller_models/runs.txt")
verzeichnis_lines = []

run_id = 0
seen_configs = set()  # safeguard for duplicates

def save_config(config, run_id, description):
    fname = out_dir / f"run{run_id:03d}.yaml"
    with open(fname, "w") as f:
        yaml.dump(config, f, sort_keys=False)
    verzeichnis_lines.append(f"{fname.name}: {description}")

# Generate configs (with safeguard)
for model_name, hidden_sizes in model_variants.items():
    for lr, d, wd, smoothing in itertools.product(learning_rates, dropouts, weight_decays, smoothing_factors):
        cfg = base_config.copy()
        cfg["model"] = cfg["model"].copy()
        cfg["optimizer"] = cfg["optimizer"].copy()

        cfg["model"]["hidden_sizes"] = hidden_sizes
        cfg["model"]["dropout"] = d
        cfg["optimizer"]["lr"] = lr
        cfg["optimizer"]["weight_decay"] = wd
        cfg["training"]["label_smoothing"] = smoothing

        # Convert config to YAML string (normalized) for deduplication
        cfg_str = yaml.dump(cfg, sort_keys=True)
        if cfg_str in seen_configs:
            continue  # skip duplicate
        seen_configs.add(cfg_str)

        desc = f"{model_name} | lr={lr} | dropout={d} | weight_decay={wd} | label_smoothing={smoothing}"
        save_config(cfg, run_id, desc)
        run_id += 1

# Write directory file
with open(verzeichnis_file, "a") as f:
    f.write("\n".join(verzeichnis_lines))
    f.write("\n")

print(f"Generated {run_id} unique configs")
print(f"Verzeichnis written to {verzeichnis_file}")
