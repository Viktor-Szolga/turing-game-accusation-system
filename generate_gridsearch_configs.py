import itertools
import yaml
from pathlib import Path

# Base config template
base_config = {
    "data": {"split_by": "gameID"},
    "model": {
        "type": "MessageClassifier",
        "input_size": 1024,
        "hidden_sizes": [],  # will be filled dynamically
        "output_size": 2,
        "dropout": 0.75,
    },
    "training": {"batch_size": 64, "epochs": 600},
    "validation": {"batch_size": 64},
    "optimizer": {"type": "adamW", "lr": 5e-4, "weight_decay": 5e-4},
}

# Model variants
model_variants = {
    "Model A": [32],
    "Model B": [48, 24],
    "Model C": [512, 128, 32],
}

# Parameter grids
learning_rates = [0.0001, 0.0005, 0.001]
dropouts = [0.55]
weight_decays = [0]

# Output folder
out_dir = Path("experiments")
out_dir.mkdir(exist_ok=True)

# Verzeichnis file (index)
verzeichnis_file = Path("runs.txt")
verzeichnis_lines = []

run_id = 288
seen_configs = set()  # safeguard for duplicates

def save_config(config, run_id, description):
    fname = out_dir / f"run{run_id:03d}.yaml"
    with open(fname, "w") as f:
        yaml.dump(config, f, sort_keys=False)
    verzeichnis_lines.append(f"{fname.name}: {description}")

# Generate configs (with safeguard)
for model_name, hidden_sizes in model_variants.items():
    for lr, d, wd in itertools.product(learning_rates, dropouts, weight_decays):
        cfg = base_config.copy()
        cfg["model"] = cfg["model"].copy()
        cfg["optimizer"] = cfg["optimizer"].copy()

        cfg["model"]["hidden_sizes"] = hidden_sizes
        cfg["model"]["dropout"] = d
        cfg["optimizer"]["lr"] = lr
        cfg["optimizer"]["weight_decay"] = wd

        # Convert config to YAML string (normalized) for deduplication
        cfg_str = yaml.dump(cfg, sort_keys=True)
        if cfg_str in seen_configs:
            continue  # skip duplicate
        seen_configs.add(cfg_str)

        desc = f"{model_name} | lr={lr} | dropout={d} | weight_decay={wd}"
        save_config(cfg, run_id, desc)
        run_id += 1

# Write directory file
with open(verzeichnis_file, "a") as f:
    f.write("\n".join(verzeichnis_lines))
    f.write("\n")

print(f"Generated {run_id} unique configs")
print(f"Verzeichnis written to {verzeichnis_file}")
