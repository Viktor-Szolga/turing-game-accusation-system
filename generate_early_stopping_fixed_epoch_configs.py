import itertools
import yaml
from pathlib import Path
import os

# Base config template
base_config = {
    "data": {"split_by": "gameID"},
    "model": {
        "type": "MessageClassifier",
        "input_size": 1024,
        "hidden_sizes": [],  # will be filled dynamically
        "output_size": 2,
        "dropout": 0.0,  # no dropout
    },
    "training": {"batch_size": 64, "epochs": 600, "early_stopping_patience": 10, "continue_training": False, "stop_at":None},
    "validation": {"batch_size": 64},
    "optimizer": {"type": "adamW", "lr": 0.0001, "weight_decay": 0.0},  # low LR, no weight decay
    "misc": {"seed": 42}
}

seeds = list(range(1, 11))
# Output folder
out_dir = Path(os.path.join("early_stopping_fixed_epoch_bow", "experiments"))
out_dir.mkdir(exist_ok=True, parents=True)

stop_ats = [
    70, 44, 19, 13, 10, 9, 6, 5, 5, 4,
    47, 51, 24, 17, 10, 8, 4, 3, 3, 2,
    89, 54, 39, 38, 15, 10, 6, 4, 1, 2
]*len(seeds)


# Directory file (index)
verzeichnis_file = Path(os.path.join("early_stopping_fixed_epoch_bow", "runs.txt"))
verzeichnis_lines = []

run_id = 0
seen_configs = set()  # safeguard for duplicates

def save_config(config, run_id, description):
    fname = out_dir / f"run{run_id:03d}.yaml"
    with open(fname, "w") as f:
        yaml.dump(config, f, sort_keys=False)
    verzeichnis_lines.append(f"{fname.name}: {description}")

# Generate model variants systematically
max_neurons = 512
step = 4
for seed in seeds:
    # 1 hidden layer
    h1 = 1
    while h1 < 513:
        hidden_sizes = [h1]
        cfg = base_config.copy()
        cfg["misc"]["seed"] = seed
        cfg["model"] = cfg["model"].copy()
        cfg["model"]["hidden_sizes"] = hidden_sizes
        cfg["training"]["stop_at"] = stop_ats[run_id]
        desc = f"seed:{seed} | 1-layer | {hidden_sizes[0]} neurons"
        cfg_str = yaml.dump(cfg, sort_keys=True)
        if cfg_str not in seen_configs:
            seen_configs.add(cfg_str)
            save_config(cfg, run_id, desc)
            run_id += 1
        h1*=2

    # 2 hidden layers: second layer half of first
    h1 = 1
    while h1 < 513:
        h2 = max(1, h1 // 2)
        hidden_sizes = [h1, h2]
        cfg = base_config.copy()
        cfg["model"] = cfg["model"].copy()
        cfg["model"]["hidden_sizes"] = hidden_sizes
        cfg["training"]["stop_at"] = stop_ats[run_id]
        desc = f"seed:{seed} | 2-layer | {hidden_sizes[0]}-{hidden_sizes[1]} neurons"
        cfg_str = yaml.dump(cfg, sort_keys=True)
        if cfg_str not in seen_configs:
            seen_configs.add(cfg_str)
            save_config(cfg, run_id, desc)
            run_id += 1
        h1 *= 2

    # 3 hidden layers: third layer half of second
    h1 = 1
    while h1 < 513:
        h2 = max(1, h1 // 2)
        h3 = max(1, h2 // 2)
        hidden_sizes = [h1, h2, h3]
        cfg = base_config.copy()
        cfg["model"] = cfg["model"].copy()
        cfg["model"]["hidden_sizes"] = hidden_sizes
        cfg["training"]["stop_at"] = stop_ats[run_id]
        desc = f"seed:{seed} | 3-layer | {hidden_sizes[0]}-{hidden_sizes[1]}-{hidden_sizes[2]} neurons"
        cfg_str = yaml.dump(cfg, sort_keys=True)
        if cfg_str not in seen_configs:
            seen_configs.add(cfg_str)
            save_config(cfg, run_id, desc)
            run_id += 1
        h1 *= 2

# Write directory file
with open(verzeichnis_file, "a") as f:
    f.write("\n".join(verzeichnis_lines))
    f.write("\n")

print(f"Generated {run_id} unique configs")
print(f"Verzeichnis written to {verzeichnis_file}")
