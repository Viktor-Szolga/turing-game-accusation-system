import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import torch
import random
from sklearn.metrics import brier_score_loss
from omegaconf import OmegaConf

from src.utils import initialize_model
from src.trainer import Trainer
from src.models import TemperatureScaledMessageClassifier
import src.utils as utils

# ---------------- CONFIG ---------------- #
N_BINS = 10
root_dir = "temperature_scaling_cv"
run_names = ["temperature_scaled_run127.pth"]   # example: saved model dict with T_final
labels = ["Temperature-scaled model"]
is_checkpoint = False
# ---------------------------------------- #

def get_bot_probability(h_i, b_i):
    """Convert logits (human, bot) to bot probability."""
    return np.exp(b_i) / (np.exp(h_i) + np.exp(b_i))

def compute_ece(probs, labels, n_bins=N_BINS):
    """Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = labels[in_bin].mean()
            avg_confidence_in_bin = probs[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

# ---------------------------------------- #
# MAIN LOOP
# ---------------------------------------- #

plt.figure(figsize=(8, 8))
plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')

for run_name, label in zip(run_names, labels):
    print(f"\n=== Processing {run_name} ===")

    # Infer config name from run
    base_run_name = run_name.replace("temperature_scaled_", "").replace(".pth", "")
    config_name = f"{base_run_name}.yaml"

    # Load configs
    default_config = OmegaConf.load(os.path.join("early_stopping_fixed_num_steps", "experiments", "default.yaml"))
    specific_config = OmegaConf.load(os.path.join("early_stopping_fixed_num_steps", "experiments", config_name))
    config = OmegaConf.merge(default_config, specific_config)
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.name = config_name

    # Reproducibility
    torch.manual_seed(config.misc.seed)
    np.random.seed(config.misc.seed)
    random.seed(config.misc.seed)

    # Trainer for data loaders
    trainer = Trainer(config, base_run_name)

    # Initialize base model and load weights
    base_model = initialize_model(config)
    checkpoint_path = os.path.join("early_stopping_fixed_num_steps", "trained_models", f"{base_run_name}.pth")
    base_model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
    base_model.eval()

    # Load saved calibration results
    saved = torch.load(os.path.join(root_dir, run_name), map_location=config.device)

    # Initialize temperature-scaled model
    model = TemperatureScaledMessageClassifier(base_model).to(config.device)

    # Apply the final temperature parameter
    if "T_final" in saved:
        T_final = float(saved["T_final"])
        model.temperature.data = torch.tensor(T_final, device=config.device)
        print(f"Loaded T_final = {T_final:.4f}")
    else:
        print("Warning: No T_final found in checkpoint; using temperature from state_dict.")

    # Load model state_dict if available
    if "state_dict" in saved:
        model.load_state_dict(saved["state_dict"], strict=False)
    else:
        print("No state_dict found — using base model + manually set temperature.")

    model.eval()

    # Evaluate on validation (or test) set
    loader = trainer.val_loader   # replace with test_loader if you prefer
    probs, labels_arr = [], []

    with torch.no_grad():
        for features, targets in loader:
            outputs = model(features.to(config.device)).detach().cpu()  # logits already scaled
            for message, target in zip(outputs, targets):
                prob = get_bot_probability(message[0].numpy(), message[1].numpy())
                probs.append(prob)
                labels_arr.append(int(target.numpy()[0] == 0))  # 1 = bot, 0 = human

    probs = np.array(probs)
    labels_arr = np.array(labels_arr)

    # Balance classes (optional)
    bot_indices = np.where(labels_arr == 1)[0]
    human_indices = np.where(labels_arr == 0)[0]
    n_bots = len(bot_indices)
    keep_humans = human_indices[:n_bots]
    balanced_indices = np.sort(np.concatenate([bot_indices, keep_humans]))
    probs_bal = probs[balanced_indices]
    labels_bal = labels_arr[balanced_indices]

    print(f"Balanced dataset: {len(labels_bal)} samples ({len(bot_indices)} bots + {len(keep_humans)} humans)")

    fraction_of_positives, mean_predicted_value = calibration_curve(
        labels_bal, probs_bal, n_bins=N_BINS, strategy='uniform'
    )

    plt.plot(mean_predicted_value, fraction_of_positives, 's-', label=f'{label}')

    ece_score = compute_ece(probs_bal, labels_bal)
    brier = brier_score_loss(labels_bal, probs_bal)
    print(f"ECE: {ece_score:.4f}")
    print(f"Brier Score: {brier:.4f}")

plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives (actual)')
plt.title('Reliability Diagram (Balanced)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
