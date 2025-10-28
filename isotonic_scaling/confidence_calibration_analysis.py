import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import torch
import random
from src.utils import initialize_model
from src.trainer import Trainer
from omegaconf import OmegaConf
from sklearn.metrics import brier_score_loss
import src.utils as utils

# Import Platt scaling wrapper
from src.models import PlattScaledMessageClassifier

# === CONFIG ===
N_BINS = 10
root_dir = "platt_scaling"
run_names = ["platt_scaled_run039.pth"]
labels = ["Platt-scaled model"]
is_checkpoint = False  # We are loading a pre-calibrated model, not a base checkpoint

# === HELPER FUNCTIONS ===
def compute_ece(probs, labels, n_bins=N_BINS):
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


# === MAIN ===
plt.figure(figsize=(8, 8))
plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')

for run_name, label in zip(run_names, labels):
    print(f"\n=== Processing {run_name} ===")

    # --- Load configs ---
    config_name = f"{run_name.replace('.pth', '')}.yaml"
    default_config = OmegaConf.load(os.path.join("early_stopping", "experiments", "default.yaml"))
    specific_config = OmegaConf.load(os.path.join("early_stopping", "experiments", config_name[13:]))  # Adjust offset if needed
    config = OmegaConf.merge(default_config, specific_config)
    config.device = "cpu"
    config.name = config_name

    # --- Set seeds for reproducibility ---
    torch.manual_seed(config.misc.seed)
    np.random.seed(config.misc.seed)
    random.seed(config.misc.seed)

    # --- Initialize trainer and base model ---
    trainer = Trainer(config, config_name.replace('.yaml',''))
    base_model = initialize_model(config)

    # --- Wrap with Platt scaling and load pre-trained scaling parameters ---
    model = PlattScaledMessageClassifier(base_model)
    model.load_state_dict(torch.load(os.path.join(root_dir, run_name), map_location=config.device))
    model.eval()

    # --- Collect probabilities and labels ---
    probs, labels_arr = [], []
    loader = utils.get_full_test_data_loader()  # or trainer.val_loader / train_loader
    loader = utils.get_test_data_loader(os.path.join("data", "test_data", "second"))
    for features, targets in loader:
        features = features.to(config.device)
        with torch.no_grad():
            probs_batch = model(features).detach().cpu().numpy()  # already sigmoid probabilities
        targets_np = targets.numpy()

        probs.extend(probs_batch.flatten())
        labels_arr.extend(targets_np.flatten())

    probs = np.array(probs)
    labels_arr = np.array(labels_arr)

    # --- Balance classes (same as before) ---
    bot_indices = np.where(labels_arr == 1)[0]
    human_indices = np.where(labels_arr == 0)[0]
    n_bots = len(bot_indices)
    keep_humans = human_indices[:n_bots]
    balanced_indices = np.sort(np.concatenate([bot_indices, keep_humans]))
    probs_bal = probs[balanced_indices]
    labels_bal = labels_arr[balanced_indices]

    print(f"Balanced dataset: {len(labels_bal)} samples ({len(bot_indices)} bots + {len(keep_humans)} humans)")

    # --- Compute calibration curve ---
    fraction_of_positives, mean_predicted_value = calibration_curve(
        labels_bal, probs_bal, n_bins=N_BINS, strategy='quantile'
    )

    plt.plot(mean_predicted_value, fraction_of_positives, 's-', label=f'{label}')

    # --- Metrics ---
    ece_score = compute_ece(probs_bal, labels_bal)
    brier = brier_score_loss(labels_bal, probs_bal)
    print(f"ECE: {ece_score:.4f}")
    print(f"Brier Score: {brier:.4f}")

# --- Plot ---
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives (actual)')
plt.title('Reliability Diagram (Balanced)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
