import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import torch
import random
from src.utils import initialize_model
from omegaconf import OmegaConf
from src.trainer import Trainer
from sklearn.metrics import brier_score_loss
import src.utils as utils

N_BINS = 10
root_dir = "gridsearch_smaller_models"
run_names = [f"run{i:03d}" for i in range(3)]
labels = run_names
is_checkpoint = True
colors = ['C0', 'C1', 'C2', 'C0', 'C1', 'C2']

def get_bot_probability(h_i, b_i):
    return np.exp(b_i) / (np.exp(h_i) + np.exp(b_i))

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

print("cuda" if torch.cuda.is_available() else "cpu")

plt.figure(figsize=(8, 8))
plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')

for run_name, label, color in zip(run_names, labels, colors):
    print(f"\n=== Processing {run_name} ===")

    config_name = f"{run_name}.yaml"
    default_config = OmegaConf.load(os.path.join(root_dir, "experiments", "default.yaml"))
    specific_config = OmegaConf.load(os.path.join(root_dir, "experiments", config_name))
    config = OmegaConf.merge(default_config, specific_config)
    config.device = "cpu"
    config.name = config_name

    torch.manual_seed(config.misc.seed)
    np.random.seed(config.misc.seed)
    random.seed(config.misc.seed)

    trainer = Trainer(config, run_name)
    model = initialize_model(config)

    if is_checkpoint:
        model.load_state_dict(torch.load(os.path.join(root_dir, "checkpoints", f"{run_name}.pth")))
    else:
        model.load_state_dict(torch.load(os.path.join(root_dir, "trained_models", f"{run_name}.pth")))
    model.eval()

    # Collect probabilities and labels
    probs, labels = [], []
    #loader = trainer.val_loader
    #loader = utils.get_test_data_loader(os.path.join("data", "test_data", "first"))
    loader = utils.get_full_test_data_loader()
    for features, targets in loader:
        output = model(features).to(config.device)
        for message, target in zip(output, targets):
            prob = get_bot_probability(message[0].detach().numpy(), message[1].detach().numpy())
            probs.append(prob)
            labels.append(int(target.detach().numpy()[0] == 0))  # 1 = bot, 0 = human

    probs = np.array(probs)
    labels = np.array(labels)

    # Balance classes
    bot_indices = np.where(labels == 1)[0]
    human_indices = np.where(labels == 0)[0]
    n_bots = len(bot_indices)
    keep_humans = human_indices[:n_bots]
    balanced_indices = np.sort(np.concatenate([bot_indices, keep_humans]))
    probs_bal = probs[balanced_indices]
    labels_bal = labels[balanced_indices]

    print(f"Balanced dataset: {len(labels_bal)} samples ({len(bot_indices)} bots + {len(keep_humans)} humans)")

    fraction_of_positives, mean_predicted_value = calibration_curve(
        labels_bal, probs_bal, n_bins=N_BINS, strategy='uniform'
    )

    plt.plot(mean_predicted_value, fraction_of_positives, 's-', label=f'{label}', color=color)

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
