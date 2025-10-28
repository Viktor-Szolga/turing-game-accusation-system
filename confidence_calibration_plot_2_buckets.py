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

root_dir = "early_stopping_fixed_epoch"
run_names = ["run127", "run138", "run147", "run304"]
labels = ["1-Layer 128-Neurons", "2-Layer 256-128-Neurons", "3-Layer 128-64-32", "2-Layer 48-24"]
is_checkpoint = False
bin_settings = [10, 50]  # Two bin sizes

def get_bot_probability(h_i, b_i):
    return np.exp(b_i) / (np.exp(h_i) + np.exp(b_i))

def compute_ece(probs, labels, n_bins):
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

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

for ax, n_bins in zip(axes, bin_settings):
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.set_title(f'Calibration Curve (N_BINS={n_bins})')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.grid(True)

    for run_name, label in zip(run_names, labels):
        print(f"\n=== Processing {run_name} with {n_bins} bins ===")

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
            model.load_state_dict(torch.load(os.path.join(root_dir, "checkpoints", f"{run_name}.pth"), map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(os.path.join(root_dir, "trained_models", f"{run_name}.pth"), map_location=torch.device('cpu')))
        model.eval()

        probs, labels_list = [], []
        loader = trainer.val_loader
        for features, targets in loader:
            output = model(features).to(config.device)
            for message, target in zip(output, targets):
                prob = get_bot_probability(message[0].detach().numpy(), message[1].detach().numpy())
                probs.append(prob)
                labels_list.append(int(target.detach().numpy()[0] == 0))  # 1 = bot, 0 = human

        probs = np.array(probs)
        labels_array = np.array(labels_list)

        # Balance classes
        bot_indices = np.where(labels_array == 1)[0]
        np.random.seed(42)
        human_indices = np.where(labels_array == 0)[0]
        np.random.seed(config.misc.seed)
        n_bots = len(bot_indices)
        np.random.shuffle(human_indices)
        keep_humans = human_indices[:n_bots]
        balanced_indices = np.sort(np.concatenate([bot_indices, keep_humans]))
        probs_bal = probs[balanced_indices]
        labels_bal = labels_array[balanced_indices]
        preds = np.where(probs_bal > 0.5, 1, 0)
        acc = np.mean(preds == labels_bal.flatten())
        print("Accuracy:", acc)
        print(f"Balanced dataset: {len(labels_bal)} samples ({len(bot_indices)} bots + {len(keep_humans)} humans)")

        fraction_of_positives, mean_predicted_value = calibration_curve(
            labels_bal, probs_bal, n_bins=n_bins, strategy='quantile'
        )

        ax.plot(mean_predicted_value, fraction_of_positives,
                linestyle='-',
                linewidth=2,
                marker='o',
                markersize=5,
                markevery=3,
                alpha=0.85,
                label=label)

        ece_score = compute_ece(probs_bal, labels_bal, n_bins)
        brier = brier_score_loss(labels_bal, probs_bal)
        print(f"ECE: {ece_score:.4f}")
        print(f"Brier Score: {brier:.4f}")

    ax.legend()

plt.tight_layout()
plt.show()
