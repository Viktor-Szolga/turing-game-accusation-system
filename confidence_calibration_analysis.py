import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import torch
import random
from src.utils import initialize_model
from omegaconf import OmegaConf
from src.trainer import Trainer
import os
import pickle
from sklearn.metrics import brier_score_loss

N_BINS = 10
run_name = "run039"
run_name = "run300"
is_checkpoint = False

training_information = {}
print("cuda" if torch.cuda.is_available() else "cpu")

config_name = f"{run_name}.yaml"
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

default_config = OmegaConf.load(os.path.join("experiments", "default.yaml"))
specific_config = OmegaConf.load(os.path.join("experiments", config_name))
config = OmegaConf.merge(default_config, specific_config)
config.device = "cpu"
config.name = config_name

# Initialize model and trainer
trainer = Trainer(config, run_name)
model = initialize_model(config)

if is_checkpoint:
    model.load_state_dict(torch.load(f"checkpoints/{run_name}.pth"))
else:
    model.load_state_dict(torch.load(f"trained_models/{run_name}.pth"))
model.eval()

def get_bot_probability(h_i, b_i):
    return np.exp(b_i) / (np.exp(h_i) + np.exp(b_i))

probs = []
labels = []

for features, targets in trainer.val_loader:
    output = model(features).to(config.device)
    for message, target in zip(output, targets):
        prob = get_bot_probability(message[0].detach().numpy(), message[1].detach().numpy())
        probs.append(prob)
        labels.append(int(target.detach().numpy()[0] == 0))  # 1 = bot, 0 = human

# --- BALANCING STEP HERE ---
# Count how many bot messages exist
probs = np.array(probs)
labels = np.array(labels)

bot_indices = np.where(labels == 1)[0]
human_indices = np.where(labels == 0)[0]

n_bots = len(bot_indices)
n_humans = len(human_indices)

# Keep only the first n human messages
keep_humans = human_indices[:n_bots]
balanced_indices = np.concatenate([bot_indices, keep_humans])
balanced_indices.sort()

probs_bal = probs[balanced_indices]
labels_bal = labels[balanced_indices]

print(f"Balanced dataset: {len(labels_bal)} samples ({len(bot_indices)} bots + {len(keep_humans)} humans)")

# --- CALIBRATION & METRICS ---
fraction_of_positives, mean_predicted_value = calibration_curve(
    labels_bal, probs_bal, n_bins=N_BINS, strategy='uniform'
)

plt.figure(figsize=(8, 8))
plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
plt.plot(mean_predicted_value, fraction_of_positives, 's-', label='Trained classifier')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives (actual)')
plt.title('Reliability Diagram (Balanced)')
plt.legend()
plt.grid(True)
plt.show()


def compute_ece(probs, labels, n_bins=N_BINS):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = labels[in_bin].mean()
            avg_confidence_in_bin = probs[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

ece_score = compute_ece(probs_bal, labels_bal)
print(f"ECE: {ece_score:.4f}")

brier = brier_score_loss(labels_bal, probs_bal)
print(f"Brier Score: {brier:.4f}")
