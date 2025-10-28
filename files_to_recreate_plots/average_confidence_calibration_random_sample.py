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

N_BINS = 10
root_dir = "early_stopping"
model_runs = {
    "Early stopping": [f"run{i:03d}" for i in range(9, 300, 30)],  # Example runs
    "Dropout": [f"run{i}" for i in range(300, 310)]
}
is_checkpoint = True

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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Create subplots for models
fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
for ax, (model_label, run_names) in zip(axes, model_runs.items()):
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    
    ece_list = []
    brier_list = []

    for run_name in run_names:
        print(f"\n=== Processing {run_name} ({model_label}) ===")
        
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

        checkpoint_path = os.path.join(
            root_dir, "checkpoints" if is_checkpoint else "trained_models", f"{run_name}.pth"
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
        model.eval()

        # Collect probabilities and labels
        probs, labels_arr = [], []
        for features, targets in trainer.train_loader:
            output = model(features).to(config.device)
            for message, target in zip(output, targets):
                prob = get_bot_probability(message[0].detach().numpy(), message[1].detach().numpy())
                probs.append(prob)
                labels_arr.append(int(target.detach().numpy()[0] == 0))  # 1 = bot, 0 = human

        probs = np.array(probs)
        labels_arr = np.array(labels_arr)

        bot_indices = np.where(labels_arr == 1)[0]
        human_indices = np.where(labels_arr == 0)[0]
        n_bots = len(bot_indices)
        shuffled_human_indices = np.copy(human_indices)
        np.random.shuffle(shuffled_human_indices)
        keep_humans = shuffled_human_indices[:n_bots]
        balanced_indices = np.sort(np.concatenate([bot_indices, keep_humans]))
        probs_bal = probs[balanced_indices]
        labels_bal = labels_arr[balanced_indices]

        print(f"Balanced dataset: {len(labels_bal)} samples ({len(bot_indices)} bots + {len(keep_humans)} humans)")

        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            labels_bal, probs_bal, n_bins=N_BINS, strategy='uniform'
        )

        ax.plot(mean_predicted_value, fraction_of_positives, 's-', label=run_name)

        ece_score = compute_ece(probs_bal, labels_bal)
        brier = brier_score_loss(labels_bal, probs_bal)
        ece_list.append(ece_score)
        brier_list.append(brier)
        print(f"ECE: {ece_score:.4f}, Brier Score: {brier:.4f}")

    # Average scores
    print(f"\n=== {model_label} Averages ===")
    print(f"Average ECE: {np.mean(ece_list):.4f} ± {np.std(ece_list):.4f}")
    print(f"Average Brier Score: {np.mean(brier_list):.4f} ± {np.std(brier_list):.4f}")

    ax.set_title(model_label)
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives (actual)')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
