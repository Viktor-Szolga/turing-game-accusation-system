import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from src.utils import initialize_model
from src.trainer import Trainer
from omegaconf import OmegaConf
import random


ROOT_DIR = "gridsearch"
CHECKPOINT_DIR = "trained_models"
N_BINS = 10
METRIC = "ece"  # "ece" or "brier"
TOP_K = 3 # Number plotted

def get_bot_probability(h_i, b_i):
    return np.exp(b_i) / (np.exp(h_i) + np.exp(b_i))

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

def evaluate_model(model, val_loader):
    model.eval()
    probs = []
    labels = []
    with torch.no_grad():
        for features, targets in val_loader:
            output = model(features)
            for message, target in zip(output, targets):
                prob = get_bot_probability(message[0].detach().numpy(), message[1].detach().numpy())
                probs.append(prob)
                labels.append(int(target.detach().numpy()[0] == 0))  # 1 = bot, 0 = human

    probs = np.array(probs)
    labels = np.array(labels)
    
    # Balance dataset: keep first n human messages
    bot_indices = np.where(labels == 1)[0]
    human_indices = np.where(labels == 0)[0]
    n_bots = len(bot_indices)
    keep_humans = human_indices[:n_bots]
    balanced_indices = np.concatenate([bot_indices, keep_humans])
    balanced_indices.sort()
    
    probs_bal = probs[balanced_indices]
    labels_bal = labels[balanced_indices]

    if METRIC == "ece":
        score = compute_ece(probs_bal, labels_bal)
    elif METRIC == "brier":
        score = brier_score_loss(labels_bal, probs_bal)
    else:
        raise ValueError(f"Unknown metric {METRIC}")

    return score, probs_bal, labels_bal


checkpoint_files = [f for f in os.listdir(os.path.join(ROOT_DIR, CHECKPOINT_DIR)) if f.endswith(".pth")]
results = []

for ckpt_file in checkpoint_files:
    default_config = OmegaConf.load(os.path.join(ROOT_DIR, "experiments", "default.yaml"))
    specific_config = OmegaConf.load(os.path.join(ROOT_DIR, "experiments", f"{ckpt_file[:-4]}.yaml"))
    config = OmegaConf.merge(default_config, specific_config)
    config.device = "cpu"
    config.name = f"{ckpt_file[:-4]}.yaml"

    torch.manual_seed(config.misc.seed)
    np.random.seed(config.misc.seed)
    random.seed(config.misc.seed)

    trainer = Trainer(config, ckpt_file[:-4])

    model = initialize_model(config)
    model.load_state_dict(torch.load(os.path.join(ROOT_DIR, CHECKPOINT_DIR, ckpt_file)))
    
    score, probs_bal, labels_bal = evaluate_model(model, trainer.val_loader)
    results.append({
        "file": ckpt_file,
        "score": score,
        "probs": probs_bal,
        "labels": labels_bal
    })
    print(f"{ckpt_file}: {METRIC.upper()} = {score:.4f}")


results.sort(key=lambda x: x["score"])
top_results = results[:TOP_K]

plt.figure(figsize=(8, 8))
plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')

for i, res in enumerate(top_results):
    frac_pos, mean_pred = calibration_curve(res["labels"], res["probs"], n_bins=N_BINS, strategy='uniform')
    plt.plot(mean_pred, frac_pos, 's-', label=f"{res['file']}")

plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives (actual)')
plt.title(f'Reliability Diagram - Top {TOP_K} models by {METRIC.upper()}')
plt.legend()
plt.grid(True)
plt.show()
