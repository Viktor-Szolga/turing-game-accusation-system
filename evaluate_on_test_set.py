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
from src.models import TemperatureScaledMessageClassifier
from sklearn.metrics import brier_score_loss
from omegaconf import OmegaConf
import torchmetrics
import src.utils as utils

root_dir = "final_model"
run_names = ["scaled_model_d.pth"]
#run_names = ["model_d.pth"]
labels = ["Model D"]
styles = ['-']
colors = ['C0']
alpha = 0.5  # Slight transparency


def get_test_metrics(model, loader, device='cpu'):
    """Compute loss, accuracy, precision, recall, F1 (per class)."""
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Use multiclass mode to get per-class metrics
    accuracy = torchmetrics.Accuracy(num_classes=2, task="multiclass").to(device)
    precision = torchmetrics.Precision(num_classes=2, task="multiclass", average=None).to(device)
    recall = torchmetrics.Recall(num_classes=2, task="multiclass", average=None).to(device)
    f1 = torchmetrics.F1Score(num_classes=2, task="multiclass", average=None).to(device)

    total_loss = 0.0

    with torch.no_grad():
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            targets = torch.argmax(targets, dim=1)
            outputs = model(features)
            preds = torch.argmax(outputs, dim=1)

            total_loss += loss_fn(outputs, targets).item()
            accuracy.update(preds, targets)
            precision.update(preds, targets)
            recall.update(preds, targets)
            f1.update(preds, targets)

    loss = total_loss / len(loader)
    acc = accuracy.compute().item()
    prec = precision.compute().cpu().numpy()
    rec = recall.compute().cpu().numpy()
    f1s = f1.compute().cpu().numpy()

    model.train()
    return loss, acc, prec, rec, f1s



def get_bot_probability(h_i, b_i):
    return np.exp(b_i) / (np.exp(h_i) + np.exp(b_i))


def compute_ece(probs, labels, n_bins=10):
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


fig, axes = plt.subplots(1, 2, figsize=(16, 7))
bins_list = [10, 50]

for ax, N_BINS in zip(axes, bins_list):
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', alpha=0.6)

    for run_name, label, style, color in zip(run_names, labels, styles, colors):
        print(f"\n=== Processing {run_name} (bins={N_BINS}) ===")

        # Load configs
        config_name = f"run097.yaml"
        default_config = OmegaConf.load(os.path.join("early_stopping_fixed_epoch", "experiments", "default.yaml"))
        specific_config = OmegaConf.load(os.path.join(
            "early_stopping_fixed_epoch", "experiments", 
            config_name[19:] if len(config_name) > 19 else config_name
        ))
        config = OmegaConf.merge(default_config, specific_config)
        config.device = "cpu"
        config.name = config_name

        torch.manual_seed(config.misc.seed)
        np.random.seed(config.misc.seed)
        random.seed(config.misc.seed)

        trainer = Trainer(config, config_name.replace('.yaml',''))
        base_model = initialize_model(config)
        model = TemperatureScaledMessageClassifier(base_model)
        #model = initialize_model(config)

        # Load model
        model.load_state_dict(torch.load(os.path.join(root_dir, run_name), map_location=config.device))
        model.eval()

        # Collect probabilities and labels
        probs, labels_arr = [], []
        loader = utils.get_full_test_data_loader()
        #loader = trainer.val_loader   # Used to report validation performance of final model
        for features, targets in loader:
            output = model(features.to(config.device)).detach().cpu()
            for message, target in zip(output, targets):
                prob = get_bot_probability(message[0].numpy(), message[1].numpy())
                probs.append(prob)
                labels_arr.append(int(target.numpy()[0] == 0))

        probs = np.array(probs)
        labels_arr = np.array(labels_arr)

        # Balance classes for calibration
        bot_indices = np.where(labels_arr == 1)[0]
        human_indices = np.where(labels_arr == 0)[0]
        n_bots = len(bot_indices)
        keep_humans = human_indices[:n_bots]
        balanced_indices = np.sort(np.concatenate([bot_indices, keep_humans]))
        probs_bal = probs[balanced_indices]
        labels_bal = labels_arr[balanced_indices]

        fraction_of_positives, mean_predicted_value = calibration_curve(
            labels_bal, probs_bal, n_bins=N_BINS, strategy='quantile'
        )

        ax.plot(
            mean_predicted_value, fraction_of_positives, 
            linestyle=style, color=color, alpha=alpha, marker='o', markersize=2, linewidth=1.5,
            label=f'{label} {"Scaled" if style=="-" else "Unscaled"}'
        )

        ece_score = compute_ece(probs_bal, labels_bal, n_bins=N_BINS)
        brier = brier_score_loss(labels_bal, probs_bal)
        print(f"ECE (bins={N_BINS}): {ece_score:.4f}")
        print(f"Brier Score: {brier:.4f}")

        # Get test metrics
        loss, acc, prec, rec, f1s = get_test_metrics(model, loader)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {acc*100:.2f}%")
        print("Per-Class Metrics:")
        print(f"  Class 0 (Human):   Precision={prec[0]:.4f}, Recall={rec[0]:.4f}, F1={f1s[0]:.4f}")
        print(f"  Class 1 (Bot):     Precision={prec[1]:.4f}, Recall={rec[1]:.4f}, F1={f1s[1]:.4f}")

    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title(f'Reliability Diagram (bins={N_BINS})')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
