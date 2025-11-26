import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, DataLoader
from omegaconf import OmegaConf

from src.utils import initialize_model
from src.trainer import Trainer
from src.models import TemperatureScaledMessageClassifier


# ============================================================
# Helper Functions
# ============================================================

def compute_class_weights(dataset):
    """Compute class weights inversely proportional to class frequency."""
    labels = np.array([y for _, y in dataset])
    classes, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = total / (len(classes) * counts)
    return weights


def set_temperature(scaled_model, valid_loader, class_weights=None, device="cpu"):
    """Optimize the temperature parameter on validation data (one fold)."""
    scaled_model.eval()

    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        nll_criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    else:
        nll_criterion = nn.CrossEntropyLoss().to(device)

    # Collect logits and labels
    logits_list, labels_list = [], []
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = scaled_model.model(inputs)
            logits_list.append(logits)
            labels_list.append(labels)
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)

    # Optimize only the temperature
    optimizer = torch.optim.LBFGS(
        [scaled_model.temperature],
        lr=0.01,
        max_iter=10000,
        line_search_fn="strong_wolfe"
    )

    def eval():
        optimizer.zero_grad()
        loss = nll_criterion(scaled_model.temperature_scale(logits), labels)
        loss.backward()
        return loss

    optimizer.step(eval)
    return scaled_model.temperature.item()

def get_bot_probability(h_i, b_i):
    return np.exp(b_i) / (np.exp(h_i) + np.exp(b_i))

def compute_ece(model, loader, device="cpu", n_bins=10):
    """Compute ECE on a balanced subset of validation data (exactly like plotting code)."""
    model.eval()
    probs_list, labels_list = [], []

    # Collect all probabilities and labels
    with torch.no_grad():
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            output = model(features)
            for message, target in zip(output, targets):
                prob = get_bot_probability(message[0].cpu().numpy(), message[1].cpu().numpy())
                probs_list.append(prob)
                labels_list.append(int(target.cpu().numpy()[0] == 0))  # 1=bot, 0=human

    probs = np.array(probs_list)
    labels = np.array(labels_list)

    # Balance classes
    bot_indices = np.where(labels == 1)[0]
    human_indices = np.where(labels == 0)[0]
    n_bots = len(bot_indices)
    np.random.seed(42)
    np.random.shuffle(human_indices)
    keep_humans = human_indices[:n_bots]
    balanced_indices = np.sort(np.concatenate([bot_indices, keep_humans]))

    probs_bal = probs[balanced_indices]
    labels_bal = labels[balanced_indices]

    # Compute ECE using the same binning as your plotting function
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (probs_bal > bin_lower) & (probs_bal <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = labels_bal[in_bin].mean()
            avg_confidence_in_bin = probs_bal[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    root_folder = "early_stopping_fixed_epoch"
    model_list = ["run127.yaml", "run138.yaml", "run147.yaml", "run304.yaml"]
    is_checkpoint = False
    save_dir = "temperature_scaling_cv"
    os.makedirs(save_dir, exist_ok=True)
    results = []

    # Use first model to define CV splits (shared across all)
    first_config = OmegaConf.load(os.path.join(root_folder, "experiments", model_list[0]))
    default_config = OmegaConf.load(os.path.join(root_folder, "experiments", "default.yaml"))
    base_config = OmegaConf.merge(default_config, first_config)
    base_config.device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(base_config.misc.seed)
    np.random.seed(base_config.misc.seed)
    random.seed(base_config.misc.seed)

    trainer = Trainer(base_config, model_list[0][:-5])
    val_dataset = trainer.val_loader.dataset
    class_weights = compute_class_weights(val_dataset)
    class_weights = None
    labels = np.array([y for _, y in val_dataset])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=base_config.misc.seed)
    folds = list(skf.split(np.arange(len(np.argmax(labels, axis=1))), np.argmax(labels, axis=1)))

    print(f"\nUsing shared 5-fold CV splits for all models.")
    print(f"Class weights: {class_weights}")

    for config_name in model_list:
        print(f"\n===============================")
        print(f"Processing model {config_name}")
        print(f"===============================")

        # Load config
        specific_config = OmegaConf.load(os.path.join(root_folder, "experiments", config_name))
        config = OmegaConf.merge(default_config, specific_config)
        config.device = base_config.device

        # Initialize model + weights
        base_model = initialize_model(config)
        checkpoint_path = os.path.join(
            root_folder,
            "checkpoints" if is_checkpoint else "trained_models",
            f"{config_name[:-5]}.pth"
        )
        base_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        base_model.eval()

        fold_temps = []
        fold_eces = []

        for fold_idx, (_, val_idx) in enumerate(folds):
            val_subset = Subset(val_dataset, val_idx)
            val_loader = DataLoader(val_subset, batch_size=config.training.batch_size, shuffle=False)

            scaled_model = TemperatureScaledMessageClassifier(base_model).to(config.device)
            T_fold = set_temperature(scaled_model, val_loader, class_weights=class_weights, device=config.device)
            fold_temps.append(T_fold)

            # Set fold temperature and compute ECE
            scaled_model.temperature.data = torch.tensor(T_fold, device=config.device)
            ece_fold = compute_ece(scaled_model, val_loader, device=config.device)
            fold_eces.append(ece_fold)

            print(f"Fold {fold_idx + 1}: T={T_fold:.4f}, ECE={ece_fold:.4f}")

        T_median = float(np.median(fold_temps))
        T_mean = float(np.mean(fold_temps))
        ECE_mean = float(np.mean(fold_eces))

        print(f"\nMedian T={T_median:.4f} | Mean T={T_mean:.4f} | Mean ECE={ECE_mean:.4f}")

        # Refit final T on full validation
        full_loader = trainer.val_loader
        final_scaled_model = TemperatureScaledMessageClassifier(base_model).to(config.device)
        final_T = set_temperature(final_scaled_model, full_loader, class_weights=class_weights, device=config.device)
        final_scaled_model.temperature.data = torch.tensor(final_T, device=config.device)

        # Compute final ECEs (with and without scaling)
        unscaled_model = TemperatureScaledMessageClassifier(base_model).to(config.device)
        unscaled_model.temperature.data = torch.tensor(1.0, device=config.device)
        ece_unscaled = compute_ece(unscaled_model, full_loader, device=config.device)
        ece_scaled = compute_ece(final_scaled_model, full_loader, device=config.device)

        # Save both models
        torch.save(unscaled_model.state_dict(), os.path.join(save_dir, f"{config_name[:-5]}_T1.pth"))
        torch.save({
            "fold_temps": fold_temps,
            "T_median": T_median,
            "T_mean": T_mean,
            "T_final": final_T,
            "state_dict": final_scaled_model.state_dict()
        }, os.path.join(save_dir, f"{config_name[:-5]}_scaled.pth"))

        # Log results
        results.append({
            "model": config_name[:-5],
            "T_mean": T_mean,
            "T_median": T_median,
            "T_final": final_T,
            "ECE_mean_CV": ECE_mean,
            "ECE_unscaled": ece_unscaled,
            "ECE_scaled": ece_scaled
        })

        print(f"\nSaved unscaled and scaled versions for {config_name[:-5]}")

    # Write CSV summary
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(save_dir, "temperature_summary.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved summary CSV to: {csv_path}")
