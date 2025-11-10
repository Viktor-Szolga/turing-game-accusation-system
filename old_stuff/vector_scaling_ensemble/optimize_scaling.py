import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from omegaconf import OmegaConf
import src.utils as utils
import numpy as np
import random
from src.utils import initialize_model
from src.trainer import Trainer
from src.models import VectorScaledEnsemble, EnsembleMessageClassifier

def set_vector_scaling(scaled_model, valid_loader, device="cpu"):
    """
    Optimize W and b parameters on validation data using LBFGS.
    """
    scaled_model.eval()
    nll_criterion = nn.CrossEntropyLoss().to(device)

    # Collect logits and labels
    logits_list, labels_list = [], []
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = scaled_model.model(inputs)  # base model output (before scaling)
            logits_list.append(logits)
            labels_list.append(labels)

    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)

    # Optimize W and b only
    optimizer = torch.optim.LBFGS(
        [scaled_model.W, scaled_model.b],
        lr=0.01,
        max_iter=1000,
        line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        loss = nll_criterion(scaled_model.vector_scale(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    loss_val = nll_criterion(scaled_model.vector_scale(logits), labels).item()
    print(f"Final calibration loss (NLL): {loss_val:.4f}")
    print("Optimized W:\n", scaled_model.W.data)
    print("Optimized b:\n", scaled_model.b.data)
    return scaled_model


# ======= Main =======
if __name__ == "__main__":
    root_folder = "early_stopping_fixed_epoch"
    run_names = [f"run{i:03d}" for i in range(7, 300, 30)]
    config_name = f"{run_names[0]}.yaml"
    is_checkpoint = False

    # Load configs
    default_config = OmegaConf.load(os.path.join(root_folder, "experiments", "default.yaml"))
    specific_config = OmegaConf.load(os.path.join(root_folder, "experiments", config_name))
    config = OmegaConf.merge(default_config, specific_config)
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.name = config_name

    # Set seeds
    torch.manual_seed(config.misc.seed)
    np.random.seed(config.misc.seed)
    random.seed(config.misc.seed)

    # Initialize trainer and base model
    trainer = Trainer(config, config_name[:-5])
    models_dir = "checkpoints" if is_checkpoint else "trained_models"

    models = [os.path.join(root_folder, models_dir, f"{run_name}.pth") for run_name in run_names]
    base_model = EnsembleMessageClassifier(1024, [128], 2, models)

    # Wrap the base model with vector scaling
    scaled_model = VectorScaledEnsemble(base_model, num_classes=2).to(config.device)

    # Optimize scaling parameters on validation set
    loader = trainer.val_loader
    epochs = 3
    for _ in range(epochs):
        scaled_model = set_vector_scaling(scaled_model, loader, device=config.device)

    # Save the calibrated model
    os.makedirs("vector_scaling_ensemble", exist_ok=True)
    torch.save(scaled_model.state_dict(), "vector_scaling_ensemble/vector_scaled_run039.pth")
    print("✅ Saved vector-scaled ensemble model!")