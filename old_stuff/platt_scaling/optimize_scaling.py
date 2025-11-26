import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from omegaconf import OmegaConf
import numpy as np
import random
from src.utils import initialize_model
from src.trainer import Trainer
import src.utils as utils
from src.models import PlattScaledMessageClassifier

def set_platt_params(scaled_model, valid_loader, device='cpu'):
    """
    Optimize the Platt scaling parameters (a, b) on validation data.
    scaled_model: instance of PlattScaledMessageClassifier
    """
    scaled_model.eval()
    bce_criterion = nn.BCELoss().to(device)

    logits_list = []
    labels_list = []

    # Collect logits and labels from the base model (without scaling)
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float()
            logits = scaled_model.model(inputs)  # base model output (pre-sigmoid)
            logits_list.append(logits)
            labels_list.append(labels)

    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)

    # Optimize a and b
    optimizer = torch.optim.LBFGS([scaled_model.a, scaled_model.b], lr=0.01, max_iter=50)

    def eval_fn():
        optimizer.zero_grad()
        probs = torch.sigmoid(scaled_model.a * logits + scaled_model.b)
        loss = bce_criterion(probs, labels)
        loss.backward()
        return loss

    optimizer.step(eval_fn)

    print(f"Optimal parameters: a = {scaled_model.a.item():.4f}, b = {scaled_model.b.item():.4f}")
    return scaled_model


# === 3. Main script ===
if __name__ == "__main__":
    root_folder = "early_stopping_fixed_epoch"
    config_name = "run247.yaml"
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
    base_model = initialize_model(config)

    # Load trained weights
    checkpoint_path = os.path.join(
        root_folder,
        "checkpoints" if is_checkpoint else "trained_models",
        f"{config_name[:-5]}.pth",
    )
    base_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    base_model.eval()

    # Wrap the base model with Platt scaling
    scaled_model = PlattScaledMessageClassifier(base_model).to(config.device)
    
    # Optimize Platt parameters on validation set
    loader = utils.get_test_data_loader(os.path.join("data", "test_data", "first"), balanced=True)
    loader = trainer.train_loader
    for i in range(100):
        scaled_model = set_platt_params(scaled_model, loader, device=config.device)

    # Save the calibrated model
    os.makedirs("platt_scaling", exist_ok=True)
    torch.save(scaled_model.state_dict(), "platt_scaling/platt_scaled_run039.pth")
    print("✅ Saved Platt-scaled model!")