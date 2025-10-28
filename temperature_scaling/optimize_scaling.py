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
from src.models import TemperatureScaledMessageClassifier

def set_temperature(scaled_model, valid_loader, device='cpu'):
    """
    Optimize the temperature parameter on validation data.
    scaled_model: instance of TemperatureScaledMessageClassifier
    """
    scaled_model.eval()
    nll_criterion = nn.CrossEntropyLoss().to(device)
    
    # Collect logits and labels from base model
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = scaled_model.model(inputs)  # base model output
            logits_list.append(logits)
            labels_list.append(labels)
    
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    
    # Optimize only the temperature parameter
    optimizer = torch.optim.LBFGS([scaled_model.temperature], lr=0.01, max_iter=10000, line_search_fn='strong_wolfe')

    def eval():
        optimizer.zero_grad()
        loss = nll_criterion(scaled_model.temperature_scale(logits), labels)
        loss.backward()
        return loss

    optimizer.step(eval)
    print(f"Optimal temperature: {scaled_model.temperature.item():.3f}")
    return scaled_model


if __name__ == "__main__":
    root_folder = "early_stopping_fixed_num_steps"
    config_name = "run039.yaml"
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
        root_folder, "checkpoints" if is_checkpoint else "trained_models", f"{config_name[:-5]}.pth"
    )
    base_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    base_model.eval()

    # Wrap the base model with temperature scaling
    scaled_model = TemperatureScaledMessageClassifier(base_model).to(config.device)

    # Optimize temperature on validation set
    epochs = 3
    #loader = utils.get_test_data_loader(os.path.join("data", "test_data", "first"), balanced=True)
    loader = trainer.val_loader
    for _ in range(epochs):
        scaled_model = set_temperature(scaled_model, loader, device=config.device)
    
    # Save the temperature-scaled model
    os.makedirs("temperature_scaling", exist_ok=True)
    torch.save(scaled_model.state_dict(), "temperature_scaling/temperature_scaled_run039.pth")
    print("Saved temperature-scaled model!")
