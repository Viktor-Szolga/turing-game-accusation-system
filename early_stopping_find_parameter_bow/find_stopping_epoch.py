import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import numpy as np
import random
from src.utils import initialize_model
from omegaconf import OmegaConf
from src.trainer import Trainer
import os
import time 
from sklearn.model_selection import StratifiedKFold
import pickle


if __name__ == "__main__":
    root_folder = "early_stopping_find_parameter_bow"  # "early_stopping" or "gridsearch"
    os.makedirs(os.path.join(root_folder, "training_information"), exist_ok=True)
    os.makedirs(os.path.join(root_folder, "trained_models"), exist_ok=True)
    os.makedirs(os.path.join(root_folder, "checkpoints"), exist_ok=True)

    print(f"Running {root_folder}")

    training_information = {}
    log_file = os.path.join(root_folder, "early_stopping_parameter_log.txt")
    print("cuda" if torch.cuda.is_available() else "cpu")

    # Repeat for each experiment to be performed
    for config_name in os.listdir(os.path.join(root_folder, "experiments")):
        if config_name == "default.yaml":
            continue
        if config_name[:-5] in [name[:-4] for name in os.listdir(os.path.join(root_folder, "training_information"))]:
            continue

        # Merge default config with run config to ensure every value is set
        default_config = OmegaConf.load(os.path.join(root_folder, "experiments", "default.yaml"))
        specific_config = OmegaConf.load(os.path.join(root_folder, "experiments", config_name))
        config = OmegaConf.merge(default_config, specific_config)
        config.device = "cuda" if torch.cuda.is_available() else "cpu"
        config.name = config_name
        trainer = Trainer(config, config_name[:-5])

        # Reset seed to ensure reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        n_epochs_till_stopping = []
        for fold, (train_idx, val_idx) in enumerate(kfold.split(np.arange(len(np.argmax(trainer.y_train, axis=1))), np.argmax(trainer.y_train, axis=1))):
            print(f"Fold {fold}:")
            

            model = initialize_model(config)
            n_epochs_till_stopping.append(trainer.find_stopping_epoch(model, root_folder, train_idx, val_idx))

        print(f"Run {config_name[:-5]} had stopping epochs: {n_epochs_till_stopping} Mean: {np.mean(n_epochs_till_stopping)} Median: {np.median(n_epochs_till_stopping)}]")

        
        with open(log_file, "a") as f:
            f.write(f"{config_name[:-5]} | stopping_epochs: {n_epochs_till_stopping} | average_stopping: {np.mean(n_epochs_till_stopping)} | median_stopping: {np.median(n_epochs_till_stopping)}\n")