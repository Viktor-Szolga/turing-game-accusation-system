import torch
import numpy as np
import random
from src.utils import initialize_model
from omegaconf import OmegaConf
from src.trainer import Trainer
import os
import time  # <-- for timing
import pickle

if __name__ == "__main__":
    training_information = {}
    log_file = "training_log.txt"
    print("cuda" if torch.cuda.is_available() else "cpu")

    # Repeat for each experiment to be performed
    for config_name in os.listdir("experiments"):
        if config_name == "default.yaml":
            continue
        if config_name[:-5] in [name[:-4] for name in os.listdir("training_information")]:
            continue
        
        # Reset seed to ensure reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        # Merge default config with run config to ensure every value is set
        default_config = OmegaConf.load(os.path.join("experiments", "default.yaml"))
        specific_config = OmegaConf.load(os.path.join("experiments", config_name))
        config = OmegaConf.merge(default_config, specific_config)
        config.device = "cuda" if torch.cuda.is_available() else "cpu"
        config.name = config_name

        # Initialize model and trainer
        trainer = Trainer(config, config_name[:-5])
        model = initialize_model(config)

        # Train model
        start_time = time.time()
        model, train_acc, validation_acc, train_loss, validation_loss = trainer.train(model)
        end_time = time.time()
        elapsed_time = end_time - start_time  # seconds
        epochs_taken = len(train_loss)  # actual number of epochs completed

        # Record results
        results = {
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'validation_loss': validation_loss,
            'validation_accuracy': validation_acc
        }
        training_information[config_name[:-5]] = results
        
        # Save state dict
        torch.save(model.state_dict(), os.path.join("trained_models", f"{config_name[:-5]}.pth"))

        # Save accuracy, loss etc. to disc
        with open(os.path.join("training_information", f"{config_name[:-5]}.pkl"), "wb") as f:
            pickle.dump(results, f)


        # Append run info to log file
        with open(log_file, "a") as f:
            f.write(f"{config_name[:-5]} | epochs: {epochs_taken} | time (s): {elapsed_time:.2f} | final validation loss: {validation_loss[-1]:.7f}\n")