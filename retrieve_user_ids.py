import torch
import numpy as np
import random
from src.utils import initialize_model
from omegaconf import OmegaConf
from src.trainer import Trainer
import os
import pickle

training_information = {}
print("cuda" if torch.cuda.is_available() else "cpu")
config_name = "run082.yaml"
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
trainer = Trainer(config)
model = initialize_model(config)

with open("data/user_ids.pkl", "rb") as f:
    user_ids = np.array(pickle.load(f))

users_in_train = set(user_ids[trainer.train_idx])
users_in_val = set(user_ids[trainer.val_idx])

with open("data/users_in_train.pkl", "wb") as f:
    pickle.dump(users_in_train, f)
with open("data/users_in_val.pkl", "wb") as f:
    pickle.dump(users_in_val, f)