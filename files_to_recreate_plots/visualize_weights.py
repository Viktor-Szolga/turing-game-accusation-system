import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models import MessageClassifier, LinearClassifier
from src.utils import visualize_weights
import torch

root_folder = "gridsearch"

# Linear Classifier
model = LinearClassifier(1024, 2)
model.load_state_dict(torch.load(os.path.join(root_folder, "trained_models", "run293.pth"), weights_only=True))

# Model B dropout and weight decay
model = MessageClassifier(input_size=1024, hidden_sizes=[48, 24], output_size=2)
model.load_state_dict(torch.load(os.path.join(root_folder, "trained_models", "run191.pth"), weights_only=True))

# No regularization Model B
model = MessageClassifier(input_size=1024, hidden_sizes=[48, 24], output_size=2)
model.load_state_dict(torch.load(os.path.join(root_folder, "trained_models", "run030.pth"), weights_only=True))

# Linear Classifier
model = LinearClassifier(1024, 2)
model.load_state_dict(torch.load(os.path.join(root_folder, "trained_models", "run293.pth"), weights_only=True))

# Model B high weight decay
model = MessageClassifier(input_size=1024, hidden_sizes=[48, 24], output_size=2)
model.load_state_dict(torch.load(os.path.join(root_folder, "trained_models", "run031.pth"), weights_only=True))

# Model B 0.65 dropout
model = MessageClassifier(input_size=1024, hidden_sizes=[48, 24], output_size=2)
model.load_state_dict(torch.load(os.path.join(root_folder, "trained_models", "run080.pth"), weights_only=True))

# Model B  0.95 dropout
model = MessageClassifier(input_size=1024, hidden_sizes=[48, 24], output_size=2)
model.load_state_dict(torch.load(os.path.join(root_folder, "trained_models", "run083.pth"), weights_only=True))

# Model B 0.85 dropout
model = MessageClassifier(input_size=1024, hidden_sizes=[48, 24], output_size=2)
model.load_state_dict(torch.load(os.path.join(root_folder, "trained_models", "run082.pth"), weights_only=True))


visualize_weights(model)