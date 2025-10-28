import torch
import torch.nn as nn

class MessageClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.75):
        super().__init__()
        self.activation = nn.ReLU()
        layers = []
        current_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(torch.nn.Linear(current_size, hidden_size))
            layers.append(torch.nn.BatchNorm1d(hidden_size))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))
            current_size = hidden_size
        layers.append(torch.nn.Linear(current_size, output_size))

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class TemperatureScaledMessageClassifier(nn.Module):
    def __init__(self, model: MessageClassifier):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, inputs):
        logits = self.model(inputs)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        # Divide logits by temperature
        return logits / self.temperature

class EnsembleMessageClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, model_paths, dropout=0.75):
        super().__init__()
        self.models = nn.ModuleList()
        for model_path in model_paths:
            model = MessageClassifier(input_size, hidden_sizes, output_size, dropout)
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            model.eval()
            self.models.append(model)

    def forward(self, x):
        all_logits = [model(x) for model in self.models]  # list of (B,C)
        stacked = torch.stack(all_logits, dim=0)          # (M,B,C)
        ensembled_logits = torch.mean(stacked, dim=0)    # (B,C)
        return ensembled_logits

class TemperatureScaledEnsemble(nn.Module):
    def __init__(self, model: EnsembleMessageClassifier):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, inputs):
        logits = self.model(inputs)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        # Divide logits by temperature
        return logits / self.temperature
    

class VectorScaledEnsemble(nn.Module):
    def __init__(self, model: nn.Module, num_classes: int):
        super().__init__()
        self.model = model
        self.W = nn.Parameter(torch.eye(num_classes))
        self.b = nn.Parameter(torch.zeros(num_classes))

    def forward(self, inputs):
        logits = self.model(inputs)
        return self.vector_scale(logits)

    def vector_scale(self, logits):
        return logits @ self.W.t() + self.b

class PlattScaledMessageClassifier(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        logits = self.model(inputs)
        return self.platt_scale(logits)

    def platt_scale(self, logits):
        return torch.sigmoid(self.a * logits + self.b)

class LinearClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear_layer = torch.nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return(self.linear_layer(x))