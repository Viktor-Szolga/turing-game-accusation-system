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
    

class LinearClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear_layer = torch.nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return(self.linear_layer(x))