import torch.nn as nn
import torch.nn.functional as F


class MLPModel(nn.Module):
    def __init__(self, input_size=150, hidden_layer_size=50, output_size=1, num_layers=2, classification_mode=False):
        super(MLPModel, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.dummy = 1

        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_layer_size, hidden_layer_size) for _ in range(num_layers - 1)
        ])

        self.output_layer = nn.Linear(hidden_layer_size, output_size)
        self.classification_mode = classification_mode

        if self.classification_mode:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))

        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        x = self.output_layer(x)
        if self.classification_mode:
            x = self.sigmoid(x)
        y = self.dummy

        return x, y

