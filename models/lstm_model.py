import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_layer_size=50, output_size=1,num_layers=1, classification_mode=False):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.classification_mode = classification_mode
        if self.classification_mode:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size).to(x.device)

        out, _ = self.lstm(x, (h_0, c_0))
        out = self.linear(out[:, -1, :])
        if self.classification_mode:
            out = self.sigmoid(out)
        return out, _ # The second dimension is supposed to match the LSTM-AM Output Dimension
