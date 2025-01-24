import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        seq_len = lstm_output.size(1)
        scores = self.Va(torch.tanh(self.Wa(lstm_output) + self.Ua(lstm_output)))
        attention_weights = F.softmax(scores, dim=1)

        context_vector = torch.bmm(attention_weights.permute(0, 2, 1), lstm_output).squeeze(1)
        return context_vector, attention_weights


class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_layer_size=50, output_size=1, num_layers=1, classification_mode=False):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, batch_first=True)
        self.attention = Attention(hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.classification_mode = classification_mode
        if self.classification_mode:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size).to(x.device)

        lstm_output, _ = self.lstm(x, (h_0, c_0))

        context_vector, attention_weights = self.attention(lstm_output)

        out = self.linear(context_vector)
        if self.classification_mode:
            out = self.sigmoid(out)

        return out, attention_weights
