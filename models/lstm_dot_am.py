import torch
import torch.nn as nn
import torch.nn.functional as F


class DotProductAttention(nn.Module):
    def __init__(self, hidden_size):
        super(DotProductAttention, self).__init__()
        self.hidden_size = hidden_size

    def forward(self, lstm_output):

        batch_size, seq_len, hidden_size = lstm_output.size()

        scores = torch.bmm(lstm_output, lstm_output.transpose(1, 2))

        attention_weights = F.softmax(scores, dim=2)

        context_vector = torch.bmm(attention_weights, lstm_output)[:, -1, :]

        return context_vector, attention_weights


class LSTMModelWithDotAttention(nn.Module):
    def __init__(self, input_size=5, hidden_layer_size=50, output_size=1, num_layers=1, classification_mode=False):
        super(LSTMModelWithDotAttention, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, batch_first=True)
        self.attention = DotProductAttention(hidden_layer_size)
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
