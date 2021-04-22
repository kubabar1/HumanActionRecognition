import torch
import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, layer_num, n_steps):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, layer_num, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.bn = nn.BatchNorm1d(n_steps)

    def forward(self, inputs):
        x = self.bn(inputs)
        lstm_out, (hn, cn) = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out
