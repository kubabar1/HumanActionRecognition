import torch
import torch.nn as nn
import torch.nn.functional as F


class MultilayerLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MultilayerLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, 3, batch_first=True, dropout=0.5)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):
        lstm_out, _ = self.lstm(inputs)
        out = F.log_softmax(self.fc(lstm_out[:, -1, :]), dim=1)
        return out
