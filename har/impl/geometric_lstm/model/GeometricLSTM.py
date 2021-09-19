import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometricLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, hidden_layers=3, dropout=0.5):
        super(GeometricLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, hidden_layers, batch_first=True, dropout=dropout)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):
        lstm_out = self.lstm(inputs)[0]
        return F.log_softmax(self.fc(lstm_out[:, -1, :]), dim=1)
