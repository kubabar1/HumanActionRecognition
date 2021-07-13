import torch
import torch.nn as nn
import torch.nn.functional as F

from .STLSTMCell import STLSTMCell


class TrustGatesSTLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, dropout, classes_count):
        super(TrustGatesSTLSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.st_lstm_cell_1 = STLSTMCell(input_size, hidden_size)
        self.st_lstm_cell_2 = STLSTMCell(hidden_size, hidden_size)
        self.fc = torch.nn.Linear(hidden_size, classes_count)
        self.dropout_l = nn.Dropout(dropout)

    def forward(self, input, state):
        h_next, c_next = self.st_lstm_cell_1(input, state)
        h_next = self.dropout_l(h_next)
        out, c_next = self.st_lstm_cell_2(h_next, state)
        out = self.fc(out)
        return h_next, c_next, F.log_softmax(out, dim=-1)
