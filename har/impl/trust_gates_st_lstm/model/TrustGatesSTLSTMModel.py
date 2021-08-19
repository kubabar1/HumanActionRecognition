import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .STLSTMCell import STLSTMCell


class TrustGatesSTLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, classes_count, spatial_dim, temporal_dim, criterion, dropout=0.5):
        super(TrustGatesSTLSTMModel, self).__init__()
        self.criterion = criterion

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim

        self.lstm_cell = nn.LSTMCell(36, hidden_size)
        self.st_lstm_cell_1 = STLSTMCell(input_size, hidden_size)
        # self.st_lstm_cell_2 = STLSTMCell(hidden_size, hidden_size)
        self.fc = torch.nn.Linear(hidden_size, classes_count)
        # self.dropout_l = nn.Dropout(dropout)

    def forward(self, input, tensor_train_y):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        hn_next = torch.zeros(self.batch_size, self.hidden_size).to(device)
        cn_next = torch.zeros(self.batch_size, self.hidden_size).to(device)

        hn_spat_arr = [torch.zeros(self.batch_size, self.hidden_size).to(device) for _ in range(self.temporal_dim)]
        cn_spat_arr = [torch.zeros(self.batch_size, self.hidden_size).to(device) for _ in range(self.temporal_dim)]
        output_arr = []
        losses_arr = []

        # for t in range(input.shape[1]):
        #     hn_next, cn_next = self.lstm_cell(input[:, t, :, :].reshape(128, 36), (hn_next, cn_next))
        #     out = F.log_softmax(self.fc(hn_next), dim=-1)

        for j in range(input.shape[2]):
            model_input = input[:, :, j, :]

            for t in range(model_input.shape[1]):
                hn_next, cn_next = self.st_lstm_cell_1(model_input[:, t, :], (hn_next, cn_next, hn_spat_arr[t], cn_spat_arr[t]))
                out = F.log_softmax(self.fc(hn_next), dim=-1)
                losses_arr.append(self.criterion(out, tensor_train_y))
                output_arr.append(out)
                hn_spat_arr[t] = hn_next
                cn_spat_arr[t] = cn_next

        #return out, self.criterion(out, tensor_train_y)

        return torch.mean(torch.stack(output_arr), dim=0), torch.mean(torch.stack(losses_arr))
