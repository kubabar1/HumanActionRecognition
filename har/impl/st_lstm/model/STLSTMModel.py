import torch
import torch.nn as nn
import torch.nn.functional as F

from .STLSTMCell import STLSTMCell


class STLSTMModel(nn.Module):
    def __init__(self, input_size, joints_count, hidden_size, classes_count, dropout=0.5, use_tau=False, bias=True, lbd=0.5,
                 use_two_layers=True):
        super(STLSTMModel, self).__init__()
        self.use_two_layers = use_two_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.st_lstm_cell_list_1 = nn.ModuleList(
            [STLSTMCell(input_size, hidden_size, use_tau, bias, lbd) for _ in range(joints_count)])
        self.st_lstm_cell_list_2 = nn.ModuleList(
            [STLSTMCell(input_size, hidden_size, use_tau, bias, lbd) for _ in range(joints_count)])
        self.dropout_l = nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_size, classes_count)

    def forward(self, input):
        batch_size, temporal_dim, spatial_dim, _ = input.shape

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        hn_next = torch.zeros(batch_size, self.hidden_size).to(device)
        cn_next = torch.zeros(batch_size, self.hidden_size).to(device)

        hn_spat_arr = [torch.zeros(batch_size, self.hidden_size).to(device) for _ in range(temporal_dim)]
        cn_spat_arr = [torch.zeros(batch_size, self.hidden_size).to(device) for _ in range(temporal_dim)]
        output_arr = []

        for j in range(spatial_dim):
            output_arr_tmp = []
            for t in range(temporal_dim):
                input_frame = input[:, t, j, :]

                hn_next, cn_next = self.st_lstm_cell_list_1[j](input_frame, (hn_next, cn_next, hn_spat_arr[t], cn_spat_arr[t]))
                if self.use_two_layers:
                    hn_next = self.dropout_l(hn_next)
                    hn_next, cn_next = self.st_lstm_cell_list_2[j](input_frame,
                                                                   (hn_next, cn_next, hn_spat_arr[t], cn_spat_arr[t]))

                output_arr_tmp.append(F.log_softmax(self.fc(hn_next), dim=-1))

                hn_spat_arr[t] = hn_next
                cn_spat_arr[t] = cn_next
            output_arr.append(output_arr_tmp)

        return torch.stack([i[-1] for i in output_arr])[-1]
