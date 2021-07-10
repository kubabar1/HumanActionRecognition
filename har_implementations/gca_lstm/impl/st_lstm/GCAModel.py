import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from .GCACell import GCACell
from .STLSTMCell import STLSTMCell, STLSTMState


class GCAModel(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_len, iterations, classes_count):
        super(GCAModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.iterations = iterations
        self.sequence_len = sequence_len
        self.cell1 = STLSTMCell(input_size, hidden_size)
        self.cell2 = STLSTMCell(hidden_size, hidden_size)
        self.gca_cell = GCACell(hidden_size)
        self.weight_F_a = Parameter(torch.randn(self.hidden_size, self.hidden_size))
        self.weight_F_prev = Parameter(torch.randn(self.hidden_size, self.hidden_size))
        self.bias_F_a = Parameter(torch.randn(self.hidden_size))
        self.bias_F_prev = Parameter(torch.randn(self.hidden_size))
        self.weight_c = Parameter(torch.randn(hidden_size, classes_count))
        self.bias_c = Parameter(torch.randn(classes_count))

    def refine(self, F_a: Tensor, F_prev: Tensor) -> Tensor:
        F_next = torch.mm(F_a, self.weight_F_a.t()) + torch.mm(F_prev, self.weight_F_prev.t()) + self.bias_F_a + self.bias_F_prev
        return torch.relu_(F_next)

    def forward(self, data, batch_size, device):
        joints_count = data.shape[1]
        spatial_dim = joints_count
        temporal_dim = self.sequence_len

        cell1_out = torch.empty((temporal_dim, spatial_dim, batch_size, self.hidden_size), dtype=torch.float, device=device)
        cell1_state_cell = torch.empty((temporal_dim, spatial_dim, batch_size, self.hidden_size), dtype=torch.float,
                                       device=device)

        for j in range(spatial_dim):
            for t in range(temporal_dim):
                if j == 0:
                    h_spat_prev = torch.zeros(batch_size, self.hidden_size).to(device)
                    c_spat_prev = torch.zeros(batch_size, self.hidden_size).to(device)
                else:
                    h_spat_prev = cell1_out[t][j - 1]
                    c_spat_prev = cell1_state_cell[t][j - 1]
                if t == 0:
                    h_temp_prev = torch.zeros(batch_size, self.hidden_size).to(device)
                    c_temp_prev = torch.zeros(batch_size, self.hidden_size).to(device)
                else:
                    h_temp_prev = cell1_out[t - 1][j]
                    c_temp_prev = cell1_state_cell[t - 1][j]
                state = STLSTMState(h_temp_prev, h_spat_prev, c_temp_prev, c_spat_prev)
                out, out_state = self.cell1(torch.tensor(data[t][j], dtype=torch.float, device=device), state)
                cell1_out[t][j] = out
                cell1_state_cell[t][j] = out_state[1]

        F_prev = torch.sum(cell1_out, dim=(0, 1)) / (spatial_dim * temporal_dim)

        # for it in range(self.iterations):
        #     e_out = torch.empty((temporal_dim, spatial_dim, batch_size, self.hidden_size), dtype=torch.float, device=device)
        #     e_exp_sum = torch.empty((batch_size, self.hidden_size), dtype=torch.float, device=device)
        #
        #     for j in range(spatial_dim):
        #         for t in range(temporal_dim):
        #             e = self.gca_cell(cell1_out[t][j], F_prev)
        #             e_out[t][j] = e
        #             e_exp_sum += torch.exp(e)
        #
        #     r_out = torch.empty((temporal_dim, spatial_dim, batch_size, self.hidden_size), dtype=torch.float, device=device)
        #
        #     for j in range(spatial_dim):
        #         for t in range(temporal_dim):
        #             r_out[t][j] = torch.exp(e_out[t][j]) / e_exp_sum
        #
        #     cell2_out = torch.empty((temporal_dim, spatial_dim, batch_size, self.hidden_size), dtype=torch.float, device=device)
        #     cell2_state_cell = torch.empty((temporal_dim, spatial_dim, batch_size, self.hidden_size), dtype=torch.float,
        #                                    device=device)
        #
        #     for j in range(spatial_dim):
        #         for t in range(temporal_dim):
        #             if j == 0:
        #                 h_spat_prev = torch.zeros(batch_size, self.hidden_size).to(device)
        #                 c_spat_prev = torch.zeros(batch_size, self.hidden_size).to(device)
        #             else:
        #                 h_spat_prev = cell2_out[t][j - 1]
        #                 c_spat_prev = cell2_state_cell[t][j - 1]
        #             if t == 0:
        #                 h_temp_prev = torch.zeros(batch_size, self.hidden_size).to(device)
        #                 c_temp_prev = torch.zeros(batch_size, self.hidden_size).to(device)
        #             else:
        #                 h_temp_prev = cell2_out[t - 1][j]
        #                 c_temp_prev = cell2_state_cell[t - 1][j]
        #             state = STLSTMState(h_temp_prev, h_spat_prev, c_temp_prev, c_spat_prev)
        #             out, out_state = self.cell2(cell1_out[t][j], state, r_out[t][j])
        #             cell2_out[t][j] = out
        #             cell2_state_cell[t][j] = out_state[1]
        #
        #     F_prev = self.refine(out, F_prev)

        return F.log_softmax(torch.mm(F_prev, self.weight_c.t()) + self.bias_c, dim=1)
