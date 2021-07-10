import torch
import torch.nn.functional as F
from torch.nn import Parameter

from .st_lstm.GCACell import GCACell
from .st_lstm.STLSTMCell import STLSTMState, STLSTMCell


def gca_loop(data, sequence_len, batch_size, hidden_size, iterations=4):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_size = 3
    joints_count = len(all_analysed_kpts)

    cell1 = STLSTMCell(input_size, hidden_size).to(device)
    cell2 = STLSTMCell(hidden_size, hidden_size).to(device)
    gca_cell = GCACell(hidden_size).to(device)

    spatial_dim = joints_count
    temporal_dim = sequence_len

    cell1_out = torch.empty((temporal_dim, spatial_dim, batch_size, hidden_size), dtype=torch.float, device=device)
    cell1_state_cell = torch.empty((temporal_dim, spatial_dim, batch_size, hidden_size), dtype=torch.float, device=device)

    for j in range(spatial_dim):
        for t in range(temporal_dim):
            if j == 0:
                h_spat_prev = torch.zeros(batch_size, hidden_size).to(device)
                c_spat_prev = torch.zeros(batch_size, hidden_size).to(device)
            else:
                h_spat_prev = cell1_out[t][j - 1]
                c_spat_prev = cell1_state_cell[t][j - 1]
            if t == 0:
                h_temp_prev = torch.zeros(batch_size, hidden_size).to(device)
                c_temp_prev = torch.zeros(batch_size, hidden_size).to(device)
            else:
                h_temp_prev = cell1_out[t - 1][j]
                c_temp_prev = cell1_state_cell[t - 1][j]
            state = STLSTMState(h_temp_prev, h_spat_prev, c_temp_prev, c_spat_prev)
            out, out_state = cell1(torch.tensor(data[t][j], dtype=torch.float, device=device), state)
            cell1_out[t][j] = out
            cell1_state_cell[t][j] = out_state[1]

    F_prev = torch.sum(cell1_out, dim=(0, 1)) / (spatial_dim * temporal_dim)

    for it in range(iterations):
        e_out = torch.empty((temporal_dim, spatial_dim, batch_size, hidden_size), dtype=torch.float, device=device)
        e_exp_sum = torch.empty((batch_size, hidden_size), dtype=torch.float, device=device)

        for j in range(spatial_dim):
            for t in range(temporal_dim):
                e = gca_cell(cell1_out[t][j], F_prev)
                e_out[t][j] = e
                e_exp_sum += torch.exp(e)

        r_out = torch.empty((temporal_dim, spatial_dim, batch_size, hidden_size), dtype=torch.float, device=device)

        for j in range(spatial_dim):
            for t in range(temporal_dim):
                r_out[t][j] = torch.exp(e_out[t][j]) / e_exp_sum

        cell2_out = torch.empty((temporal_dim, spatial_dim, batch_size, hidden_size), dtype=torch.float, device=device)
        cell2_state_cell = torch.empty((temporal_dim, spatial_dim, batch_size, hidden_size), dtype=torch.float, device=device)

        for j in range(spatial_dim):
            for t in range(temporal_dim):
                if j == 0:
                    h_spat_prev = torch.zeros(batch_size, hidden_size).to(device)
                    c_spat_prev = torch.zeros(batch_size, hidden_size).to(device)
                else:
                    h_spat_prev = cell2_out[t][j - 1]
                    c_spat_prev = cell2_state_cell[t][j - 1]
                if t == 0:
                    h_temp_prev = torch.zeros(batch_size, hidden_size).to(device)
                    c_temp_prev = torch.zeros(batch_size, hidden_size).to(device)
                else:
                    h_temp_prev = cell2_out[t - 1][j]
                    c_temp_prev = cell2_state_cell[t - 1][j]
                state = STLSTMState(h_temp_prev, h_spat_prev, c_temp_prev, c_spat_prev)
                out, out_state = cell2(cell1_out[t][j], state, r_out[t][j])
                cell2_out[t][j] = out
                cell2_state_cell[t][j] = out_state[1]

        F_prev = gca_cell.refine(out, F_prev, device)

    weight_c = Parameter(torch.randn(hidden_size, hidden_size)).to(device)
    bias_c = Parameter(torch.randn(hidden_size)).to(device)

    return F.log_softmax(torch.mm(F_prev, weight_c.t()) + bias_c, dim=1)
