import numpy as np
import torch

from .st_lstm.STLSTMCell import STLSTMState, STLSTMCell
from .utils import get_batch


def train(iterations=4, batch_size=5, hidden_size=128, sequence_len=140):
    input_size = 3
    analysed_kpts_left = [4, 5, 6, 11, 12, 13]
    analysed_kpts_right = [1, 2, 3, 14, 15, 16]
    all_analysed_kpts = analysed_kpts_left + analysed_kpts_right
    joints_count = len(all_analysed_kpts)
    dataset_path = '../../datasets/berkeley_mhad/3d'
    data, labels = get_batch(dataset_path, sequence_len=sequence_len, batch_size=batch_size, training=True)
    data = data[:, :, all_analysed_kpts, :]
    data = np.transpose(data, (1, 2, 0, 3))  # [frame, joint, batch, channel]

    cell1 = STLSTMCell(input_size, hidden_size)

    spatial_dim = joints_count
    temporal_dim = sequence_len

    cell1_out = torch.empty((temporal_dim, spatial_dim, batch_size, hidden_size), dtype=torch.float)

    for j in range(spatial_dim):
        for t in range(temporal_dim):
            if j == 0:
                h_spat_prev = torch.zeros(batch_size, hidden_size)
                c_spat_prev = torch.zeros(batch_size, hidden_size)
            else:
                h_spat_prev = cell1_out[t][j - 1]
                c_spat_prev = cell1_out[t][j - 1]
            if t == 0:
                h_temp_prev = torch.zeros(batch_size, hidden_size)
                c_temp_prev = torch.zeros(batch_size, hidden_size)
            else:
                h_temp_prev = cell1_out[t - 1][j]
                c_temp_prev = cell1_out[t - 1][j]
            state = STLSTMState(h_temp_prev, h_spat_prev, c_temp_prev, c_spat_prev)
            out, out_state = cell1(torch.tensor(data[t][j], dtype=torch.float), state)
            cell1_out[t][j] = out

    print(data.shape)
    print(cell1_out.shape)
    # print(cell1_out[t][j].shape)

    print(torch.sum(cell1_out, dim=(0, 1)).shape)
    print(torch.sum(cell1_out, dim=(0, 1)) / (spatial_dim * temporal_dim))

    # cell1 = STLSTMCell(input_size, hidden_size)
    # # cell2 = STLSTMCell(input_size, hidden_size)
    # # gca_cells = [GCACell(lstm_size, it) for it in range(1, iterations + 1)]
    #
    # t_0 = data[0][0]
    # j_0_t_0 = t_0[0]
    #
    # inp = torch.tensor([data[0][0][0]])  # , data[1][0][0], data[2][0][0], data[3][0][0]
    #
    # # inp = torch.randn(seq_len, batch, input_size)
    # # print(inp.shape)
    # # print(data[0].shape)
    #
    # out, out_state = cell1(inp, state)
    #
    # print(inp.shape)
    # print(out.shape)
    # print(len(data))
    # print(data[0].shape)
    # # print(out_state[0].shape)
    # # print(out_state[1].shape)
