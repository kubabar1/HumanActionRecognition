import math
from collections import namedtuple
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, jit
from torch.nn import Parameter, RNNCellBase

LSTMState = namedtuple('LSTMState', ['hx', 'cx'])


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        # self.c2c = Tensor(hidden_size * 3)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        hx, cx = hidden

        x = x.view(-1, x.size(1))

        gates = self.x2h(x) + self.h2h(hx)

        gates = gates.squeeze()

        # c2c = self.c2c.unsqueeze(0)
        # ci, cf, co = c2c.chunk(3, 1)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        # ci = ci.to(device)
        # cf = cf.to(device)
        # co = co.to(device)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = forgetgate * cx + ingate * torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        hm = outgate * F.tanh(cellgate)
        return (hm, cellgate)


class LSTMCell2(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        # nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))

    @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        # gates = self.weight_ih(input) + self.weight_hh(hx)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class LSTMCell3(RNNCellBase):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super(LSTMCell3, self).__init__(input_size, hidden_size, bias, num_chunks=4)

    def lstm_cell(self, input, hidden, w_ih, w_hh, b_ih, b_hh):
        # type: (Tensor, Tuple[Tensor, Tensor], Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
        hx, cx = hidden
        gates = torch.mm(input, w_ih.t()) + torch.mm(hx, w_hh.t()) + b_ih + b_hh

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, cy

    def forward(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        self.check_forward_input(input)
        if hx is None:
            zeros = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        self.check_forward_hidden(input, hx[0], '[0]')
        self.check_forward_hidden(input, hx[1], '[1]')
        return self.lstm_cell(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )


class LSTMLayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(LSTMLayer, self).__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        inputs = input.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state


def test_script_rnn_layer(seq_len, batch, input_size, hidden_size):
    inp = torch.randn(seq_len, batch, input_size)
    state = LSTMState(torch.randn(batch, hidden_size),
                      torch.randn(batch, hidden_size))
    rnn = LSTMLayer(LSTMCell, input_size, hidden_size)
    out, out_state = rnn(inp, state)

    # Control: pytorch native LSTM
    lstm = nn.LSTM(input_size, hidden_size, 1, dropout=0.5)
    lstm_state = LSTMState(state.hx.unsqueeze(0), state.cx.unsqueeze(0))
    for lstm_param, custom_param in zip(lstm.all_weights[0], rnn.parameters()):
        assert lstm_param.shape == custom_param.shape
        with torch.no_grad():
            lstm_param.copy_(custom_param)
    lstm_out, lstm_out_state = lstm(inp, lstm_state)

    assert (out - lstm_out).abs().max() < 1e-5
    assert (out_state[0] - lstm_out_state[0]).abs().max() < 1e-5
    assert (out_state[1] - lstm_out_state[1]).abs().max() < 1e-5


def test_script_rnn_layer_2():
    import numpy as np

    def get_batch(shilouetes_berkeley_path, batch_size=128, training=True, train_threshold=0.8, shilouetes_count=8,
                  actions_count=11, repetitions_count=5, split_t=20):
        from random import randrange
        from pathlib import Path
        import os
        shilouetes_dirs = sorted(
            [os.path.join(shilouetes_berkeley_path, x.name) for x in Path(shilouetes_berkeley_path).iterdir() if x.is_dir()])
        coordinates_file_name = '3d_coordinates.npy'
        train_size = int(shilouetes_count * train_threshold)
        train_shilouetes = range(train_size)
        test_shilouetes = range(shilouetes_count - train_size)

        data = []
        labels = []
        for i in range(batch_size):
            rand_shilouete_id = randrange(shilouetes_count)
            rand_action_id = randrange(actions_count)
            rand_repetition_id = randrange(repetitions_count)
            coordinates_path = os.path.join(shilouetes_dirs[rand_shilouete_id],
                                            'a' + str(rand_action_id + 1).zfill(2),
                                            'r' + str(rand_repetition_id + 1).zfill(2),
                                            coordinates_file_name)
            data.append(np.array([a[randrange(len(a))] for a in np.array_split(np.load(coordinates_path), split_t)]))

            # if len(pos) < sequence_len:
            #     tmp = []
            #     rp = sequence_len // len(pos)
            #     lt = sequence_len - len(pos) * rp
            #     for _ in range(rp):
            #         tmp.extend(pos)
            #     tmp.extend(pos[:lt])
            #     data.append(np.array(tmp))
            # else:
            #     data.append(pos[:sequence_len])
            labels.append(rand_action_id)

        return np.array(data, dtype='float'), labels

    def get_data(dataset_path, batch_size, analysed_kpts):
        data, labels = get_batch(dataset_path, batch_size=batch_size, training=True)
        data = data[:, :, analysed_kpts, :]
        return np.transpose(data, (1, 2, 0, 3)), labels  # [frame, joint, batch, channels]

    batch_size = 5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    analysed_kpts_left = [4, 5, 6, 11, 12, 13]
    analysed_kpts_right = [1, 2, 3, 14, 15, 16]
    all_analysed_kpts = analysed_kpts_left + analysed_kpts_right
    dataset_path = '../../../../datasets/berkeley_mhad/3d'
    data, train_y = get_data(dataset_path, batch_size, all_analysed_kpts)

    data = np.transpose(data, (2, 0, 1, 3))
    data = data.reshape((batch_size, 20, -1))
    data = torch.tensor(data, dtype=torch.float, device=device)

    lstm = nn.LSTM(36, 128, 1).to(device)
    lstm_out, _ = lstm(data)
    lstm_out = lstm_out[:, -1, :]

    hn = torch.zeros(batch_size, 128).to(device)
    cn = torch.zeros(batch_size, 128).to(device)
    lstm_cell_custom = LSTMCell3(36, 128).to(device)
    state = LSTMState(hn, cn)
    outs = []
    for seq in range(data.shape[1]):
        hn, cn = lstm_cell_custom(data[:, seq, :], state)
        outs.append(hn)
    lstm_out_custom = outs[-1].squeeze()

    print((lstm_out_custom).abs().max())
    print((lstm_out).abs().max())

    assert (lstm_out_custom - lstm_out).abs().max() < 1e-5
    # assert (out_state[0] - lstm_out_state[0]).abs().max() < 1e-5
    # assert (out_state[1] - lstm_out_state[1]).abs().max() < 1e-5


# test_script_rnn_layer(5, 2, 3, 128)
# test_script_rnn_layer_2()
