from collections import namedtuple
from typing import Tuple, List

import torch
import torch.nn as nn
from torch import Tensor, jit
from torch.nn import Parameter

STLSTMState = namedtuple('STLSTMState', ['h0', 'h1', 'c0', 'c1'])
LSTMState = namedtuple('LSTMState', ['hx', 'cx'])


class GCACell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(GCACell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(5 * hidden_size, input_size))
        self.weight_hh0 = Parameter(torch.randn(5 * hidden_size, hidden_size))
        self.weight_hh1 = Parameter(torch.randn(5 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(5 * hidden_size))
        self.bias_hh0 = Parameter(torch.randn(5 * hidden_size))
        self.bias_hh1 = Parameter(torch.randn(5 * hidden_size))

    @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        h0, h1, c0, c1 = state

        gates = (torch.mm(input, self.weight_ih.t()) + torch.mm(h0, self.weight_hh0.t()) + torch.mm(h1, self.weight_hh1.t()))
        gates += self.bias_ih + self.bias_hh0 + self.bias_hh1
        in_gate, forget_gate_s, forget_gate_t, out_gate, u_gate = gates.chunk(5, 1)

        in_gate = torch.sigmoid(in_gate)
        forget_gate_s = torch.sigmoid(forget_gate_s)
        forget_gate_t = torch.sigmoid(forget_gate_t)
        out_gate = torch.sigmoid(out_gate)
        u_gate = torch.tanh(u_gate)

        cy = (in_gate * u_gate) + (forget_gate_s * c0) + (forget_gate_t * c1)
        hy = out_gate * torch.tanh(cy)

        return hy, (hy, cy)


class STLSTMLayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(STLSTMLayer, self).__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(self, input: Tensor, state_prev: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        inputs = input.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        state = (state_prev[1], state_prev[3])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state_prev)
            outputs += [out]
        return torch.stack(outputs), state


def test_script_rnn_layer(seq_len, batch, input_size, hidden_size):
    inp = torch.randn(seq_len, batch, input_size)
    state = STLSTMState(torch.randn(batch, hidden_size),
                        torch.randn(batch, hidden_size),
                        torch.randn(batch, hidden_size),
                        torch.randn(batch, hidden_size))
    st_lstm = STLSTMLayer(STLSTMCell, input_size, hidden_size)
    out, out_state = st_lstm(inp, state)

    # Control: pytorch native LSTM
    lstm = nn.LSTM(input_size, hidden_size, 1)
    lstm_state = LSTMState(state.h1.unsqueeze(0), state.c1.unsqueeze(0))
    # for lstm_param, custom_param in zip(lstm.all_weights[0], st_lstm.parameters()):
    #     assert lstm_param.shape == custom_param.shape
    #     with torch.no_grad():
    #         lstm_param.copy_(custom_param)
    lstm_out, lstm_out_state = lstm(inp, lstm_state)

    # print(out)
    # print(lstm_out)
    # print((out.cpu().detach().numpy() - lstm_out.cpu().detach().numpy()))

    # assert (out - lstm_out).abs().max() < 1e-5
    # assert (out_state[0] - lstm_out_state[0]).abs().max() < 1e-5
    # assert (out_state[1] - lstm_out_state[1]).abs().max() < 1e-5


test_script_rnn_layer(5, 2, 3, 7)
