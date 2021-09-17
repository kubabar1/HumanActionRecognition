import math
from collections import namedtuple
from typing import Tuple, Optional

import torch
from torch import Tensor
from torch.nn import Parameter, Module, init

STLSTMState = namedtuple('STLSTMState', ['h_temp_prev', 'c_temp_prev', 'h_spat_prev', 'c_spat_prev'])


class STLSTMCell(Module):
    def __init__(self, input_size: int, hidden_size: int, use_tau=False, bias: bool = True, lbd=0.5, num_chunks=5) -> None:
        super(STLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_chunks = num_chunks
        self.bias = bias

        self.w_ih = Parameter(torch.Tensor(num_chunks * hidden_size, input_size))
        self.w_hh0 = Parameter(torch.Tensor(num_chunks * hidden_size, hidden_size))
        self.w_hh1 = Parameter(torch.Tensor(num_chunks * hidden_size, hidden_size))
        self.b_ih = Parameter(torch.Tensor(num_chunks * hidden_size))
        self.b_hh0 = Parameter(torch.Tensor(num_chunks * hidden_size))
        self.b_hh1 = Parameter(torch.Tensor(num_chunks * hidden_size))

        self.w_mx = Parameter(torch.Tensor(hidden_size, input_size))
        self.b_mx = Parameter(torch.Tensor(hidden_size))
        self.w_mp1 = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.w_mp2 = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_mp1 = Parameter(torch.Tensor(hidden_size))
        self.b_mp2 = Parameter(torch.Tensor(hidden_size))

        self.lbd = lbd
        self.use_tau = use_tau

        self.reset_parameters()

    def check_forward_input(self, input: Tensor) -> None:
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def check_forward_hidden(self, input: Tensor, hx: Tensor, hidden_label: str = '') -> None:
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size))

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, input: Tensor, state: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        self.check_forward_input(input)
        self.check_forward_hidden(input, state[0], '[0]')
        self.check_forward_hidden(input, state[1], '[1]')
        self.check_forward_hidden(input, state[2], '[2]')
        self.check_forward_hidden(input, state[3], '[3]')
        return self.lstm_cell(
            input, state,
            self.w_ih, self.w_hh0, self.w_hh1,
            self.b_ih, self.b_hh0, self.b_hh1,
        )

    def lstm_cell(self, input: Tensor, state: Tuple[Tensor, Tensor, Tensor, Tensor], w_ih: Tensor, w_hh0: Tensor, w_hh1: Tensor,
                  b_ih: Tensor, b_hh0: Tensor, b_hh1: Tensor) -> Tuple[Tensor, Tensor]:
        h_temp_prev, c_temp_prev, h_spat_prev, c_spat_prev = state
        gates = (torch.mm(input, w_ih.t()) + torch.mm(h_spat_prev, w_hh0.t()) + torch.mm(h_temp_prev, w_hh1.t()))
        if self.bias:
            gates += b_ih + b_hh0 + b_hh1

        in_gate, forget_gate_s, forget_gate_t, out_gate, u_gate = gates.chunk(self.num_chunks, 1)

        in_gate = torch.sigmoid(in_gate)
        forget_gate_s = torch.sigmoid(forget_gate_s)
        forget_gate_t = torch.sigmoid(forget_gate_t)
        out_gate = torch.sigmoid(out_gate)
        u_gate = torch.tanh(u_gate)

        if self.use_tau:
            p = torch.mm(h_spat_prev, self.w_mp1.t()) + torch.mm(h_temp_prev, self.w_mp2.t())
            if self.bias:
                p += self.b_mp1 + self.b_mp2
            p = torch.tanh(p)

            x_prim = torch.mm(input, self.w_mx.t())
            if self.bias:
                x_prim += self.b_mx
            x_prim = torch.tanh(x_prim)

            tau = self.G(x_prim - p, self.lbd)

            cy = (tau * in_gate * u_gate) + ((1 - tau) * forget_gate_s * c_spat_prev) + ((1 - tau) * forget_gate_t * c_temp_prev)
        else:
            cy = (in_gate * u_gate) + (forget_gate_s * c_spat_prev) + (forget_gate_t * c_temp_prev)

        hy = out_gate * torch.tanh(cy)

        return hy, cy

    def G(self, z, lbd):
        return torch.exp(-1 * lbd * z * z)
