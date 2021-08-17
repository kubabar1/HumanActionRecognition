import math
from collections import namedtuple
from typing import Tuple, Optional

import torch
from torch import Tensor
from torch.nn import Parameter, RNNCellBase, init

STLSTMState = namedtuple('STLSTMState', ['h_temp_prev', 'h_spat_prev', 'c_temp_prev', 'c_spat_prev'])


class STLSTMCell(RNNCellBase):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super(STLSTMCell, self).__init__(input_size, hidden_size, bias, num_chunks=5)
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.w_ih = Parameter(torch.randn(5 * hidden_size, input_size))
        self.w_hh0 = Parameter(torch.randn(5 * hidden_size, hidden_size))
        self.w_hh1 = Parameter(torch.randn(5 * hidden_size, hidden_size))
        self.b_ih = Parameter(torch.randn(5 * hidden_size))
        self.b_hh0 = Parameter(torch.randn(5 * hidden_size))
        self.b_hh1 = Parameter(torch.randn(5 * hidden_size))

        self.w_mx = Parameter(torch.randn(hidden_size, input_size))
        self.b_mx = Parameter(torch.randn(hidden_size))
        self.w_mp1 = Parameter(torch.randn(hidden_size, hidden_size))
        self.w_mp2 = Parameter(torch.randn(hidden_size, hidden_size))
        self.b_mp1 = Parameter(torch.randn(hidden_size))
        self.b_mp2 = Parameter(torch.randn(hidden_size))

        self.lbd = 0.5
        self.use_tau = True

        self.reset_parameters()

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

    def lstm_cell(self, input, state, w_ih, w_hh0, w_hh1, b_ih, b_hh0, b_hh1):
        # type: (Tensor, Tuple[Tensor, Tensor, Tensor, Tensor], Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
        h_temp_prev, h_spat_prev, c_temp_prev, c_spat_prev = state
        gates = (torch.mm(input, w_ih.t())
                 + torch.mm(h_spat_prev, w_hh0.t())
                 + torch.mm(h_temp_prev, w_hh1.t()))
        gates += b_ih + b_hh0 + b_hh1

        in_gate, forget_gate_s, forget_gate_t, out_gate, u_gate = gates.chunk(5, 1)

        in_gate = torch.sigmoid(in_gate)
        forget_gate_s = torch.sigmoid(forget_gate_s)
        forget_gate_t = torch.sigmoid(forget_gate_t)
        out_gate = torch.sigmoid(out_gate)
        u_gate = torch.tanh(u_gate)

        if self.use_tau:
            p = torch.mm(h_spat_prev, self.w_mp1.t()) + torch.mm(h_temp_prev, self.w_mp2.t()) + self.b_mp1 + self.b_mp2
            p = torch.tanh(p)

            x_prim = torch.tanh(torch.mm(input.clone(), self.w_mx.t())) + self.b_mx
            tau = self.G(x_prim - p, self.lbd)

            cy = (tau * in_gate * u_gate) + ((1 - tau) * forget_gate_s * c_spat_prev) + ((1 - tau) * forget_gate_t * c_temp_prev)
        else:
            cy = (in_gate * u_gate) + (forget_gate_s * c_spat_prev) + (forget_gate_t * c_temp_prev)
        hy = out_gate * torch.tanh(cy)

        return hy, cy

    def G(self, z, lbd):
        return torch.exp(-1 * lbd * z * z)
