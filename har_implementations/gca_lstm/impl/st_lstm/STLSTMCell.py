from collections import namedtuple
from typing import Tuple, Optional

import torch
from torch import Tensor, jit
from torch.nn import Parameter

STLSTMState = namedtuple('STLSTMState', ['h_temp_prev', 'h_spat_prev', 'c_temp_prev', 'c_spat_prev'])


class STLSTMCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(STLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(5 * hidden_size, input_size))
        self.weight_hh0 = Parameter(torch.randn(5 * hidden_size, hidden_size))
        self.weight_hh1 = Parameter(torch.randn(5 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(5 * hidden_size))
        self.bias_hh0 = Parameter(torch.randn(5 * hidden_size))
        self.bias_hh1 = Parameter(torch.randn(5 * hidden_size))

    @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor, Tensor, Tensor], r: Optional[Tensor] = None) -> Tuple[
        Tensor, Tuple[Tensor, Tensor]]:
        h_temp_prev, h_spat_prev, c_temp_prev, c_spat_prev = state

        # print('###################################')
        # print(self.input_size)
        # print(self.hidden_size)
        # print(input.shape)
        # print(self.weight_ih.t().shape)
        # print(torch.mm(input, self.weight_ih.t()).shape)
        # print('###################################')
        gates = (torch.mm(input, self.weight_ih.t())
                 + torch.mm(h_spat_prev, self.weight_hh0.t())
                 + torch.mm(h_temp_prev, self.weight_hh1.t()))
        gates += self.bias_ih + self.bias_hh0 + self.bias_hh1
        in_gate, forget_gate_s, forget_gate_t, out_gate, u_gate = gates.chunk(5, 1)

        in_gate = torch.sigmoid(in_gate)
        forget_gate_s = torch.sigmoid(forget_gate_s)
        forget_gate_t = torch.sigmoid(forget_gate_t)
        out_gate = torch.sigmoid(out_gate)
        u_gate = torch.tanh(u_gate)

        if r is None:
            cy = (in_gate * u_gate) + (forget_gate_s * h_spat_prev) + (forget_gate_t * h_temp_prev)
        else:
            cy = (r * in_gate * u_gate) + ((1 - r) * forget_gate_s * c_spat_prev) + ((1 - r) * forget_gate_t * c_temp_prev)

        hy = out_gate * torch.tanh(cy)

        return hy, (hy, cy)
