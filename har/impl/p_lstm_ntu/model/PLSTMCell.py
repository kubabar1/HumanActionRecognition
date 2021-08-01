import math
from collections import namedtuple
from typing import Tuple, Optional

import torch
from torch import Tensor
from torch.nn import Module, init, Parameter, ParameterList

LSTMState = namedtuple('LSTMState', ['hx', 'cx'])


class PLSTMCell(Module):
    def __init__(self, input_size: int, hidden_size: int, parts: int, num_chunks: int = 3) -> None:
        super(PLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.parts = parts
        self.num_chunks = num_chunks

        self.weight_px = ParameterList([Parameter(torch.Tensor(num_chunks * hidden_size, input_size)) for _ in range(parts)])
        self.bias_px = ParameterList([Parameter(torch.Tensor(num_chunks * hidden_size)) for _ in range(parts)])
        self.weight_ph = ParameterList([Parameter(torch.Tensor(num_chunks * hidden_size, hidden_size)) for _ in range(parts)])
        self.bias_ph = ParameterList([Parameter(torch.Tensor(num_chunks * hidden_size)) for _ in range(parts)])

        self.weight_ox = Parameter(torch.Tensor(hidden_size, input_size))
        self.bias_ox = Parameter(torch.Tensor(hidden_size))
        self.weight_oh = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_oh = Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight_arr in [self.weight_px, self.bias_px, self.weight_ph, self.bias_ph]:
            for weight in weight_arr:
                init.uniform_(weight, -stdv, stdv)

        for weight in [self.weight_ox, self.bias_ox, self.weight_oh, self.bias_oh]:
            init.uniform_(weight, -stdv, stdv)

    def forward(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        return self.lstm_cell(input, hx)

    def lstm_cell(self, input, hidden):
        input = input.chunk(self.parts, dim=1)

        hx, cx = hidden
        cx = torch.chunk(cx, self.parts, dim=1)
        cy = []

        for it, p in enumerate(input):
            gates = torch.mm(p, self.weight_px[it].t()) + torch.mm(hx, self.weight_ph[it].t())
            gates += self.bias_px[it] + self.bias_ph[it]
            ingate, forgetgate, cellgate = gates.chunk(3, 1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            cy.append((forgetgate * cx[it]) + (ingate * cellgate))
        output_gate = torch.stack([torch.mm(x, self.weight_ox.t()) + self.bias_ox for x in input], dim=0).sum(0)
        output_gate += torch.mm(hx, self.weight_oh.t()) + self.bias_oh
        outgate = torch.sigmoid(output_gate)

        hy = outgate * torch.tanh(torch.stack(cy, dim=0).sum(0))

        return hy, torch.cat(cy, dim=1)
