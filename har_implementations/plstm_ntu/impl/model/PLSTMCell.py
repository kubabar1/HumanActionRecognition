import math
from collections import namedtuple
from typing import Tuple, Optional

import torch
from torch import Tensor
from torch.nn import RNNCellBase, init, Parameter, ParameterList

LSTMState = namedtuple('LSTMState', ['hx', 'cx'])


class PLSTMCell(RNNCellBase):
    def __init__(self, input_size: int, hidden_size: int, parts: int, bias: bool = True) -> None:
        super(PLSTMCell, self).__init__(input_size, hidden_size, bias, num_chunks=3)
        self.weight_ox = ParameterList([Parameter(torch.randn(hidden_size, input_size)) for _ in range(parts)])
        self.bias_ox = ParameterList([Parameter(torch.randn(hidden_size)) for _ in range(parts)])
        self.weight_oh = Parameter(torch.randn(hidden_size, hidden_size))
        self.bias_oh = Parameter(torch.randn(hidden_size))
        # self.reset_parameters2()

    # def reset_parameters2(self) -> None:
    #     stdv = 1.0 / math.sqrt(self.hidden_size)
    #     for weight in [self.weight_oh, self.bias_oh]:
    #         init.uniform_(weight, -stdv, stdv)
    #     for weight_arr in [self.weight_oh, self.bias_oh]:
    #         for weight in weight_arr:
    #             init.uniform_(weight, -stdv, stdv)

    def forward(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        for inp in input:
            self.check_forward_input(inp)
        if hx is None:
            zeros = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        for inp in input:
            self.check_forward_hidden(inp, hx[0], '[0]')
            self.check_forward_hidden(inp, hx[1], '[1]')
        return self.lstm_cell(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )

    def lstm_cell(self, input, hidden, w_ih, w_hh, b_ih, b_hh):
        hx, cx = hidden

        tmp = None
        cy = []
        for it, p in enumerate(input):
            gates = torch.mm(p, w_ih.t()) + torch.mm(hx, w_hh.t()) + b_ih + b_hh
            ingate, forgetgate, cellgate = gates.chunk(3, 1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            cy.append((forgetgate * cx) + (ingate * cellgate))
            if tmp is None:
                tmp = torch.mm(p, self.weight_ox[it].t()) + self.bias_ox[it]
            else:
                tmp += torch.mm(p, self.weight_ox[it].t()) + self.bias_ox[it]
        tmp += torch.mm(hx, self.weight_oh.t()) + self.bias_oh
        outgate = torch.sigmoid(tmp)
        hy = outgate * torch.tanh(torch.stack(cy, dim=0).sum(dim=0))

        return hy, torch.stack(cy, dim=0).sum(dim=0)
