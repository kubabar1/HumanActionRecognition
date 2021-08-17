from collections import namedtuple
from typing import Tuple, Optional

import torch
from torch import Tensor
from torch.nn import RNNCellBase

LSTMState = namedtuple('LSTMState', ['hx', 'cx'])


class LSTMCell(RNNCellBase):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super(LSTMCell, self).__init__(input_size, hidden_size, bias, num_chunks=4)

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
