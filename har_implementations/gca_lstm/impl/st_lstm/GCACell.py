import torch
from torch import Tensor, jit
from torch.nn import Parameter


class GCACell(jit.ScriptModule):
    def __init__(self, hidden_size):
        super(GCACell, self).__init__()
        self.hidden_size = hidden_size
        self.weight_e2_h = Parameter(torch.randn(hidden_size, hidden_size))
        self.weight_e2_f = Parameter(torch.randn(hidden_size, hidden_size))
        self.weight_e1 = Parameter(torch.randn(hidden_size, hidden_size))
        self.bias_e2_h = Parameter(torch.randn(hidden_size))
        self.bias_e2_f = Parameter(torch.randn(hidden_size))
        self.bias_e1 = Parameter(torch.randn(hidden_size))

    @jit.script_method
    def forward(self, h: Tensor, F: Tensor) -> Tensor:
        e = (torch.mm(h, self.weight_e2_h.t()) + torch.mm(F, self.weight_e2_f.t()))
        e += self.bias_e2_h + self.bias_e2_f

        e = torch.mm(torch.tanh(e), self.weight_e1)
        e += self.bias_e1
        return e

