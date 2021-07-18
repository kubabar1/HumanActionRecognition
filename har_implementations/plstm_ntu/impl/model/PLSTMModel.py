import torch
import torch.nn as nn
import torch.nn.functional as F

from .PLSTMCell import PLSTMCell


class PLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, classes_count, parts, dropout=0.5):
        super(PLSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.p_lstm_cell = PLSTMCell(input_size, hidden_size, parts)

        self.fc = torch.nn.Linear(hidden_size, classes_count)

    def forward(self, inputs):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        hn = torch.zeros(self.batch_size, self.hidden_size).to(device)
        cn = torch.zeros(self.batch_size, self.hidden_size).to(device)

        outs = []
        for seq in range(inputs.shape[2]):
            hn, cn = self.p_lstm_cell(inputs[:, :, seq, :], (hn, cn))
            outs.append(hn)

        out = outs[-1].squeeze()
        out = self.fc(out)

        return F.log_softmax(out, dim=-1)
