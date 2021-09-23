import torch
import torch.nn as nn
import torch.nn.functional as F

from .PLSTMCell import PLSTMCell


class PLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, classes_count, parts=4):
        super(PLSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.parts = parts

        self.parts = parts

        self.p_lstm_cell_1 = PLSTMCell(input_size, hidden_size, parts)
        self.p_lstm_cell_2 = PLSTMCell(input_size, hidden_size, parts)

        self.dropout = nn.Dropout(0.5)

        self.fc = torch.nn.Linear(hidden_size, classes_count)

    def forward(self, inputs):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        batch_size = inputs.shape[1]
        hn_1 = torch.zeros(batch_size, self.hidden_size).to(device)
        cn_1 = torch.zeros(batch_size, self.parts * self.hidden_size).to(device)

        outs = []

        for seq in range(inputs.shape[2]):
            input_t = inputs[:, :, seq, :]
            input_t = torch.cat([i for i in input_t], dim=1)
            hn_1, cn_1 = self.p_lstm_cell_1(input_t, (hn_1, cn_1))
            hn_1 = self.dropout(hn_1)
            hn_1, cn_1 = self.p_lstm_cell_2(input_t, (hn_1, cn_1))
            outs.append(hn_1)

        out = outs[-1].squeeze()
        out = self.fc(out)

        return F.log_softmax(out, dim=-1)
