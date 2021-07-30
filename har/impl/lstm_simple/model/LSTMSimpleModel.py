import torch
import torch.nn as nn
import torch.nn.functional as F


# from .LSTMCell import LSTMCell


class LSTMSimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, classes_count):
        super(LSTMSimpleModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size, 3, batch_first=True, dropout=0.5)
        self.fc = torch.nn.Linear(hidden_size, classes_count)
        # self.lstm_cell = torch.nn.LSTMCell(input_size, hidden_size)
        # self.lstm_cell_custom = LSTMCell(input_size, hidden_size)
        # self.dropout_l = nn.Dropout(0.5)

    def forward(self, inputs):
        lstm_out, _ = self.lstm(inputs)
        out = F.log_softmax(self.fc(lstm_out[:, -1, :]), dim=1)
        return out

    # def forward(self, inputs):
    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     hn = torch.zeros(self.batch_size, self.hidden_size).to(device)
    #     cn = torch.zeros(self.batch_size, self.hidden_size).to(device)
    #     outs = []
    #     for seq in range(inputs.shape[1]):
    #         hn, cn = self.lstm_cell(inputs[:, seq, :], (hn, cn))
    #         outs.append(hn)
    #     out = outs[-1].squeeze()
    #     out = self.fc(out)
    #     out = self.dropout_l(out)
    #     return F.log_softmax(out, dim=-1)
