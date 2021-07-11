import torch
import torch.nn as nn
import torch.nn.functional as F

from .LSTMCell import LSTMCell, LSTMState, LSTMCell3


class LSTMSimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, classes_count, dropout=0.5):
        super(LSTMSimpleModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lstm_cell_custom = LSTMCell3(input_size, hidden_size)
        self.lstm_cell = torch.nn.LSTMCell(input_size, hidden_size)
        self.lstm = torch.nn.LSTM(input_size, hidden_size, 1, batch_first=True, dropout=0.5)
        self.fc = torch.nn.Linear(hidden_size, classes_count)
        self.dropout_l = nn.Dropout(dropout)

    def forward2(self, inputs):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # out, out_state = self.rnn(input_joint, state)
        # h_next = out
        # c_next = out_state[1]
        # return h_next, c_next, F.log_softmax(self.fc(out[-1, :, :]), dim=-1)
        hn = torch.zeros(self.batch_size, self.hidden_size).to(device)
        cn = torch.zeros(self.batch_size, self.hidden_size).to(device)
        state = LSTMState(hn, cn)
        outs = []
        for seq in range(inputs.shape[1]):
            hn, cn = self.lstm_cell_custom(inputs[:, seq, :], state)
            outs.append(hn)
        out = outs[-1].squeeze()
        out = self.fc(out)
        # out = self.dropout_l(out)
        return F.log_softmax(out, dim=-1)

    def forward(self, inputs):
        lstm_out, _ = self.lstm(inputs)
        out = F.log_softmax(self.fc(lstm_out[:, -1, :]), dim=1)
        return out

    def forward3(self, inputs):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # out, out_state = self.rnn(input_joint, state)
        # h_next = out
        # c_next = out_state[1]
        # return h_next, c_next, F.log_softmax(self.fc(out[-1, :, :]), dim=-1)
        hn = torch.zeros(self.batch_size, self.hidden_size).to(device)
        cn = torch.zeros(self.batch_size, self.hidden_size).to(device)
        state = LSTMState(hn, cn)
        outs = []
        for seq in range(inputs.shape[1]):
            hn, cn = self.lstm_cell(inputs[:, seq, :], state)
            outs.append(hn)
        out = outs[-1].squeeze()
        out = self.fc(out)
        # out = self.dropout_l(out)
        return F.log_softmax(out, dim=-1)
