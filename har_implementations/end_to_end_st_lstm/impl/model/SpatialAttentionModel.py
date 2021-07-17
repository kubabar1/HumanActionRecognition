import torch
import torch.nn as nn
from torch.nn import Parameter


class SpatialAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, classes_count, dropout=0.5):
        super(SpatialAttentionModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lstm_cell = torch.nn.LSTMCell(input_size, hidden_size)

        self.W_x_s = Parameter(torch.randn(hidden_size, input_size))
        self.W_h_s = Parameter(torch.randn(hidden_size, input_size))
        self.U_s = Parameter(torch.randn(input_size, input_size))
        self.b_s = Parameter(torch.randn(input_size))
        self.b_us = Parameter(torch.randn(input_size))

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(input_size, int(input_size / 3))

    def forward(self, inputs):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        hn1 = torch.zeros(self.batch_size, self.hidden_size).to(device)
        cn1 = torch.zeros(self.batch_size, self.hidden_size).to(device)

        outs1 = []
        for seq in range(inputs.shape[1]):
            hn1, cn1 = self.lstm_cell(inputs[:, seq, :], (hn1, cn1))
            outs1.append(hn1)

        s = []
        for seq in range(inputs.shape[1]):
            inpts = self.fc1(inputs[:, seq, :])
            st = torch.tanh(torch.matmul(inpts, self.W_x_s) + torch.matmul(outs1[seq], self.W_h_s) + self.b_s)
            st = torch.matmul(st, self.U_s)
            st += self.b_us
            s.append(st)

        alpha_arr = []
        for seq in range(len(s)):
            q = self.fc2(s[seq])
            alpha = torch.exp(q) / torch.stack([torch.exp(i) for i in q], dim=0).sum(dim=1).unsqueeze(dim=1)
            alpha_arr.append(alpha)

        return torch.stack(alpha_arr, dim=0).reshape((inputs.shape[0], inputs.shape[1], -1))  # [batch, sequence, joint_alpha]
