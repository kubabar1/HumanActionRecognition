import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class MainLSTMNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, classes_count, dropout=0.5):
        super(MainLSTMNNModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lstm_cell_t = torch.nn.LSTMCell(input_size, hidden_size)

        self.lstm_cell1 = torch.nn.LSTMCell(input_size, hidden_size)
        self.lstm_cell2 = torch.nn.LSTMCell(hidden_size, hidden_size)
        self.lstm_cell3 = torch.nn.LSTMCell(hidden_size, hidden_size)
        self.fc = torch.nn.Linear(hidden_size, classes_count)

        self.lstm_cell = torch.nn.LSTMCell(input_size, hidden_size)

        self.W_x_t = Parameter(torch.randn(hidden_size, hidden_size))
        self.W_h_t = Parameter(torch.randn(hidden_size, hidden_size))

        self.fc_t = torch.nn.Linear(input_size, hidden_size)
        self.relu_t = torch.nn.ReLU()

        self.b_t = Parameter(torch.randn(hidden_size))

        self.W_x_s = Parameter(torch.randn(hidden_size, hidden_size))
        self.W_h_s = Parameter(torch.randn(hidden_size, hidden_size))
        self.U_s = Parameter(torch.randn(hidden_size, hidden_size))

        self.b_s = Parameter(torch.randn(hidden_size))
        self.b_us = Parameter(torch.randn(hidden_size))

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, int(input_size / 3))

    def forward(self, inputs, epoch):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        hn_t = torch.zeros(self.batch_size, self.hidden_size).to(device)
        cn_t = torch.zeros(self.batch_size, self.hidden_size).to(device)

        hn_s = torch.zeros(self.batch_size, self.hidden_size).to(device)
        cn_s = torch.zeros(self.batch_size, self.hidden_size).to(device)

        hn1 = torch.zeros(self.batch_size, self.hidden_size).to(device)
        cn1 = torch.zeros(self.batch_size, self.hidden_size).to(device)
        hn2 = torch.zeros(self.batch_size, self.hidden_size).to(device)
        cn2 = torch.zeros(self.batch_size, self.hidden_size).to(device)
        hn3 = torch.zeros(self.batch_size, self.hidden_size).to(device)
        cn3 = torch.zeros(self.batch_size, self.hidden_size).to(device)

        outs = []
        alpha_arr = []
        beta_arr = []
        for seq in range(inputs.shape[1]):
            h_prev_temp = hn_t
            hn_t, cn_t = self.lstm_cell_t(inputs[:, seq, :], (hn_t, cn_t))
            beta = torch.matmul(self.fc_t(inputs[:, seq, :]), self.W_x_t) + torch.matmul(h_prev_temp, self.W_h_t) + self.b_s
            beta += self.b_t
            beta = self.relu_t(beta)

            # if 100 < epoch < 1100:
            #     for param in self.lstm_cell.parameters():
            #         param.requires_grad = False
            #     for param in self.fc1.parameters():
            #         param.requires_grad = False
            #     for param in self.fc2.parameters():
            #         param.requires_grad = False
            # elif epoch >= 1100:
            #     for param in self.lstm_cell.parameters():
            #         param.requires_grad = True
            #     for param in self.fc1.parameters():
            #         param.requires_grad = True
            #     for param in self.fc2.parameters():
            #         param.requires_grad = True

            h_prev = hn_s
            hn_s, cn_s = self.lstm_cell(inputs[:, seq, :], (hn_s, cn_s))
            st = torch.tanh(torch.matmul(self.fc1(inputs[:, seq, :]), self.W_x_s) + torch.matmul(h_prev, self.W_h_s) + self.b_s)
            st = torch.matmul(st, self.U_s)
            st += self.b_us

            q = self.fc2(st)
            alpha = torch.exp(q) / torch.exp(q).sum(dim=1).unsqueeze(dim=1)

            # if epoch < 100:
            #     hn1, cn1 = self.lstm_cell1(inputs[:, seq, :] * torch.repeat_interleave(alpha, 3, dim=1), (hn1, cn1))
            #     outs.append(hn1)
            # else:
            hn1, cn1 = self.lstm_cell1(inputs[:, seq, :] * torch.repeat_interleave(alpha, 3, dim=1), (hn1, cn1))
            # hn2, cn2 = self.lstm_cell2(hn1, (hn2, cn2))
            # hn3, cn3 = self.lstm_cell3(hn1, (hn3, cn3))

            zt = hn1
            zt *= beta

            outs.append(zt)
            alpha_arr.append(alpha)
            beta_arr.append(beta)

        # print(torch.stack(alpha_arr, dim=0).reshape((inputs.shape[0], inputs.shape[1], -1)).shape)
        # print(alpha.shape)
        # print(alpha_arr[0])
        # s = []
        # for seq in range(inputs.shape[1]):
        #     inpts = self.fc1(inputs[:, seq, :])
        #     st = torch.tanh(torch.matmul(inpts, self.W_x_s) + torch.matmul(outs1[seq], self.W_h_s) + self.b_s)
        #     st = torch.matmul(st, self.U_s)
        #     st += self.b_us
        #     s.append(st)
        #
        # alpha_arr = []
        # for seq in range(len(s)):
        #     q = self.fc2(s[seq])
        #     alpha = torch.exp(q) / torch.stack([torch.exp(i) for i in q], dim=0).sum(dim=1).unsqueeze(dim=1)
        #     alpha_arr.append(alpha)
        #
        # hn = torch.zeros(self.batch_size, self.hidden_size).to(device)
        # cn = torch.zeros(self.batch_size, self.hidden_size).to(device)
        #
        # for t in range(inputs.shape[1]):
        #     for k in range(inputs.shape[2]):
        #         inputs[:, t, k] *= alpha_arr[:, t, int(k / 3)]
        #
        # outs = []
        # for seq in range(inputs.shape[1]):
        #     if single:
        #         hn, cn = self.lstm_cell1(inputs[:, seq, :], (hn, cn))
        #     else:
        #         hn, cn = self.lstm_cell1(inputs[:, seq, :], (hn, cn))
        #         hn, cn = self.lstm_cell2(hn, (hn, cn))
        #         hn, cn = self.lstm_cell2(hn, (hn, cn))
        #     outs.append(hn)

        output = self.fc(outs[-1].squeeze())

        return F.log_softmax(output, dim=-1), \
               torch.stack(alpha_arr, dim=0).reshape((inputs.shape[0], inputs.shape[1], -1)), \
               torch.stack(beta_arr, dim=0).reshape((inputs.shape[0], inputs.shape[1], -1))  # [batch, sequence, joint_alpha]
