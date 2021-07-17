import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalRNNModel(nn.Module):
    def __init__(self, hidden_size, batch_size, classes_count, dropout=0.5):
        super(HierarchicalRNNModel, self).__init__()
        self.hidden_size = hidden_size

        self.bl1_left_arm = torch.nn.RNN(3 * 3, hidden_size, bidirectional=True, batch_first=True)
        self.bl1_right_arm = torch.nn.RNN(3 * 3, hidden_size, bidirectional=True, batch_first=True)
        self.bl1_left_leg = torch.nn.RNN(3 * 3, hidden_size, bidirectional=True, batch_first=True)
        self.bl1_right_leg = torch.nn.RNN(3 * 3, hidden_size, bidirectional=True, batch_first=True)

        self.bl2_arm = torch.nn.RNN(4 * hidden_size, hidden_size, bidirectional=True, batch_first=True)
        self.bl2_leg = torch.nn.RNN(4 * hidden_size, hidden_size, bidirectional=True, batch_first=True)

        self.bl3_all = torch.nn.LSTM(4 * hidden_size, hidden_size, bidirectional=True, batch_first=True)

        self.fc = torch.nn.Linear(2 * hidden_size, classes_count)

        # self.lstm_all = torch.nn.RNN(8 * hidden_size, 4 * hidden_size, bidirectional=True, batch_first=True)

        # self.lstm = torch.nn.LSTM(input_size, hidden_size, 1, batch_first=True)  # , dropout=0.5
        # self.fc = torch.nn.Linear(hidden_size, classes_count)
        # self.dropout_l = nn.Dropout(dropout)

    def forward(self, inputs):
        out1_left_arm, _ = self.bl1_left_arm(inputs[0])
        out1_right_arm, _ = self.bl1_right_arm(inputs[1])
        out1_left_leg, _ = self.bl1_left_leg(inputs[2])
        out1_right_leg, _ = self.bl1_right_leg(inputs[3])

        out2_arm, _ = self.bl2_arm(torch.cat([out1_left_arm, out1_right_arm], dim=2))
        out2_leg, _ = self.bl2_arm(torch.cat([out1_left_leg, out1_right_leg], dim=2))

        out3_all, _ = self.bl3_all(torch.cat([out2_arm, out2_leg], dim=2))

        fc_all = self.fc(out3_all)

        out = torch.sum(fc_all, dim=1)

        # print(inputs[0].shape)
        # print(out2_arm.shape)
        # print(out2_leg.shape)
        # print(hidden2_arm.shape)
        # print(hidden2_leg.shape)
        # print(out3_all.shape)
        # print(fc_all.shape)
        # print(out.shape)
        # print(F.log_softmax(out, dim=-1)[0])
        # print(F.softmax(out, dim=-1)[0])

        return F.log_softmax(out, dim=-1)

        # # out, out_state = self.rnn(input_joint, state)
        # # h_next = out
        # # c_next = out_state[1]
        # # return h_next, c_next, F.log_softmax(self.fc(out[-1, :, :]), dim=-1)
        # hn = torch.zeros(self.batch_size, self.hidden_size).to(device)
        # cn = torch.zeros(self.batch_size, self.hidden_size).to(device)
        # outs = []
        # for seq in range(inputs.shape[1]):
        #     hn, cn = self.lstm_cell(inputs[:, seq, :], (hn, cn))
        #     outs.append(hn)
        # out = outs[-1].squeeze()
        # out = self.fc(out)
        # # out = self.dropout_l(out)
        # return F.log_softmax(out, dim=-1)
