import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalRNNModel(nn.Module):
    def __init__(self, hidden_size, classes_count):
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

        return F.log_softmax(out, dim=-1)
