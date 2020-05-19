import torch.nn as nn
import torch


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()

        self.net1 = nn.Conv2d(1, 64, 9, 1, padding=0)
        self.net2 = nn.Conv2d(64, 32, 1, 1, padding=0)
        self.net3 = nn.Conv2d(32, 1, 5, 1, padding=0)
        self.relu = nn.ReLU()

        self.load_state_dict(torch.load("../pretrain/SRCNNx2.pt"))
        print('load SRCNN pretrain!')

    def forward(self, x):
        net1_x = self.relu(self.net1(x))
        net2_x = self.relu(self.net2(net1_x))
        output = self.net3(net2_x)

        return output
