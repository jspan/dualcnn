import torch.nn as nn
from srcnn import SRCNN
from vdsr import VDSR


class DUAL_CNN_DENOISE(nn.Module):
    def __init__(self):
        super(DUAL_CNN_DENOISE, self).__init__()

        self.structure_net = SRCNN()
        self.detail_net = VDSR()

    def forward(self, x):
        structure = self.structure_net(x)
        detail = self.detail_net(x)
        detail = detail[:, :, 6:-6, 6:-6]
        output = structure + detail

        return output, structure
