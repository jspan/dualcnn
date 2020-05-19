import torch.nn as nn
import torch
from math import sqrt


def make_model(args):
    return Dual_cnn(args=args)


class SRCNN_Structure(nn.Module):

    def __init__(self):
        super(SRCNN_Structure, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=0, bias=True)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0, bias=True)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=0, bias=True)

    def forward(self, input):
        out_1 = self.relu(self.conv1(input))
        out_2 = self.relu(self.conv2(out_1))
        out = self.conv3(out_2)
        return out


class conv_relu_block(nn.Module):
    def __init__(self):
        super(conv_relu_block, self).__init__()
        self.conv = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, input):
        return self.relu(self.conv(input))


class VDSR_Detail(nn.Module):

    def __init__(self):
        super(VDSR_Detail, self).__init__()

        self.input = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
        self.output = nn.Conv2d(64, 1, 3, 1, 1, bias=False)
        self.relu = nn.ReLU()
        self.residual_layer = self.make_layer(conv_relu_block, 18)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layer = []
        for _ in range(num_of_layer):
            layer.append(block())
        return nn.Sequential(*layer)

    def forward(self, input):

        out = self.relu(self.input(input))
        out = self.residual_layer(out)
        out = self.output(out)

        return out


class Dual_cnn(nn.Module):

    def __init__(self, args):
        super(Dual_cnn, self).__init__()
        self.args = args
        self.srcnn_structure = SRCNN_Structure()

        self.vdsr_detail = VDSR_Detail()
        if not args.quickly_test:

            print('loading srcnn pre-trained model ...')
            srcnn_weights = torch.load('../pretrain/matlab/SRCNN/SRCNNx{}.pt'.format(self.args.scale))
            self.srcnn_structure.load_state_dict(srcnn_weights, strict=False)

            print('loading vdsr pre-trained model ...')
            vdsr_weights = torch.load('../pretrain/matlab/VDSR/VDSR.pt')
            self.vdsr_detail.load_state_dict(vdsr_weights, strict=False)

    def forward(self, input):
        structure = self.srcnn_structure(input)
        detail = self.vdsr_detail(input)[:, :, 6:-6, 6:-6]
        output = structure + detail
        return structure, detail, output
