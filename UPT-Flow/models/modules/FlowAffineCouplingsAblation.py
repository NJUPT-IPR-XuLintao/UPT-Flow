
import torch
from torch import nn as nn
import torch.nn.functional as F
from models.modules import thops
from models.modules.flow import Conv2d, Conv2dZeros
from utils.util import opt_get


class CondAffineSeparatedAndCond(nn.Module):
    def __init__(self, in_channels, opt,fFeatures_firstConv):
        super().__init__()
        self.need_features = True
        self.in_channels = in_channels
        self.in_channels_rrdb = opt_get(opt, ['network_G', 'flow', 'conditionInFeaDim'], 320)####需要修改
        self.kernel_hidden = 1
        self.affine_eps = 0.0003
        self.n_hidden_layers = 1
        hidden_channels = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'hidden_channels'])
        self.hidden_channels = 64 if hidden_channels is None else hidden_channels
        self.fFeatures_firstConv=fFeatures_firstConv
        self.affine_eps = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'eps'], 0.0003)

        self.channels_for_nn = self.in_channels // 2
        self.channels_for_co = self.in_channels - self.channels_for_nn

        if self.channels_for_nn is None:
            self.channels_for_nn = self.in_channels // 2

        self.fAffine_1 = NN_F(in_channels=self.channels_for_nn + fFeatures_firstConv,
                              out_channels=self.channels_for_co * 2,
                              hidden_channels=self.hidden_channels,
                              kernel_hidden=self.kernel_hidden,
                              n_hidden_layers=self.n_hidden_layers)

        self.fAffine_2 = NN_F(in_channels=self.channels_for_nn + fFeatures_firstConv,
                              out_channels=self.channels_for_co * 2,
                              hidden_channels=self.hidden_channels,
                              kernel_hidden=self.kernel_hidden,
                              n_hidden_layers=self.n_hidden_layers)

        self.fFeatures_1 = NN_F(in_channels=fFeatures_firstConv,
                                out_channels=self.in_channels * 2,
                                hidden_channels=self.hidden_channels,
                                kernel_hidden=self.kernel_hidden,
                                n_hidden_layers=self.n_hidden_layers)

        self.fFeatures_2 = NN_F(in_channels=fFeatures_firstConv,
                                out_channels=self.in_channels * 2,
                                hidden_channels=self.hidden_channels,
                                kernel_hidden=self.kernel_hidden,
                                n_hidden_layers=self.n_hidden_layers)

        self.opt = opt
        self.le_curve = opt['le_curve'] if opt['le_curve'] is not None else False
        if self.le_curve:
            self.fCurve = self.F(in_channels=self.in_channels_rrdb,
                                 out_channels=self.in_channels,
                                 hidden_channels=self.hidden_channels,
                                 kernel_hidden=self.kernel_hidden,
                                 n_hidden_layers=self.n_hidden_layers)

    def forward(self, input: torch.Tensor, logdet=None, reverse=False, ft=None):
        if not reverse:
            z = input
            assert z.shape[1] == self.in_channels, (z.shape[1], self.in_channels)

            # Feature Conditional
            scaleFt_1, shiftFt_1 = self.feature_extract(ft, self.fFeatures_1)
            z = z + shiftFt_1
            z = z * scaleFt_1
            logdet = logdet + self.get_logdet(scaleFt_1)
            # Self Conditional
            z1, z2 = self.split(z)
            scale_1, shift_1 = self.feature_extract_aff(z1, ft, self.fAffine_1)
            self.asserts(scale_1, shift_1, z1, z2)
            z2 = z2 + shift_1
            z2 = z2 * scale_1
            logdet = logdet + self.get_logdet(scale_1)
            z = thops.cat_feature(z1, z2)

            # Feature Conditional
            scaleFt_2, shiftFt_2 = self.feature_extract(ft, self.fFeatures_2)
            z = z + shiftFt_2
            z = z * scaleFt_2
            logdet = logdet + self.get_logdet(scaleFt_2)
            # Self Conditional
            z1, z2 = self.split(z)
            scale_2, shift_2 = self.feature_extract_aff(z2, ft, self.fAffine_2)
            self.asserts(scale_2, shift_2, z1, z2)
            z1 = z1 + shift_2
            z1 = z1 * scale_2
            logdet = logdet + self.get_logdet(scale_2)
            z = thops.cat_feature(z1, z2)

            output = z
        else:
            z = input

            # Self Conditional
            z1, z2 = self.split(z)
            scale_2, shift_2 = self.feature_extract_aff(z2, ft, self.fAffine_2)  #fAffine_2=384->192
            self.asserts(scale_2, shift_2, z1, z2)
            z1 = z1 / scale_2
            z1 = z1 - shift_2
            z = thops.cat_feature(z1, z2)  #192*2=384
            logdet = logdet - self.get_logdet(scale_2)
            # Feature Conditional
            scaleFt_2, shiftFt_2 = self.feature_extract(ft, self.fFeatures_2)  #fFeatures_2=288->384,384/2=192
            z = z / scaleFt_2
            z = z - shiftFt_2
            logdet = logdet - self.get_logdet(scaleFt_2)

            # Self Conditional
            z1, z2 = self.split(z)
            scale_1, shift_1 = self.feature_extract_aff(z1, ft, self.fAffine_1)
            self.asserts(scale_1, shift_1, z1, z2)
            z2 = z2 / scale_1
            z2 = z2 - shift_1
            z = thops.cat_feature(z1, z2)
            logdet = logdet - self.get_logdet(scale_2)
            # Feature Conditional
            scaleFt_1, shiftFt_1 = self.feature_extract(ft, self.fFeatures_1)
            z = z / scaleFt_1
            z = z - shiftFt_1
            logdet = logdet - self.get_logdet(scaleFt_1)

            output = z
        return output, logdet


    def asserts(self, scale, shift, z1, z2):
        assert z1.shape[1] == self.channels_for_nn, (z1.shape[1], self.channels_for_nn)
        assert z2.shape[1] == self.channels_for_co, (z2.shape[1], self.channels_for_co)
        assert scale.shape[1] == shift.shape[1], (scale.shape[1], shift.shape[1])
        assert scale.shape[1] == z2.shape[1], (scale.shape[1], z1.shape[1], z2.shape[1])

    def get_logdet(self, scale):
        return thops.sum(torch.log(scale), dim=[1, 2, 3])

    def feature_extract(self, z, f):

        h = f(z)  #z=8,320,96,96, h=8,24,96,96
        #h = h + z
        shift, scale = thops.split_feature(h, "cross")   #s、c=8,12,96,96
        scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        return scale, shift


    def feature_extract_aff(self, z1, ft, f):
        z = torch.cat([z1, ft], dim=1)#第二层次，通道数是48，一半是24，所以是320+24=344
        h = f(z)

        shift, scale = thops.split_feature(h, "cross")
        scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        return scale, shift

    def split(self, z):
        z1 = z[:, :self.channels_for_nn]
        z2 = z[:, self.channels_for_nn:]
        assert z1.shape[1] + z2.shape[1] == z.shape[1], (z1.shape[1], z2.shape[1], z.shape[1])
        return z1, z2


    def F(self, in_channels, out_channels, hidden_channels, kernel_hidden=1, n_hidden_layers=1):
        layers = [Conv2d(in_channels, hidden_channels), nn.ReLU(inplace=False)]

        for _ in range(n_hidden_layers):
            layers.append(Conv2d(hidden_channels, hidden_channels, kernel_size=[kernel_hidden, kernel_hidden]))
            layers.append(nn.ReLU(inplace=False))
        layers.append(Conv2dZeros(hidden_channels, out_channels))

        return nn.Sequential(*layers)

###############################################################################################################
class NN_F(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_hidden=1, n_hidden_layers=1):
        super(NN_F, self).__init__()
        layers = [Conv2d(in_channels, hidden_channels, kernel_size=kernel_hidden), nn.ReLU(inplace=False)]

        for _ in range(n_hidden_layers):
            layers.append(Conv2d(hidden_channels, hidden_channels))
            layers.append(nn.ReLU(inplace=False))  #
        layers.append(CBAM(gate_channels=hidden_channels))
        layers.append(Conv2dZeros(hidden_channels, out_channels, kernel_size=kernel_hidden))

        self.model = nn.Sequential(*layers)
        # self.shortcut = Conv2dZeros(in_channels, out_channels, kernel_size=kernel_hidden)
        # self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):

        return self.model(x)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        # self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        # if self.bn is not None:
        #     x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=('avg', 'max')):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        channel_att_raw = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=('avg', 'max'), no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out + x


if __name__ == '__main__':
    mode = NN_F(384, 288, 64)
    x = torch.randn([1, 384, 128, 128])
    y = mode(x)
    print(mode)
    print(y)
    print("Parameters of full network %.4f " % (sum([m.numel() for m in mode.parameters()]) / 1e6))

