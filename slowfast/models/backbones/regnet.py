import math
import torch
import torch.nn as nn

from ..build import MODEL_REGISTRY
import slowfast.utils.weight_init_helper as init_helper
from slowfast.models.batchnorm_helper import get_norm


_CFG = {
    "400M": {
        'd': [1, 2, 7, 12],
        'wi': [32, 64, 160, 384],
        'g': 16,
        'b': 1,
        'w0': 24
    },
    "4G": {
        'd': [2, 5, 14, 2],
        'wi': [80, 240, 560, 1360],
        'g': 24,
        'b': 1,
        'w0': 48
    }
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, temporal_k=1, temporal_p=0):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(temporal_k, 3, 3), stride=(1, stride, stride),
                     padding=(temporal_p, 1, 1), groups=groups, bias=False, dilation=1)


def conv1x1(in_planes, out_planes, stride=1, temporal_k=1, temporal_p=0):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(temporal_k, 1, 1), stride=(1, stride, stride),
                     padding=(temporal_p, 0, 0), bias=False)


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, bottle_ratio=1, temporal_k=1, temporal_p=0):
        super(Bottleneck, self).__init__()
        norm_layer = nn.BatchNorm3d
        intra_plane = planes // bottle_ratio

        self.conv1 = conv1x1(inplanes, intra_plane,
                             temporal_k=temporal_k, temporal_p=temporal_p)
        self.bn1 = norm_layer(intra_plane)
        self.conv2 = conv3x3(intra_plane, intra_plane, stride, groups)
        self.bn2 = norm_layer(intra_plane)
        self.conv3 = conv1x1(intra_plane, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.relu_final = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if identity.shape != out.shape:
            identity = identity[:, :, 1:-1]
        out += identity

        out = self.relu_final(out)

        return out


@MODEL_REGISTRY.register()
class RegNet(nn.Module):
    def __init__(self, cfg, zero_init_residual=True):
        super(RegNet, self).__init__()

        self.cfg = cfg
        self.model_cfg = _CFG[self.cfg.REGNET.DEPTH]
        self.model_cfg['sa'] = [0, 0, 0, 0]  # FIXME
        if self.cfg.PGT.ENABLE:
            temporal_p = 0
        else:
            temporal_p = 1
        self.conv1 = conv3x3(3, self.model_cfg['w0'], stride=2)
        self.bn1 = nn.BatchNorm3d(self.model_cfg['w0'])
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(
            self.model_cfg['w0'], self.model_cfg['wi'][0], self.model_cfg['d'][0], self.model_cfg['sa'][0])
        self.layer2 = self._make_layer(
            self.model_cfg['wi'][0], self.model_cfg['wi'][1], self.model_cfg['d'][1], self.model_cfg['sa'][1])

        self.layer3 = self._make_layer(self.model_cfg['wi'][1], self.model_cfg['wi'][2], self.model_cfg['d'][2], self.model_cfg['sa'][2],
                                       temporal_k=3, temporal_p=temporal_p)
        self.layer4 = self._make_layer(self.model_cfg['wi'][2], self.model_cfg['wi'][3], self.model_cfg['d'][3], self.model_cfg['sa'][3],
                                       temporal_k=3, temporal_p=temporal_p)

        # input shape of each temporal layer, [channel, spatial stride]
        self.padding_shape = [*[[self.model_cfg['wi'][1], 8]] * 1,
                              *[[self.model_cfg['wi'][2], 16]] * self.model_cfg['d'][2],
                              *[[self.model_cfg['wi'][3], 32]] * (self.model_cfg['d'][3] - 1), ]
        # self.selfAtt_padding_shape = [*[[self.model_cfg['wi'][0], 4]] * self.model_cfg['sa'][0],
        #                               *[[self.model_cfg['wi'][1], 8]] * self.model_cfg['sa'][1],
        #                               *[[self.model_cfg['wi'][2], 16]] * self.model_cfg['sa'][2],
        #                               *[[self.model_cfg['wi'][3], 32]] * self.model_cfg['sa'][3],
        #                               ]

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        if self.cfg.MODEL.DROPOUT_RATE > 0:
            self.dropout = nn.Dropout(self.cfg.MODEL.DROPOUT_RATE)
        self.fc = nn.Linear(self.model_cfg['wi'][3],
                            self.cfg.MODEL.NUM_CLASSES, bias=True)

        self.act = nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                if hasattr(m, "transform_final_bn") and m.transform_final_bn:
                    batchnorm_weight = 0.0
                else:
                    batchnorm_weight = 1.0
                m.weight.data.fill_(batchnorm_weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, inplanes, planes, n_blocks, n_sa, temporal_k=1, temporal_p=0):

        downsample = nn.Sequential(
            conv1x1(inplanes, planes, 2),
            nn.BatchNorm3d(planes),
        )

        layers = []
        layers.append(Bottleneck(inplanes, planes, 2, downsample, self.model_cfg['g'],
                                 self.model_cfg['b'], temporal_k=temporal_k, temporal_p=temporal_p))
        if n_sa == n_blocks:
            layers.append(PrgSelfAtt(dim=planes,
                                     dim_inner=planes // 2,
                                     pool_size=[None, 4, 4]))
        for i in range(1, n_blocks):
            layers.append(Bottleneck(planes, planes, 1, None, self.model_cfg['g'],
                                     self.model_cfg['b'], temporal_k=temporal_k, temporal_p=temporal_p))
            if i > (n_blocks - n_sa - 1):
                layers.append(PrgSelfAtt(dim=planes,
                                         dim_inner=planes // 2,
                                         pool_size=[None, 4, 4]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x[0]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        if x.shape[2] > 1:
            need_mean = True
            x = x.permute(0, 2, 1, 3, 4)
            x = x.view(x.size(0), x.size(1), -1)
        else:
            x = x.view(x.size(0), -1)
            need_mean = False

        # head
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.fc(x)

        if not self.training:
            x = self.act(x)

        if need_mean:
            x = x.mean(dim=1)

        return x
