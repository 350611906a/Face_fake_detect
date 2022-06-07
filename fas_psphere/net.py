# date: 2022-01-04 17:58
# author: liucc
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LowPassFilter(torch.nn.Module):
    def __init__(self, channels, stride=1):
        super(LowPassFilter, self).__init__()
        fit = torch.tensor([1., 4., 6., 4., 1.])
        fit = fit[:, None] * fit[None, :]
        fit = fit / fit.sum()
        self.register_buffer('filter', fit[None, None, ...].repeat((channels, 1, 1, 1)))
        self.pad = torch.nn.ZeroPad2d(padding=2)  # adjustable
        self.stride = stride
        self.channels = channels

    def forward(self, x):
        pad_x = self.pad(x)
        y = torch.nn.functional.conv2d(pad_x, self.filter, stride=self.stride, groups=self.channels)
        return y


def conv_bn(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )


def conv_bn1X1(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def depth_conv2d(inp, oup, kernel=1, stride=1, pad=0):
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel_size=kernel, stride=stride, padding=pad, groups=inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, kernel_size=1)
    )


def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


class FPN3(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN3, self).__init__()
        leaky = 0
        if out_channels <= 64:
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride=1)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride=1)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride=1)

        self.merge1 = conv_bn(out_channels, out_channels)
        self.merge2 = conv_bn(out_channels, out_channels)

    def forward(self, x):
        output1 = self.output1(x[0])
        output2 = self.output2(x[1])
        output3 = self.output3(x[2])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]

        return out


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class InvertedResidualAntiAlias(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidualAntiAlias, self).__init__()
        assert stride in [1, 2], stride

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                LowPassFilter(hidden_dim, stride),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                LowPassFilter(hidden_dim, 1),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                LowPassFilter(hidden_dim, stride),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, input_ch, width_mult=0.25, using_anti_alias=True):
        super(MobileNetV2, self).__init__()
        self.backbone_o_ch = [8, 24, 80]
        self.cfgs = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        block = InvertedResidualAntiAlias if using_anti_alias else InvertedResidual
        self.stage1 = nn.Sequential(
            conv_3x3_bn(input_ch, _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8), 2),
            block(_make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8),
                  _make_divisible(16 * width_mult, 4 if width_mult == 0.1 else 8), 1, 1),

            block(_make_divisible(16 * width_mult, 4 if width_mult == 0.1 else 8),
                  _make_divisible(24 * width_mult, 4 if width_mult == 0.1 else 8), 2, 6),

            block(_make_divisible(24 * width_mult, 4 if width_mult == 0.1 else 8),
                  _make_divisible(24 * width_mult, 4 if width_mult == 0.1 else 8), 1, 6),

            block(_make_divisible(24 * width_mult, 4 if width_mult == 0.1 else 8),
                  _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8), 2, 6),

            block(_make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8),
                  _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8), 1, 6),

            block(_make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8),
                  _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8), 1, 6),
        )

        self.stage2 = nn.Sequential(
            block(_make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8),
                  _make_divisible(64 * width_mult, 4 if width_mult == 0.1 else 8), 2, 6),

            block(_make_divisible(64 * width_mult, 4 if width_mult == 0.1 else 8),
                  _make_divisible(64 * width_mult, 4 if width_mult == 0.1 else 8), 1, 6),

            block(_make_divisible(64 * width_mult, 4 if width_mult == 0.1 else 8),
                  _make_divisible(64 * width_mult, 4 if width_mult == 0.1 else 8), 1, 6),

            block(_make_divisible(64 * width_mult, 4 if width_mult == 0.1 else 8),
                  _make_divisible(64 * width_mult, 4 if width_mult == 0.1 else 8), 1, 6),

            block(_make_divisible(64 * width_mult, 4 if width_mult == 0.1 else 8),
                  _make_divisible(96 * width_mult, 4 if width_mult == 0.1 else 8), 1, 6),

            block(_make_divisible(96 * width_mult, 4 if width_mult == 0.1 else 8),
                  _make_divisible(96 * width_mult, 4 if width_mult == 0.1 else 8), 1, 6),

            block(_make_divisible(96 * width_mult, 4 if width_mult == 0.1 else 8),
                  _make_divisible(96 * width_mult, 4 if width_mult == 0.1 else 8), 1, 6),
        )

        self.stage3 = nn.Sequential(
            block(_make_divisible(96 * width_mult, 4 if width_mult == 0.1 else 8),
                  _make_divisible(160 * width_mult, 4 if width_mult == 0.1 else 8), 2, 6),

            block(_make_divisible(160 * width_mult, 4 if width_mult == 0.1 else 8),
                  _make_divisible(160 * width_mult, 4 if width_mult == 0.1 else 8), 1, 6),

            block(_make_divisible(160 * width_mult, 4 if width_mult == 0.1 else 8),
                  _make_divisible(160 * width_mult, 4 if width_mult == 0.1 else 8), 1, 6),

            block(_make_divisible(160 * width_mult, 4 if width_mult == 0.1 else 8),
                  _make_divisible(320 * width_mult, 4 if width_mult == 0.1 else 8), 1, 6),

            # nn.Sequential(
            #     nn.Conv2d(in_channels=80, out_channels=40, kernel_size=1),
            #     nn.ReLU(inplace=True),
            #     depth_conv2d(40, self.backbone_o_ch[2], kernel=3, stride=1, pad=1),
            #     nn.ReLU(inplace=True)
            # ),
        )

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        return x1, x2, x3


class EquivariantBackbone(torch.nn.Module):
    def __init__(self, input_ch):
        super(EquivariantBackbone, self).__init__()

        self.state_1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_ch, 16, kernel_size=1, stride=1, padding=0, bias=True),
            torch.nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=True),
            LowPassFilter(16, 2),

            torch.nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            LowPassFilter(32, 2),

            torch.nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            LowPassFilter(64, 2),
        )
        self.state_1_pool = torch.nn.AdaptiveMaxPool2d(1)

        self.state_2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            LowPassFilter(128, 2),

            torch.nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            LowPassFilter(128, 2),

            torch.nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            LowPassFilter(128, 2),
        )
        self.state_2_pool = torch.nn.AdaptiveMaxPool2d(1)

        self.state_3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            LowPassFilter(256, 2),

            torch.nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            LowPassFilter(256, 2),
        )
        self.state_3_pool = torch.nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        x1 = self.state_1(x)
        x2 = self.state_2(x1)
        x3 = self.state_3(x2)
        x1p = self.state_1_pool(x1)
        x2p = self.state_2_pool(x2)
        x3p = self.state_3_pool(x3)
        y = torch.cat([x1p, x2p, x3p], dim=1)
        return y


def weights_init(net):
    class_names = net.__class__.__name__
    if class_names.find('Conv') != -1:
        torch.nn.init.normal_(net.weight.data, 0.0, 0.02)
    elif class_names.find('BatchNorm') != -1:
        torch.nn.init.normal_(net.weight.data, 0.1, 0.02)
        torch.nn.init.constant_(net.bias.data, 0.0)
    elif class_names.find('Linear') != -1:
        torch.nn.init.normal_(net.weight.data, 0.0, 0.02)
        

class FasNet(torch.nn.Module):
    def __init__(self, input_ch, img_dim, hidden_dim=128, export=False):
        super(FasNet, self).__init__()

        using_anti_alias = False
        requires_grad = True
        self.center = torch.nn.Parameter(torch.ones(1, hidden_dim).float(), requires_grad=True)
        self.radius = torch.nn.Parameter(torch.ones(1, 1).float() * 0.01, requires_grad=requires_grad)
        self.margin = torch.nn.Parameter(torch.ones(1, 1).float() * 0.001, requires_grad=requires_grad)
        self.backbone = MobileNetV2(input_ch, width_mult=0.5, using_anti_alias=using_anti_alias)
        self.sqz = torch.nn.Sequential(
            torch.nn.Conv2d(16, 4, 1, 1, 0),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(4, 8, 3, 2, 1, bias=False),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(8, 1, 1, 1, 0, bias=False),
            torch.nn.BatchNorm2d(1),
        )
        self.sqz_2 = torch.nn.Sequential(
            torch.nn.Conv2d(160, 320, 3, 2, 1),
            torch.nn.BatchNorm2d(320),
            torch.nn.AdaptiveAvgPool2d(1),
        )
        self.fc = torch.nn.Sequential(
            # torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(320 + 16 * 16, hidden_dim * 2),
            torch.nn.BatchNorm1d(hidden_dim * 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
        )
        self.register_buffer('running_radius', torch.ones_like(self.radius).float() * 0.01)
        self.register_buffer('running_margin', torch.ones_like(self.margin).float() * 0.001)
        self.export = export
        self.img_dim = img_dim
        # self.apply(weights_init)
        self._init_weights()

    def forward(self, x):
        bs = x.size(0)
        x1, x2, x3 = self.backbone(x)
        # x_sum = self.sqz(x1).view(bs, -1) + x3.view(bs, -1)
        x_cat = torch.cat([self.sqz_2(x3).view(bs, -1), self.sqz(x1).view(bs, -1)], dim=1)
        x = self.fc(x_cat)

        if self.training:
            x = torch.min(torch.tensor(15.0).to(x.device), x)
            x = torch.max(torch.tensor(-15.0).to(x.device), x)  # in case we get NAN
        pd = torch.softmax(x, dim=1)  # probability distribution
        center_prob = torch.softmax(self.center, dim=1)
        if self.export:
            return pd
        return pd, center_prob, self.radius, self.margin

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
        return


def fast_test():
    ev_net = FasNet(1, img_dim=256, hidden_dim=128)
    x = torch.randn(2, 1, 256, 256)
    y, c, r, m = ev_net(x)
    print(y.size())
    print(c.size())
    print(r.size())
    print(m.size())


def fast_export():
    net = FasNet(1, img_dim=224, hidden_dim=128, export=True)
    x = torch.randn(1, 1, 256, 256)
    torch.onnx.export(net, x, 'prob_sphere_fas_h256-w256.onnx')


if __name__ == "__main__":
    fast_test()
    fast_export()




















