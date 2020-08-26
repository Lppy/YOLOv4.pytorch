import time
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

from yolov4.eval import Yolo_inference
from yolov4.loss import Yolo_loss

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x

class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, x, target_size, inference=False):
        _, _, tH, tW = target_size
        if inference:
            B, C, H, W = x.shape
            return x.view(B, C, H, 1, W, 1).expand(B, C, H, tH // H, W, tW // W).contiguous().view(B, C, tH, tW)
        else:
            return F.interpolate(x, size=(tH, tW), mode='nearest')

class UpBlock(nn.Module):
    def __init__(self, ch):
        super(UpBlock, self).__init__()
        self.conv1 = Conv_Bn_Activation(ch*2, ch, 1, 1, 'leaky')
        self.upsample = Upsample()
        self.conv2 = Conv_Bn_Activation(ch*2, ch, 1, 1, 'leaky')

    def forward(self, x, downsample, inference):
        x = self.conv1(x)
        up = self.upsample(x, downsample.size(), inference)
        x = self.conv2(downsample)
        x = torch.cat([x, up], dim=1)
        return x

class Conv_Bn_Activation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, bn=True, bias=False):
        super().__init__()
        pad = (kernel_size - 1) // 2

        self.conv = []
        if bias:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad))
        else:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False))
        
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))

        if activation == "mish":
            self.conv.append(Mish())
        elif activation == "relu":
            self.conv.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "linear":
            pass
        else:
            assert False, "Wrong type of activation!"

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        return self.conv(x)

class Block(nn.Module):
    def __init__(self, ch, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.res = nn.Sequential(
            Conv_Bn_Activation(ch, ch, 1, 1, 'mish'),
            Conv_Bn_Activation(ch, ch, 3, 1, 'mish')
        )

    def forward(self, x):
        h = self.res(x)
        return x + h if self.shortcut else h

class ResBlock(nn.Module):
    """
    Sequential residual blocks each of which consists of \
    two convolution layers.
    Args:
        ch (int): number of input and output channels.
        nblocks (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """
    def __init__(self, ch, nblocks=1, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.res_blocks = []
        for i in range(nblocks):
            self.res_blocks.append(Block(ch, shortcut))
        self.res_blocks = nn.Sequential(*self.res_blocks)

    def forward(self, x):
        return self.res_blocks(x)


class DownSample1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(3, 32, 3, 1, 'mish')
        self.conv2 = Conv_Bn_Activation(32, 64, 3, 2, 'mish')
        self.conv3 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        self.conv4 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(64, 32, 1, 1, 'mish')
        self.conv6 = Conv_Bn_Activation(32, 64, 3, 1, 'mish')
        self.conv7 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        self.conv8 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')

    def forward(self, input):
        # init conv layer
        x1 = self.conv1(input)
        # flow: 2 - [4 - res - 7 | 3] - 8
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x2)

        # 2 res block
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x6 = x6 + x4
        # res block end
        x7 = self.conv7(x6)

        x7 = torch.cat([x7, x3], dim=1)
        x8 = self.conv8(x7)
        return x8


class DownBlock(nn.Module):
    def __init__(self, ch, n_resblocks):
        super(DownBlock, self).__init__()
        self.conv1 = Conv_Bn_Activation(ch, ch*2, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(ch*2, ch, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(ch*2, ch, 1, 1, 'mish')
        self.resblock = ResBlock(ch=ch, nblocks=n_resblocks)
        self.conv4 = Conv_Bn_Activation(ch, ch, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(ch*2, ch*2, 1, 1, 'mish')

    def forward(self, input):
        # input: ch, output: ch*2
        # flow: 1 - [3 - res - 4 | 2] - 5
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class Neck(nn.Module):
    def __init__(self, inference=False):
        super(Neck, self).__init__()
        self.inference = inference

        self.CBAx3_1 = nn.Sequential(OrderedDict([
            ('conv1', Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')),
            ('conv2', Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')),
            ('conv3', Conv_Bn_Activation(1024, 512, 1, 1, 'leaky'))
        ]))
        # SPP
        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)
        # SPP
        self.CBAx3_2 = nn.Sequential(OrderedDict([
            ('conv4', Conv_Bn_Activation(2048, 512, 1, 1, 'leaky')),
            ('conv5', Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')),
            ('conv6', Conv_Bn_Activation(1024, 512, 1, 1, 'leaky'))
        ]))

        self.upConv_1 = UpBlock(256)
        self.CBAx5_1 = nn.Sequential(OrderedDict([
            ('conv9', Conv_Bn_Activation(512, 256, 1, 1, 'leaky')),
            ('conv10', Conv_Bn_Activation(256, 512, 3, 1, 'leaky')),
            ('conv11', Conv_Bn_Activation(512, 256, 1, 1, 'leaky')),
            ('conv12', Conv_Bn_Activation(256, 512, 3, 1, 'leaky')),
            ('conv13', Conv_Bn_Activation(512, 256, 1, 1, 'leaky'))
        ]))

        self.upConv_2 = UpBlock(128)
        self.CBAx5_2 = nn.Sequential(OrderedDict([
            ('conv16', Conv_Bn_Activation(256, 128, 1, 1, 'leaky')),
            ('conv17', Conv_Bn_Activation(128, 256, 3, 1, 'leaky')),
            ('conv18', Conv_Bn_Activation(256, 128, 1, 1, 'leaky')),
            ('conv19', Conv_Bn_Activation(128, 256, 3, 1, 'leaky')),
            ('conv20', Conv_Bn_Activation(256, 128, 1, 1, 'leaky'))
        ]))

    def forward(self, downsample5, downsample4, downsample3, inference=False):
        x3 = self.CBAx3_1(downsample5)
        # SPP
        m1 = self.maxpool1(x3)
        m2 = self.maxpool2(x3)
        m3 = self.maxpool3(x3)
        spp = torch.cat([m3, m2, m1, x3], dim=1)
        # SPP end
        x6 = self.CBAx3_2(spp)

        x8 = self.upConv_1(x6, downsample4, self.inference)
        x13 = self.CBAx5_1(x8)

        x15 = self.upConv_2(x13, downsample3, self.inference)
        x20 = self.CBAx5_2(x15)

        return x20, x13, x6

class Yolov4Head(nn.Module):
    def __init__(self, output_ch, anchors, num_classes=80, inference=False):
        super(Yolov4Head, self).__init__()
        self.inference = inference

        self.out1 = nn.Sequential(OrderedDict([
            ('conv1', Conv_Bn_Activation(128, 256, 3, 1, 'leaky')),
            ('conv2', Conv_Bn_Activation(256, output_ch, 1, 1, 'linear', bn=False, bias=True))
        ]))
        # Yolo_inference YoloLayer
        self.yolo1 = Yolo_inference(anchor_mask=[0, 1, 2], num_classes=num_classes,
            anchors=anchors, num_anchors=9, stride=8)

        self.conv3 = Conv_Bn_Activation(128, 256, 3, 2, 'leaky')
        self.CBAx5_1 = nn.Sequential(OrderedDict([
            ('conv4', Conv_Bn_Activation(512, 256, 1, 1, 'leaky')),
            ('conv5', Conv_Bn_Activation(256, 512, 3, 1, 'leaky')),
            ('conv6', Conv_Bn_Activation(512, 256, 1, 1, 'leaky')),
            ('conv7', Conv_Bn_Activation(256, 512, 3, 1, 'leaky')),
            ('conv8', Conv_Bn_Activation(512, 256, 1, 1, 'leaky'))
        ]))
        self.out2 = nn.Sequential(OrderedDict([
            ('conv9', Conv_Bn_Activation(256, 512, 3, 1, 'leaky')),
            ('conv10', Conv_Bn_Activation(512, output_ch, 1, 1, 'linear', bn=False, bias=True))
        ]))
        
        self.yolo2 = Yolo_inference(anchor_mask=[3, 4, 5], num_classes=num_classes,
            anchors=anchors, num_anchors=9, stride=16)

        self.conv11 = Conv_Bn_Activation(256, 512, 3, 2, 'leaky')
        self.CBAx5_2 = nn.Sequential(OrderedDict([
            ('conv12', Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')),
            ('conv13', Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')),
            ('conv14', Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')),
            ('conv15', Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')),
            ('conv16', Conv_Bn_Activation(1024, 512, 1, 1, 'leaky'))
        ]))
        self.out3 = nn.Sequential(OrderedDict([
            ('conv17', Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')),
            ('conv18', Conv_Bn_Activation(1024, output_ch, 1, 1, 'linear', bn=False, bias=True))
        ]))
        
        self.yolo3 = Yolo_inference(anchor_mask=[6, 7, 8], num_classes=num_classes,
            anchors=anchors, num_anchors=9, stride=32)

    def forward(self, input1, input2, input3):
        x2 = self.out1(input1)

        x3 = self.conv3(input1)
        x3 = torch.cat([x3, input2], dim=1)
        x8 = self.CBAx5_1(x3)
        x10 = self.out2(x8)

        x11 = self.conv11(x8)
        x11 = torch.cat([x11, input3], dim=1)
        x16 = self.CBAx5_2(x11)
        x18 = self.out3(x16)
        
        if self.inference:
            y1 = self.yolo1(x2)
            y2 = self.yolo2(x10)
            y3 = self.yolo3(x18)

            return [y1, y2, y3]
        else:
            return [x2, x10, x18]



class Yolov4(nn.Module):
    def __init__(self, anchors, backbone_weight=None, n_classes=80, image_size=608, inference=False):
        super(Yolov4, self).__init__()
        self.inference = inference

        output_ch = (4 + 1 + n_classes) * 3

        # backbone
        self.down1 = DownSample1()
        self.down2 = DownBlock(64, n_resblocks=2)
        self.down3 = DownBlock(128, n_resblocks=8)
        self.down4 = DownBlock(256, n_resblocks=8)
        self.down5 = DownBlock(512, n_resblocks=4)
        # neck
        self.neck = Neck(inference)
        
        if backbone_weight:
            pretrained_dict = torch.load(backbone_weight)
            model_dict = self.state_dict()
            self.load_state_dict(pretrained_dict)
            print("Loaded CSPDarknet53 backbone.")

            eval_blocks = [ self.down1, self.down2, self.down3, self.down4, self.down5 ]
            for block in eval_blocks:
                block.eval()
                for p in block.parameters(): p.requires_grad=False
            print("Frozen CSPDarknet53 backbone.")
        
        # head
        self.head = Yolov4Head(output_ch, anchors, num_classes=n_classes, inference=inference)

        # self.neck.apply(weights_init)
        self.head.apply(weights_init)

        if not self.inference:
            self.criterion = Yolo_loss(anchors, image_size, n_classes=n_classes)

    def train(self, mode=True):
        self.neck.train()
        self.head.train()

    def eval(self, mode=True):
        eval_blocks = [ self.down1, self.down2, self.down3, self.down4, self.down5, self.neck, self.head ]
        for block in eval_blocks:
            block.eval()

    def forward(self, input, target=None, label=None):
        d1 = self.down1(input)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        x20, x13, x6 = self.neck(d5, d4, d3)

        output = self.head(x20, x13, x6)

        if not self.inference:
            output = self.criterion(output, target, label)

        return output