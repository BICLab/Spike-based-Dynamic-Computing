import torch
import torch.nn as nn
from spikingjelly.clock_driven import layer
from spikingjelly.clock_driven import neuron as cext_neuron
from spikingjelly.clock_driven import surrogate

from module.RM import *


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        connect_f="ADD",
        T=8,
        attn="no",
        attn_flags="0",
    ):
        super(BasicBlock, self).__init__()
        self.connect_f = connect_f
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                "SpikingBasicBlock only supports groups=1 and base_width=64"
            )
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in SpikingBasicBlock")
        self.T = T

        self.attn1 = nn.Identity()
        self.attn2 = nn.Identity()
        if attn_flags == "1":
            if attn == "HAM":
                self.attn1 = HAM(T, planes)
            elif attn == "RM_SEW":
                self.attn1 = RM_SEW(T, planes)
            elif attn == "TCA_SEW":
                self.attn1 = TCA_SEW(T, planes)
            elif attn == "RM_TCA_SEW":
                if planes < 256:
                    self.attn1 = RM_SEW(T, planes)
                else:
                    self.attn1 = TCA_SEW(T, planes)
            elif attn == "RM_SEW_only_CA":
                self.attn1 = RM_SEW_only_CA(T, planes)
            elif attn == "HTSA_sa":
                self.attn1 = HTSA_sa(T, planes)
        elif attn_flags == "2":
            if attn == "HAM":
                self.attn1 = HAM(T, planes)
                self.attn2 = HAM(T, planes)
            elif attn == "RM_SEW":
                self.attn1 = RM_SEW(T, planes)
                self.attn2 = RM_SEW(T, planes)
            elif attn == "TCA_SEW":
                self.attn1 = TCA_SEW(T, planes)
                self.attn2 = TCA_SEW(T, planes)
            elif attn == "RM_TCA_SEW":
                if planes < 256:
                    self.attn1 = RM_SEW(T, planes)
                    self.attn2 = RM_SEW(T, planes)
                else:
                    self.attn1 = TCA_SEW(T, planes)
                    self.attn2 = TCA_SEW(T, planes)
            elif attn == "RM_SEW_only_CA":
                self.attn1 = RM_SEW_only_CA(T, planes)
                self.attn2 = RM_SEW_only_CA(T, planes)
            elif attn == "HTSA_sa":
                self.attn1 = HTSA_sa(T, planes)
                self.attn2 = HTSA_sa(T, planes)

        self.conv1 = layer.SeqToANNContainer(
            conv3x3(inplanes, planes, stride),
            norm_layer(planes, track_running_stats=False),
        )
        self.sn1 = cext_neuron.MultiStepIFNode(
            # surrogate_function=surrogate.ATan(),
        )

        self.conv2 = layer.SeqToANNContainer(
            conv3x3(planes, planes), norm_layer(planes, track_running_stats=False)
        )
        self.downsample = downsample
        self.stride = stride
        self.sn2 = cext_neuron.MultiStepIFNode(
            # surrogate_function=surrogate.ATan(),
        )

    def _forward_attn(self, out, attn_block):
        out = out.transpose(0, 1).contiguous()
        out = attn_block(out)
        out = out.transpose(0, 1).contiguous()
        return out

    def forward(self, x):
        identity = x[0]
        firing_num = x[1]

        out = self.conv1(identity)
        out = self._forward_attn(out, self.attn1)
        out = self.sn1(out)
        # firing_num.append(out.sum())

        out = self.conv2(out)
        out = self._forward_attn(out, self.attn2)
        out = self.sn2(out)
        # firing_num.append(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        if self.connect_f == "ADD":
            out += identity
        elif self.connect_f == "AND":
            out *= identity
        elif self.connect_f == "IAND":
            out = identity * (1.0 - out)
        else:
            raise NotImplementedError(self.connect_f)

        firing_num.append(out.sum())

        return out, firing_num


def zero_init_blocks(net: nn.Module, connect_f: str):
    for m in net.modules():
        if isinstance(m, BasicBlock):
            for m_conv in m.conv2.modules():
                if isinstance(m_conv, nn.GroupNorm):
                    nn.init.constant_(m_conv.weight, 0)
                    if connect_f == "AND":
                        nn.init.constant_(m_conv.bias, 1)


class SEWResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        T=8,
        connect_f="ADD",
        attn="no",
        attn_where="0000",
    ):
        super(SEWResNet, self).__init__()
        self.T = T
        self.connect_f = connect_f
        self.attn_where = attn_where

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = layer.SeqToANNContainer(
            nn.Conv2d(2, self.inplanes, kernel_size=7, stride=2, padding=3, bias=True)
        )
        self.bn1 = layer.SeqToANNContainer(
            norm_layer(self.inplanes, track_running_stats=False)
        )

        self.sn1 = layer.MultiStepContainer(
            cext_neuron.ParametricLIFNode(
                surrogate_function=surrogate.ATan(),
            )
        )
        self.maxpool = layer.SeqToANNContainer(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(
            block,
            64,
            layers[0],
            connect_f=connect_f,
            attn=attn,
            attn_flags=attn_where[0],
        )
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            connect_f=connect_f,
            attn=attn,
            attn_flags=attn_where[1],
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            connect_f=connect_f,
            attn=attn,
            attn_flags=attn_where[2],
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            connect_f=connect_f,
            attn=attn,
            attn_flags=attn_where[3],
        )
        self.avgpool = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1,
        dilate=False,
        connect_f=None,
        attn=None,
        attn_flags=None,
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                layer.SeqToANNContainer(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion, track_running_stats=False),
                ),
                cext_neuron.MultiStepIFNode(
                    # tau=2.0,
                    # surrogate_function=surrogate.ATan(),
                ),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                connect_f,
                T=self.T,
                attn=attn,
                attn_flags=attn_flags,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    connect_f=connect_f,
                    T=self.T,
                    attn=attn,
                    attn_flags=attn_flags,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        firing_num = []

        x = self.sn1(x)
        firing_num.append(x.sum())
        x = self.maxpool(x)

        x = self.layer1((x, firing_num))
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        firing_num = x[1]

        x = x[0]
        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        return self.fc(x.mean(dim=0)), firing_num

    def forward(self, x):
        return self._forward_impl(x)


def _sew_resnet(block, layers, **kwargs):
    model = SEWResNet(block, layers, **kwargs)
    return model


def sew_resnet18(**kwargs):
    return _sew_resnet(BasicBlock, [2, 2, 2, 2], **kwargs)


def sew_resnet34(**kwargs):
    return _sew_resnet(BasicBlock, [3, 4, 6, 3], **kwargs)
