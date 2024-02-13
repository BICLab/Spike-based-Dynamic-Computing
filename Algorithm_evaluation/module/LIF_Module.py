import torch.nn as nn
from spikingjelly.clock_driven import layer as sj_layer

from module.LIF import *
from module.RM import *
from module.utils import *


class RMFCLIF(nn.Module):
    def __init__(
        self,
        inputSize,
        hiddenSize,
        spikeActFun,
        attention="TR",
        onlyLast=False,
        useBatchNorm=False,
        init_method=None,
        scale=0.3,
        pa_dict=None,
        pa_train_self=False,
        bias=True,
        T=60,
        p=0,
        track_running_stats=False,
        mode_select="spike",
        mem_act=torch.relu,
        TR_model="NTR",
    ):
        super().__init__()
        self.onlyLast = onlyLast

        self.useBatchNorm = useBatchNorm

        self.network = nn.Sequential()
        self.attention_flag = attention
        self.linear = sj_layer.SeqToANNContainer(
            nn.Linear(
                in_features=inputSize,
                out_features=hiddenSize,
                bias=bias,
            )
        )

        if self.useBatchNorm:
            self.BNLayer = sj_layer.SeqToANNContainer(
                nn.BatchNorm1d(
                    num_features=hiddenSize, track_running_stats=track_running_stats
                )
            )

        if init_method is not None:
            paramInit(model=self.linear, method=init_method)
        self.attention = nn.Identity()
        if self.attention_flag == "TCR":
            self.attention = TCR(T, hiddenSize)
        elif self.attention_flag == "TR":
            self.attention = TR(T, hiddenSize)
        elif self.attention_flag == "CR":
            self.attention = CR(T, hiddenSize)
        elif self.attention_flag == "no":
            pass

        self.network.add_module(
            "IF",
            sj_layer.MultiStepContainer(
                IFCell(
                    inputSize,
                    hiddenSize,
                    spikeActFun,
                    bias=bias,
                    scale=scale,
                    pa_dict=pa_dict,
                    pa_train_self=pa_train_self,
                    p=p,
                    mode_select=mode_select,
                    mem_act=mem_act,
                    TR_model=TR_model,
                )
            ),
        )

    def forward(self, data):

        for module in self.network.children():
            if hasattr(module, "reset"):
                module.reset()
            else:
                if isinstance(module, sj_layer.MultiStepContainer):
                    for l in module.children():
                        if hasattr(l, "reset"):
                            l.reset()
                else:
                    print(f"{module} has no reset function.")

        # b, t, _ = data.size()
        output = self.linear(data)

        if self.useBatchNorm:
            output = self.BNLayer(output)

        data = self.attention(output.permute(1, 0, 2)).permute(1, 0, 2)

        for layer in self.network:
            out = layer(data)
        outputsum = out

        if self.onlyLast:
            return output
        else:
            return outputsum


class RMConvLIF(nn.Module):
    def __init__(
        self,
        inputSize,
        hiddenSize,
        kernel_size,
        spikeActFun,
        attention="no",
        bias=True,
        onlyLast=False,
        padding=1,
        useBatchNorm=False,
        init_method=None,
        scale=0.02,
        pa_dict=None,
        pa_train_self=False,
        T=60,
        stride=1,
        pooling_kernel_size=1,
        p=0,
        track_running_stats=False,
        mode_select="spike",
        mem_act=torch.relu,
        TR_model="NTR",
        RM=False,
        c_sparsity_ratio=1.0,
        t_sparsity_ratio=1.0,
        reserve_coefficient=True,
    ):
        super().__init__()
        self.reserve_coefficient = reserve_coefficient
        self.onlyLast = onlyLast
        self.attention_flag = attention
        self.RM = RM
        self.c_sparsity_ratio = c_sparsity_ratio
        self.t_sparsity_ratio = t_sparsity_ratio
        self.conv2d = sj_layer.SeqToANNContainer(
            nn.Conv2d(
                in_channels=inputSize,
                out_channels=hiddenSize,
                kernel_size=kernel_size,
                bias=True,
                padding=padding,
                stride=stride,
            )
        )

        if init_method is not None:
            paramInit(model=self.conv2d, method=init_method)

        self.useBatchNorm = useBatchNorm

        if self.useBatchNorm:
            self.BNLayer = sj_layer.SeqToANNContainer(
                nn.BatchNorm2d(hiddenSize, track_running_stats=track_running_stats)
            )

        self.pooling_kernel_size = pooling_kernel_size
        if self.pooling_kernel_size > 1:
            self.pooling = sj_layer.SeqToANNContainer(
                nn.AvgPool2d(kernel_size=pooling_kernel_size)
            )

        self.attention = nn.Identity()
        if self.attention_flag == "TCR":
            self.attention = TCR(T, hiddenSize, RM=RM)
        elif self.attention_flag == "TR":
            self.attention = TR(T, hiddenSize, RM=RM)
        elif self.attention_flag == "CR":
            self.attention = CR(T, hiddenSize, RM=RM)
        elif self.attention_flag == "no":
            pass

        self.network = nn.Sequential()
        self.network.add_module(
            "ConvIF",
            sj_layer.MultiStepContainer(
                ConvIFCell(
                    inputSize=inputSize,
                    hiddenSize=hiddenSize,
                    kernel_size=kernel_size,
                    bias=bias,
                    spikeActFun=spikeActFun,
                    padding=padding,
                    scale=scale,
                    pa_dict=pa_dict,
                    pa_train_self=pa_train_self,
                    p=p,
                    mode_select=mode_select,
                    mem_act=mem_act,
                    TR_model=TR_model,
                )
            ),
        )

        if RM:
            self.avg_c = nn.AdaptiveAvgPool2d(1)
            self.avg_t = nn.AdaptiveAvgPool3d(1)

    def forward(self, data):

        for module in self.network.children():
            if hasattr(module, "reset"):
                module.reset()
            else:
                if isinstance(module, sj_layer.MultiStepContainer):
                    for l in module.children():
                        if hasattr(l, "reset"):
                            l.reset()
                else:
                    print(f"{module} has no reset function.")

        # t, b, c, h, w = data.size()

        output = self.conv2d(data)

        if self.useBatchNorm:
            output = self.BNLayer(output)

        if self.pooling_kernel_size > 1:
            output = self.pooling(output)

        # _, _, c, h, w = output.size()

        if self.attention_flag == "no":
            data = output
        else:
            if self.RM:
                if self.attention_flag == "TR":
                    data = output.permute(1, 0, 2, 3, 4)
                    ta = self.attention(data)
                    pred_saliency_t = self.avg_t(ta).squeeze()
                    pred_saliency_wta, winner_mask_t = winner_take_all(
                        pred_saliency_t, sparsity_ratio=self.t_sparsity_ratio
                    )
                    pred_saliency_wta = (
                        pred_saliency_wta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    )
                    winner_mask = (
                        winner_mask_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    )
                    if not self.training:
                        data = data * winner_mask
                    if self.reserve_coefficient:
                        data = data * pred_saliency_wta

                elif self.attention_flag == "CR":
                    data = output.permute(1, 0, 2, 3, 4)
                    ca = self.attention(data)

                    pred_saliency_c = self.avg_c(ca).squeeze()
                    pred_saliency_wta, winner_mask_c = winner_take_all(
                        pred_saliency_c, sparsity_ratio=self.c_sparsity_ratio
                    )
                    pred_saliency_wta = (
                        pred_saliency_wta.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                    )
                    winner_mask = winner_mask_c.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                    if not self.training:
                        data = data * winner_mask
                    if self.reserve_coefficient:
                        data = data * pred_saliency_wta

                elif self.attention_flag == "TCR":
                    data = output.permute(1, 0, 2, 3, 4)
                    ta, ca = self.attention(data)

                    pred_saliency_c = self.avg_c(ca).squeeze()
                    pred_saliency_t = self.avg_t(ta).squeeze()
                    pred_saliency_c_wta, winner_mask_c = winner_take_all(
                        pred_saliency_c, sparsity_ratio=self.c_sparsity_ratio
                    )
                    pred_saliency_t_wta, winner_mask_t = winner_take_all(
                        pred_saliency_t, sparsity_ratio=self.t_sparsity_ratio
                    )
                    pred_saliency_wta = pred_saliency_c_wta.unsqueeze(1).unsqueeze(
                        -1
                    ).unsqueeze(-1) * pred_saliency_t_wta.unsqueeze(-1).unsqueeze(
                        -1
                    ).unsqueeze(
                        -1
                    )
                    winner_mask = winner_mask_c.unsqueeze(1).unsqueeze(-1).unsqueeze(
                        -1
                    ) * winner_mask_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    if not self.training:
                        data = data * winner_mask
                    if self.reserve_coefficient:
                        data = data * pred_saliency_wta

            else:
                data = self.attention(output.permute(1, 0, 2, 3, 4))
                pred_saliency_wta = torch.tensor(0).cuda()

        data = data.permute(1, 0, 2, 3, 4)
        # _, _, c, h, w = data.size()

        for layer in self.network:
            out = layer(data)
        outputsum = out

        if self.onlyLast:
            return output, pred_saliency_wta
        else:
            return outputsum, pred_saliency_wta
