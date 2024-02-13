import torch
from einops import rearrange
from torch import nn


class TimeAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(TimeAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class TimeAttention_(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(TimeAttention_, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.sharedMLP = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_planes // ratio, in_planes, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=5):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = rearrange(x, "b f c h w -> b c f h w")
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        out = self.sigmoid(avgout + maxout)
        out = rearrange(out, "b c f h w -> b f c h w")
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c = x.shape[2]
        x = rearrange(x, "b f c h w -> b (f c) h w")
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        x = x.unsqueeze(1)

        return self.sigmoid(x)


class TCR(nn.Module):
    def __init__(self, timeWindows, channels, stride=1, RM=False):
        super(TCR, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.RM = RM

        self.ca = ChannelAttention(channels)
        self.ta = TimeAttention(timeWindows)

        self.stride = stride

    def forward(self, x):
        out_ = self.ta(x) * x
        out = self.ca(out_) * out_

        out = self.relu(out)
        if self.RM:
            return self.ta(x), self.ca(out_)
        else:
            return out


class TR(nn.Module):
    def __init__(self, timeWindows, stride=1, RM=False):
        super(TR, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.RM = RM
        self.ta = TimeAttention(timeWindows)

        self.stride = stride

    def forward(self, x):
        out = self.ta(x) * x

        out = self.relu(out)
        if self.RM:
            return self.ta(x)
        else:
            return out


class CR(nn.Module):
    def __init__(self, channels, stride=1, RM=False):
        super(CR, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.RM = RM

        self.ca = ChannelAttention(channels)

        self.stride = stride

    def forward(self, x):
        out = self.ca(x) * x

        out = self.relu(out)
        if self.RM:
            return self.ca(x)
        else:
            return out


class TR_(nn.Module):
    def __init__(self, timeWindows, stride=1):
        super(TR_, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.ta = TimeAttention_(timeWindows)

        self.stride = stride

    def forward(self, x):
        out = self.ta(x) * x

        out = self.relu(out)
        return out


class RM_SEW(nn.Module):
    def __init__(
        self,
        timeWindows,
        channels,
        reduction=1,
        dimension=5,
        c_sparsity_ratio=0.8,
        t_sparsity_ratio=0.9,
    ):
        super(RM_SEW, self).__init__()

        self.reserve_coefficient = True

        self.ca = ChannelAttention(channels, reduction)
        self.ta = TimeAttention(timeWindows, reduction)

        self.avg_c = nn.AdaptiveAvgPool2d(1)
        self.avg_t = nn.AdaptiveAvgPool3d(1)

        self.c_sparsity_ratio = c_sparsity_ratio
        self.t_sparsity_ratio = t_sparsity_ratio

    def forward(self, x):
        ta = self.ta(x)
        out = ta * x
        ca = self.ca(out)

        pred_saliency_c = self.avg_c(ca).squeeze()
        pred_saliency_t = self.avg_t(ta).squeeze()
        pred_saliency_c_wta, winner_mask_c = winner_take_all(
            pred_saliency_c, sparsity_ratio=self.c_sparsity_ratio
        )
        pred_saliency_t_wta, winner_mask_t = winner_take_all(
            pred_saliency_t, sparsity_ratio=self.t_sparsity_ratio
        )

        pred_saliency_wta = pred_saliency_c_wta.unsqueeze(1).unsqueeze(-1).unsqueeze(
            -1
        ) * pred_saliency_t_wta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        winner_mask = winner_mask_c.unsqueeze(1).unsqueeze(-1).unsqueeze(
            -1
        ) * winner_mask_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        out = x * winner_mask
        if self.reserve_coefficient:
            out = out * pred_saliency_wta
        return out


class RM_SEW_only_CA(nn.Module):
    def __init__(
        self,
        timeWindows,
        channels,
        reduction=1,
        dimension=5,
        c_sparsity_ratio=0.8,
        t_sparsity_ratio=0.9,
    ):
        super(RM_SEW_only_CA, self).__init__()

        self.reserve_coefficient = True

        self.ca = ChannelAttention(channels, reduction)
        # self.ta = TimeAttention(timeWindows, reduction)

        self.avg_c = nn.AdaptiveAvgPool2d(1)
        # self.avg_t = nn.AdaptiveAvgPool3d(1)

        self.c_sparsity_ratio = c_sparsity_ratio
        # self.t_sparsity_ratio = t_sparsity_ratio

    def forward(self, x):
        # ta = self.ta(x)
        # out = ta * x
        ca = self.ca(x)

        pred_saliency_c = self.avg_c(ca).squeeze()
        # pred_saliency_t = self.avg_t(ta).squeeze()
        pred_saliency_c_wta, winner_mask_c = winner_take_all(
            pred_saliency_c, sparsity_ratio=self.c_sparsity_ratio
        )
        # pred_saliency_t_wta, winner_mask_t = winner_take_all(
        #     pred_saliency_t, sparsity_ratio=self.t_sparsity_ratio
        # )

        pred_saliency_wta = pred_saliency_c_wta.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        winner_mask = winner_mask_c.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)

        out = x * winner_mask
        if self.reserve_coefficient:
            out = out * pred_saliency_wta
        return out


class TCA_SEW(nn.Module):
    def __init__(
        self,
        timeWindows,
        channels,
        reduction=1,
        dimension=5,
        c_sparsity_ratio=1,
        t_sparsity_ratio=1,
    ):
        super(TCA_SEW, self).__init__()

        self.reserve_coefficient = True

        self.ca = ChannelAttention(channels, reduction)
        self.ta = TimeAttention(timeWindows, reduction)

        self.avg_c = nn.AdaptiveAvgPool2d(1)
        self.avg_t = nn.AdaptiveAvgPool3d(1)

        self.c_sparsity_ratio = c_sparsity_ratio
        self.t_sparsity_ratio = t_sparsity_ratio

    def forward(self, x):
        ta = self.ta(x)
        out = ta * x
        ca = self.ca(out)

        pred_saliency_c = self.avg_c(ca).squeeze()
        pred_saliency_t = self.avg_t(ta).squeeze()
        pred_saliency_c_wta, winner_mask_c = winner_take_all(
            pred_saliency_c, sparsity_ratio=self.c_sparsity_ratio
        )
        pred_saliency_t_wta, winner_mask_t = winner_take_all(
            pred_saliency_t, sparsity_ratio=self.t_sparsity_ratio
        )

        pred_saliency_wta = pred_saliency_c_wta.unsqueeze(1).unsqueeze(-1).unsqueeze(
            -1
        ) * pred_saliency_t_wta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        winner_mask = winner_mask_c.unsqueeze(1).unsqueeze(-1).unsqueeze(
            -1
        ) * winner_mask_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        out = x * winner_mask
        if self.reserve_coefficient:
            out = out * pred_saliency_wta
        return out


class ChannelAttention_HAM(nn.Module):
    def __init__(self, channels, T, gamma=2, b=1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.max_pool = nn.AdaptiveMaxPool3d((None, 1, 1))
        self.alpha = nn.Parameter(data=torch.FloatTensor([0.5]), requires_grad=True)
        self.beta = nn.Parameter(data=torch.FloatTensor([0.5]), requires_grad=True)
        self.k = self.get_kernel_num(channels, gamma, b)
        self.conv1d = nn.Conv1d(
            kernel_size=self.k,
            in_channels=channels,
            out_channels=channels,
            padding=self.k // 2,
        )

    def get_kernel_num(self, C, gamma=2, b=1):
        t = math.log2(C) / gamma + b / gamma
        f = math.floor(t)
        k = f + (1 - f % 2)
        return k

    def forward(self, x):
        F_avg = self.avg_pool(x)
        F_max = self.max_pool(x)
        F_add = 0.5 * (F_avg + F_max) + self.alpha * F_avg + self.beta * F_max
        F_add_ = F_add.squeeze(-1).squeeze(-1).permute(0, 2, 1)
        F_add_ = self.conv1d(F_add_).permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        out = torch.sigmoid(F_add_)
        return out


class SpatialAttention_HAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel = channels
        self.Lambda = 0.5
        self.channel_im = self.get_important_channel_num(channels)
        self.channel_subim = channels - self.channel_im
        self.conv = nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False)
        self.norm_active = nn.Sequential(nn.BatchNorm3d(1), nn.ReLU(), nn.Sigmoid())

    def get_important_channel_num(self, channels):
        floor = math.floor(self.Lambda * channels)
        channels_im = floor + floor % 2
        return channels_im

    def get_im_subim_channels(self, channels_im, M):
        _, topk = torch.topk(M, dim=2, k=channels_im)
        topk_ = topk.squeeze(-1).squeeze(-1)
        important_channels = torch.zeros_like(M.squeeze(-1).squeeze(-1)).to(M.device)
        subimportant_channels = torch.ones_like(M.squeeze(-1).squeeze(-1)).to(M.device)
        for i in range(M.shape[1]):
            important_channels[:, i, topk_[:, i]] = 1
            subimportant_channels[:, i, topk_[:, i]] = 0
        important_channels = important_channels.unsqueeze(-1).unsqueeze(-1)
        subimportant_channels = subimportant_channels.unsqueeze(-1).unsqueeze(-1)
        return important_channels, subimportant_channels

    def get_features(self, im_channels, subim_channels, channel_refined_feature):
        import_features = im_channels * channel_refined_feature
        subimportant_features = subim_channels * channel_refined_feature
        return import_features, subimportant_features

    def forward(self, x, M):
        important_channels, subimportant_channels = self.get_im_subim_channels(
            self.channel_im, M
        )
        important_features, subimportant_features = self.get_features(
            important_channels, subimportant_channels, x
        )

        im_AvgPool = torch.mean(important_features, dim=2, keepdim=True) * (
            self.channel / self.channel_im
        )
        im_MaxPool, _ = torch.max(important_features, dim=2, keepdim=True)

        subim_AvgPool = torch.mean(subimportant_features, dim=2, keepdim=True) * (
            self.channel / self.channel_subim
        )
        subim_MaxPool, _ = torch.max(subimportant_features, dim=2, keepdim=True)

        im_x = torch.cat([im_AvgPool, im_MaxPool], dim=2)
        subim_x = torch.cat([subim_AvgPool, subim_MaxPool], dim=2)

        im_x = im_x.transpose(1, 2).contiguous()
        subim_x = subim_x.transpose(1, 2).contiguous()

        A_S1 = self.norm_active(self.conv(im_x))
        A_S2 = self.norm_active(self.conv(subim_x))

        A_S1 = A_S1.transpose(1, 2).contiguous()
        A_S2 = A_S2.transpose(1, 2).contiguous()

        F1 = important_features * A_S1
        F2 = subimportant_features * A_S2

        refined_feature = F1 + F2

        return refined_feature


class HAM(nn.Module):
    def __init__(self, timeWindows, channels, reduction=1, dimension=5):
        super().__init__()
        self.ca = ChannelAttention_HAM(channels, timeWindows)
        self.sa = SpatialAttention_HAM(channels)

    def forward(self, x):
        ca_map = self.ca(x)
        ca_refined = ca_map * x
        out = self.sa(ca_refined, ca_map)
        return out


class HTSA_sa(nn.Module):
    def __init__(self, timeWindows, channels, reduction=1, dimension=5):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.max_pool = nn.AdaptiveMaxPool3d((None, 1, 1))
        self.lam_ = 0.5

        # ---------------------
        # SplitSpatialAttention
        # ---------------------
        self.conv1 = layer.SeqToANNContainer(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
        )
        self.conv2 = layer.SeqToANNContainer(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
        )
        self.active = nn.Sigmoid()
        self.conv_last = nn.Conv3d(
            timeWindows * 2, timeWindows, kernel_size=3, padding=1, bias=False
        )

    def get_im_subim_channels(self, channels_im, M):
        _, topk = torch.topk(M, dim=2, k=channels_im)
        topk_ = topk.squeeze(-1).squeeze(-1)

        important_channels = torch.zeros_like(M.squeeze(-1).squeeze(-1)).to(M.device)
        subimportant_channels = torch.ones_like(M.squeeze(-1).squeeze(-1)).to(M.device)

        for i in range(M.shape[1]):
            important_channels[:, i, topk_[:, i]] = 1
            subimportant_channels[:, i, topk_[:, i]] = 0

        important_channels = important_channels.unsqueeze(-1).unsqueeze(-1)
        subimportant_channels = subimportant_channels.unsqueeze(-1).unsqueeze(-1)
        return important_channels, subimportant_channels

    def forward(self, x):
        n, t, c, _, _ = x.shape
        tca_avg_map = self.avg_pool(x)
        tca_max_map = self.max_pool(x)
        map_add = tca_avg_map + tca_max_map
        map_add = map_add.reshape(n, t, c).unsqueeze(-1).unsqueeze(-1)

        # ---------------------
        # SplitSpatialAttention
        # ---------------------
        # map_add shape: N, T, C, H, W
        important_channels, _ = self.get_im_subim_channels(int(c * self.lam_), map_add)
        important_times, _ = self.get_im_subim_channels(
            int(t * self.lam_), map_add.transpose(1, 2).contiguous()
        )
        important_times = important_times.transpose(1, 2).contiguous()

        important_tc = (important_channels + important_times) / 2
        important_tc = torch.where(important_tc == 0.5, 1, 0)
        subimportant_tc = (
            torch.ones_like(important_tc).to(important_tc.device) - important_tc
        )

        important_features = important_tc * x
        subimportant_features = subimportant_tc * x

        im_AvgPool = torch.mean(important_features, dim=2, keepdim=True) / self.lam_
        im_MaxPool, _ = torch.max(important_features, dim=2, keepdim=True)

        subim_AvgPool = torch.mean(subimportant_features, dim=2, keepdim=True) / (
            1 - self.lam_
        )
        subim_MaxPool, _ = torch.max(subimportant_features, dim=2, keepdim=True)

        im_x = torch.cat([im_AvgPool, im_MaxPool], dim=2)
        subim_x = torch.cat([subim_AvgPool, subim_MaxPool], dim=2)

        im_x = im_x.transpose(1, 0).contiguous()
        subim_x = subim_x.transpose(1, 0).contiguous()

        im_map = self.active(self.conv1(im_x))
        subim_map = self.active(self.conv2(subim_x))

        im_map = im_map.transpose(1, 0).contiguous()
        subim_map = subim_map.transpose(1, 0).contiguous()

        important_features = im_map * important_features
        subimportant_features = subim_map * subimportant_features

        htsa_out = torch.cat([important_features, subimportant_features], dim=1)
        htsa_out = self.conv_last(htsa_out)

        return htsa_out
