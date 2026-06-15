"""
SpecGateNet: a dual-branch (ResNet-50 CNN + Swin Transformer) network for
cloud / cloud-shadow semantic segmentation in remote-sensing imagery.

Key modules
-----------
* CrossFeatureFusionV1 : CNN <-> Transformer cross-branch interaction (1/4, 1/8, 1/16).
* DynamicFilter        : learnable frequency-domain (FFT) filtering at the bottleneck.
* PooledAFT2D          : pooled Attention-Free-Transformer attention (1/16 stage).
* SpectrumGuidedFusion : spectrum-gated decoder; an FFT energy map gates the fusion
                         of high-level and low-level features (the "SpecGate").
"""

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models.swin_transformer import _swin_transformer


def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)


class Conv2dBnRelu(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
        super(Conv2dBnRelu, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ConvBNGeLU(nn.Module):

    def __init__(self, in_chan, out_chan, kernel_size=1, stride=1, padding=0,
                 dilation=1, groups=1, bias=False):
        super(ConvBNGeLU, self).__init__()
        self.conv = nn.Conv2d(
            in_chan, out_chan, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.gelu = nn.GELU()

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.gelu(feat)
        return feat


class CBAM_channel_atten(nn.Module):
    def __init__(self, channel, reduction):
        super(CBAM_channel_atten, self).__init__()
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        return channel_out * x


class CBAM_spatical_atten(nn.Module):
    def __init__(self, spatial_kernel):
        super(CBAM_spatical_atten, self).__init__()
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        return spatial_out * x


class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        self.channel_atten = CBAM_channel_atten(channel, reduction)
        self.spatial_atten = CBAM_spatical_atten(spatial_kernel)

    def forward(self, x):
        x = self.channel_atten(x)
        return self.spatial_atten(x)


class CrossFeatureFusionV1(nn.Module):  # 用于自注意力机制
    def __init__(self, cnn_channels, transformer_channels):
        super(CrossFeatureFusionV1, self).__init__()
        self.con1x1_1 = ConvBNGeLU(in_chan=transformer_channels, out_chan=cnn_channels, kernel_size=1, stride=1,
                                   padding=0)   #一个1x1卷积层，用于将Transformer特征的通道数调整为与CNN特征相同的通道数。这样可以确保后续的特征融合操作时，它们的通道数一致。
        self.tf_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) #通过双线性插值将Transformer特征图上采样2倍。上采样的目标是使Transformer特征图的尺寸与CNN特征图的尺寸对齐，从而方便后续的融合操作。
        self.cbam = CBAM(cnn_channels)  #CBAM（Convolutional Block Attention Module）模块，用于在CNN特征上应用注意力机制，提升重要特征的表达能力。
        #cbam包括两个部份：（1）通道注意力（Channel Attention）：根据各通道的重要性对通道进行加权    （2）空间注意力（Spatial Attention）：根据空间位置的重要性对空间位置进行加权

        self.con1x1_2_1 = ConvBNGeLU(in_chan=cnn_channels * 2, out_chan=cnn_channels, kernel_size=1,
                                     stride=1,
                                     padding=0)     #一个1x1卷积层，用于将CNN和Transformer特征融合后的通道数（cnn_channels * 2）调整为CNN通道数。
        self.con1x1_2_2 = ConvBNGeLU(in_chan=cnn_channels * 2, out_chan=transformer_channels, kernel_size=3, stride=2,
                                     padding=1)     #一个3x3卷积层，用于将CNN和Transformer特征融合后的通道数（cnn_channels * 2）调整为Transformer特征通道数，并使用步幅为2的卷积来下采样特征图。

    def forward(self, x_cnn, x_tf):
        # self.tf_upsample(x_tf)：对调整后的Transformer特征图进行上采样，将其大小扩大2倍，以便与CNN特征图对齐
        x_tf = self.tf_upsample(x_tf.permute(0, 3, 1, 2))  # (N, H, W, C) -> (N, C, H, W)   调整Transformer特征图的维度，符合PyTorch中nn.Conv2d的输入格式
        x_tf = self.con1x1_1(x_tf)  #通过1x1卷积，将上采样后的Transformer特征图的通道数调整为与CNN特征图的通道数一致。
        x_cnn = self.cbam(x_cnn)    #对CNN特征图应用CBAM模块，增强重要的特征信息。CBAM会根据通道和空间的注意力机制对特征进行加权
        x = torch.cat((x_cnn, x_tf), dim=1)     #将处理后的CNN特征图和Transformer特征图进行拼接，拼接后的特征图通道数是原CNN和Transformer通道数的总和。
        return self.con1x1_2_1(x), self.con1x1_2_2(x).permute(0, 2, 3, 1)


class StarReLU(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4, out_features=None,
                 act_layer=StarReLU, drop=0., bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def resize_complex_weight(origin_weight, new_h, new_w):
    h, w, num_filters = origin_weight.shape[0:3]
    origin_weight = (origin_weight
                     .reshape(1, h, w, num_filters * 2)
                     .permute(0, 3, 1, 2))
    new_weight = F.interpolate(
        origin_weight, size=(new_h, new_w),
        mode='bicubic', align_corners=True
    ).permute(0, 2, 3, 1).reshape(new_h, new_w, num_filters, 2)
    return new_weight


class DynamicFilter(nn.Module):
    """Input/Output: [B, H, W, C]"""

    def __init__(self, dim, expansion_ratio=2, reweight_expansion_ratio=.25,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, num_filters=4, size=14, weight_resize=False, **kwargs):
        super().__init__()
        size = to_2tuple(size)
        self.size = size[0]
        self.filter_size = size[1] // 2 + 1
        self.num_filters = num_filters
        self.dim = dim
        self.med_channels = int(expansion_ratio * dim)
        self.weight_resize = weight_resize

        self.pwconv1 = nn.Linear(dim, self.med_channels, bias=bias)
        self.act1 = act1_layer()
        self.reweight = Mlp(dim, reweight_expansion_ratio,
                            num_filters * self.med_channels)
        self.complex_weights = nn.Parameter(
            torch.randn(self.size, self.filter_size, num_filters, 2,
                        dtype=torch.float32) * 0.02
        )
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(self.med_channels, dim, bias=bias)

    def forward(self, x):
        B, H, W, _ = x.shape
        routeing = (self.reweight(x.mean(dim=(1, 2)))
                    .view(B, self.num_filters, -1)
                    .softmax(dim=1))
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

        if self.weight_resize:
            complex_weights = resize_complex_weight(
                self.complex_weights, x.shape[1], x.shape[2])
            complex_weights = torch.view_as_complex(
                complex_weights.contiguous())
        else:
            complex_weights = torch.view_as_complex(self.complex_weights)

        routeing = routeing.to(torch.complex64)
        weight = torch.einsum('bfc,hwf->bhwc', routeing, complex_weights)

        if self.weight_resize:
            weight = weight.view(-1, x.shape[1], x.shape[2], self.med_channels)
        else:
            weight = weight.view(-1, self.size, self.filter_size, self.med_channels)

        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')
        x = self.act2(x)
        x = self.pwconv2(x)
        return x


class AFT_FULL(nn.Module):
    """Input/Output: [B, N, C]"""

    def __init__(self, d_model, n=49, simple=False):
        super().__init__()
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        if simple:
            self.position_biases = torch.zeros((n, n))
        else:
            self.position_biases = nn.Parameter(torch.ones((n, n)))
        self.d_model = d_model
        self.n = n
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        bs, n, dim = x.shape
        if n != self.n:
            raise ValueError(f"AFT_FULL expected n={self.n}, got n={n}.")
        q = self.fc_q(x)
        k = self.fc_k(x).view(1, bs, n, dim)
        v = self.fc_v(x).view(1, bs, n, dim)
        pb = self.position_biases.view(n, 1, -1, 1)
        numerator = torch.sum(torch.exp(k + pb) * v, dim=2)
        denominator = torch.sum(torch.exp(k + pb), dim=2)
        out = numerator / (denominator + 1e-6)
        out = self.sigmoid(q) * out.permute(1, 0, 2).contiguous()
        return out


class PooledAFT2D(nn.Module):
    """BCHW → pool → AFT → upsample → BCHW（带可学习门控）"""

    def __init__(self, channels, pool_size=7, simple=False,
                 use_residual=True, alpha_init=0.0):
        super().__init__()
        self.pool_size = int(pool_size)
        self.use_residual = use_residual
        self.aft = AFT_FULL(d_model=channels,
                            n=self.pool_size * self.pool_size,
                            simple=simple)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
        xp = F.adaptive_avg_pool2d(x, (self.pool_size, self.pool_size))
        xp = xp.permute(0, 2, 3, 1).contiguous().view(
            B, self.pool_size * self.pool_size, C)
        xp = self.aft(xp)
        xp = (xp.view(B, self.pool_size, self.pool_size, C)
              .permute(0, 3, 1, 2).contiguous())
        xp = F.interpolate(xp, size=(H, W), mode='bilinear', align_corners=True)
        xp = self.alpha * xp
        if self.use_residual:
            return shortcut + xp
        return xp


class SpectrumGuidedFusion(nn.Module):
    """频谱引导融合模块（SGF）"""

    def __init__(self, high_channels, low_channels, out_channels, gate_kernel=3):
        super().__init__()
        self.out_channels = out_channels

        self.high_align = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.low_align = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        mid = max(out_channels // 4, 8)
        self.gate_gen = nn.Sequential(
            nn.Conv2d(out_channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_channels,
                      kernel_size=gate_kernel,
                      padding=gate_kernel // 2,
                      bias=False),
            nn.Sigmoid(),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.gamma = nn.Parameter(torch.ones(1))

    def _spectrum_energy_map(self, x):
        x_f = torch.fft.rfft2(x.float(), norm='ortho')
        energy = torch.abs(x_f)
        energy_spatial = torch.fft.irfft2(
            energy, s=x.shape[-2:], norm='ortho'
        )
        return energy_spatial.to(x.dtype)

    def forward(self, high_feat, low_feat):
        h = self.high_align(high_feat)
        l = self.low_align(low_feat)
        energy = self._spectrum_energy_map(h)
        gate = self.gate_gen(energy)
        fused = l * gate + h * (1.0 - gate)
        out = self.refine(fused)
        out = out + self.gamma * h
        return out


class SpecGateNet(nn.Module):

    def __init__(
            self,
            n_classes=3,
            size=224,
            cnn_pretrained=False,
            dropout=0.1,
            cnn_backbone='resnet50',
            swin_embed_dim=64,
            swin_depths=(1, 1, 3, 1),
            swin_num_heads=(2, 4, 8, 16),
            patch_size=(4, 4),
            aux_loss=False,
            aft_pool_size=7,
            aft_simple=False,
            aft_use_residual=True,
            aft_alpha_init=0.1,
            dec_channels=64,
            sgf_gate_kernel=3,
    ):
        super().__init__()

        # ── CNN 骨架 ──
        if cnn_backbone.lower() == 'resnet18':
            encoder = torchvision.models.resnet34(
                weights='IMAGENET1K_V1' if cnn_pretrained else None)
            bottom_ch = 512
        elif cnn_backbone.lower() == 'resnet50':
            encoder = torchvision.models.resnet50(
                weights='IMAGENET1K_V2' if cnn_pretrained else None)
            bottom_ch = 2048
        elif cnn_backbone.lower() == 'resnet101':
            encoder = torchvision.models.resnet101(
                weights='IMAGENET1K_V2' if cnn_pretrained else None)
            bottom_ch = 2048
        elif cnn_backbone.lower() == 'resnet152':
            encoder = torchvision.models.resnet152(
                weights='IMAGENET1K_V2' if cnn_pretrained else None)
            bottom_ch = 2048
        else:
            raise NotImplementedError(f'{cnn_backbone} not implemented')

        self.size = size
        self.dec_channels = dec_channels

        # ── ResNet 各阶段 ──
        self.conv1 = Conv2dBnRelu(3, 64, kernel_size=7, stride=1, padding=3)
        self.pool = encoder.maxpool
        self.conv2_x = encoder.layer1  # 1/4,  ch = 256
        self.conv3_x = encoder.layer2  # 1/8,  ch = 512
        self.conv4_x = encoder.layer3  # 1/16, ch = 1024
        self.conv5_x = encoder.layer4  # 1/32, ch = 2048

        # ── Swin Transformer ──
        swin = _swin_transformer(
            patch_size=list(patch_size),
            embed_dim=swin_embed_dim,
            depths=list(swin_depths),
            num_heads=list(swin_num_heads),
            window_size=[7, 7],
            stochastic_depth_prob=0.2,
            weights=None,
            progress=True,
            num_classes=n_classes,
        )
        self.swin_partition = swin.features[0]
        self.s1 = swin.features[1]
        self.s2 = swin.features[2:4]
        self.s3 = swin.features[4:6]
        self.s4 = swin.features[6:]

        # ── ★ CFF 双路交互（新增） ──
        base_cnn = bottom_ch // 8  # 256
        base_tf = swin_embed_dim  # 64
        self.cff1 = CrossFeatureFusionV1(base_cnn, base_tf)  # 1/4
        self.cff2 = CrossFeatureFusionV1(base_cnn * 2, base_tf * 2)  # 1/8
        self.cff3 = CrossFeatureFusionV1(base_cnn * 4, base_tf * 4)  # 1/16

        # ── 瓶颈融合 + DynamicFilter ──
        fuse_in_ch = bottom_ch + swin_embed_dim * 8  # 2048+512=2560
        self.fuse_dim = 256
        self.fuse_reduce = nn.Conv2d(fuse_in_ch, self.fuse_dim, kernel_size=1)

        df_size = max(1, size // 16)
        self.dynamic_filter = DynamicFilter(
            dim=self.fuse_dim,
            size=df_size,
            weight_resize=True,
        )

        # ── PooledAFT2D（1/16 尺度） ──
        self.aft_x4 = PooledAFT2D(
            channels=dec_channels,
            pool_size=aft_pool_size,
            simple=aft_simple,
            use_residual=aft_use_residual,
            alpha_init=aft_alpha_init,
        )

        # ── 瓶颈降维 ──
        self.conv1x1_5 = nn.Conv2d(self.fuse_dim, dec_channels, kernel_size=1)

        # ── SGF 解码器 ──
        self.sgf_4 = SpectrumGuidedFusion(
            high_channels=dec_channels,
            low_channels=bottom_ch // 2,
            out_channels=dec_channels,
            gate_kernel=sgf_gate_kernel,
        )
        self.sgf_3 = SpectrumGuidedFusion(
            high_channels=dec_channels,
            low_channels=bottom_ch // 4,
            out_channels=dec_channels,
            gate_kernel=sgf_gate_kernel,
        )
        self.sgf_2 = SpectrumGuidedFusion(
            high_channels=dec_channels,
            low_channels=bottom_ch // 8,
            out_channels=dec_channels,
            gate_kernel=sgf_gate_kernel,
        )

        # ── 最终分类头 ──
        self.final = nn.Sequential(
            nn.Dropout2d(p=dropout),
            nn.Conv2d(dec_channels, n_classes, kernel_size=1),
        )

        # ── 辅助损失头 ──
        self.aux_loss = aux_loss
        if aux_loss:
            self.auxs = nn.ModuleList([
                nn.Sequential(
                    Conv2dBnRelu(dec_channels, dec_channels, 3, 1, 1),
                    nn.Dropout2d(p=dropout),
                    nn.Conv2d(dec_channels, n_classes, 1),
                )
                for _ in range(3)
            ])

    # ── 前向传播 ──

    def forward(self, data, data_aux=None):
        h, w = data.shape[2], data.shape[3]

        # ── CNN 主干 ──
        x = self.conv1(data)  # [B, 64, H, W]
        x1 = self.pool(x)

        # ── Swin 主干 ──
        y1 = self.swin_partition(data)  # [B, H/4, W/4, 64] (BHWC)

        # ── Stage 1: 1/4 + CFF1 双路交互 ──
        x2 = self.conv2_x(x1)  # [B, 256, H/4, W/4]
        y2 = self.s1(y1)  # [B, H/4, W/4, 64]
        z2_cnn, z2_tf = self.cff1(x2, y2)

        # ── Stage 2: 1/8 + CFF2 双路交互 ──
        x3 = self.conv3_x(x2 + z2_cnn)  # [B, 512, H/8, W/8]
        y3 = self.s2(y2 + z2_tf)  # [B, H/8, W/8, 128]
        z3_cnn, z3_tf = self.cff2(x3, y3)

        # ── Stage 3: 1/16 + CFF3 双路交互 ──
        x4 = self.conv4_x(x3 + z3_cnn)  # [B, 1024, H/16, W/16]
        y4 = self.s3(y3 + z3_tf)  # [B, H/16, W/16, 256]
        z4_cnn, z4_tf = self.cff3(x4, y4)

        # ── Stage 4: 1/32（不用CFF，直接concat） ──
        x5 = self.conv5_x(x4 + z4_cnn)  # [B, 2048, H/32, W/32]
        y5 = self.s4(y4 + z4_tf)  # [B, H/32, W/32, 512]
        y5 = y5.permute(0, 3, 1, 2).contiguous()
        y5 = F.interpolate(y5, size=x5.shape[-2:],
                           mode='bilinear', align_corners=True)

        # ── 瓶颈融合 ──
        feat = torch.cat((x5, y5), dim=1)  # [B, 2560, h5, w5]
        z = self.fuse_reduce(feat)  # [B, 256, h5, w5]

        # ── DynamicFilter（BHWC 格式） ──
        z = z.permute(0, 2, 3, 1).contiguous()
        z = self.dynamic_filter(z)
        z = z.permute(0, 3, 1, 2).contiguous()

        # ── 瓶颈降维 ──
        x5_up = self.conv1x1_5(z)  # [B, dec_ch, h5, w5]

        # ── 1/16 融合（SGF_4） ──
        x5_up_16 = F.interpolate(x5_up, size=x4.shape[2:],
                                 mode='bilinear', align_corners=True)
        x4_feat = self.sgf_4(x5_up_16, x4)

        # ── PooledAFT2D ──
        x4_feat = self.aft_x4(x4_feat)

        # ── 1/8 融合（SGF_3） ──
        x4_up = F.interpolate(x4_feat, size=x3.shape[2:],
                              mode='bilinear', align_corners=True)
        x3_feat = self.sgf_3(x4_up, x3)

        # ── 1/4 融合（SGF_2） ──
        x3_up = F.interpolate(x3_feat, size=x2.shape[2:],
                              mode='bilinear', align_corners=True)
        x2_feat = self.sgf_2(x3_up, x2)

        # ── 上采样到原始分辨率 ──
        out_feat = F.interpolate(x2_feat, size=x.shape[2:],
                                 mode='bilinear', align_corners=True)
        out_feat = out_feat + x
        out = self.final(out_feat)

        if out.shape[2:] != (h, w):
            out = F.interpolate(out, size=(h, w),
                                mode='bilinear', align_corners=True)

        # ── 辅助损失输出 ──
        if self.aux_loss and self.training:
            aux1 = self.auxs[0](x4_feat)
            aux1 = F.interpolate(aux1, size=(h, w),
                                 mode='bilinear', align_corners=True)
            aux2 = self.auxs[1](x3_feat)
            aux2 = F.interpolate(aux2, size=(h, w),
                                 mode='bilinear', align_corners=True)
            aux3 = self.auxs[2](x2_feat)
            aux3 = F.interpolate(aux3, size=(h, w),
                                 mode='bilinear', align_corners=True)
            return out, aux1, aux2, aux3

        return out


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    model = SpecGateNet(
        n_classes=7,
        size=224,
        cnn_pretrained=False,
        cnn_backbone='resnet50',
        aux_loss=False,
        aft_alpha_init=0.1,
    ).to(device)

    dummy = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        out = model(dummy)
    print(f'Input : {tuple(dummy.shape)}')
    print(f'Output: {tuple(out.shape)}')

    total = sum(p.numel() for p in model.parameters())
    print(f'Total params: {total / 1e6:.2f} M')
