"""
MedViTv3: Improved MedViT with Parallel Global/Local Branches and Cleaned Architecture.
Supports WavKAN integration.
"""
from functools import partial
import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
import natten
from natten import NeighborhoodAttention2D as NeighborhoodAttention
from fasterkan import FasterKAN, SplineLinear
try:
    from wavkan import WavKAN
except ImportError:
    WavKAN = None

is_natten_post_017 = hasattr(natten, "context")

NORM_EPS = 1e-5

def merge_pre_bn(module, pre_bn_1, pre_bn_2=None):
    """ Merge pre BN to reduce inference runtime. """
    weight = module.weight.data
    if module.bias is None:
        zeros = torch.zeros(module.out_channels, device=weight.device).type(weight.type())
        module.bias = nn.Parameter(zeros)
    bias = module.bias.data
    if pre_bn_2 is None:
        scale_invstd = pre_bn_1.running_var.add(pre_bn_1.eps).pow(-0.5)
        extra_weight = scale_invstd * pre_bn_1.weight
        extra_bias = pre_bn_1.bias - pre_bn_1.weight * pre_bn_1.running_mean * scale_invstd
    else:
        scale_invstd_1 = pre_bn_1.running_var.add(pre_bn_1.eps).pow(-0.5)
        scale_invstd_2 = pre_bn_2.running_var.add(pre_bn_2.eps).pow(-0.5)
        extra_weight = scale_invstd_1 * pre_bn_1.weight * scale_invstd_2 * pre_bn_2.weight
        extra_bias = scale_invstd_2 * pre_bn_2.weight *(pre_bn_1.bias - pre_bn_1.weight * pre_bn_1.running_mean * scale_invstd_1 - pre_bn_2.running_mean) + pre_bn_2.bias

    if isinstance(module, nn.Linear):
        extra_bias = weight @ extra_bias
        weight.mul_(extra_weight.view(1, weight.size(1)).expand_as(weight))
    elif isinstance(module, nn.Conv2d):
        weight = weight.reshape(weight.shape[0], weight.shape[1])
        extra_bias = weight @ extra_bias
        weight.mul_(extra_weight.view(1, weight.size(1)).expand_as(weight))
        weight = weight.reshape(weight.shape[0], weight.shape[1], 1, 1)
    bias.add_(extra_bias)

    module.weight.data = weight
    module.bias.data = bias

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=1, groups=groups, bias=False)
        self.norm = nn.BatchNorm2d(out_channels, eps=NORM_EPS)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class PatchEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(PatchEmbed, self).__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        if stride == 2:
            self.avgpool = nn.AvgPool2d((2, 2), stride=2, ceil_mode=True, count_include_pad=False)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.norm = norm_layer(out_channels)
        elif in_channels != out_channels:
            self.avgpool = nn.Identity()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.norm = norm_layer(out_channels)
        else:
            self.avgpool = nn.Identity()
            self.conv = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x):
        return self.norm(self.conv(self.avgpool(x)))

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_h * a_w
        return out

class MHCA(nn.Module):
    """ Multi-Head Convolutional Attention """
    def __init__(self, out_channels, head_dim, use_coord=False):
        super(MHCA, self).__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        self.group_conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                                       padding=1, groups=out_channels // head_dim, bias=False)
        self.norm = norm_layer(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.projection = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.use_coord = use_coord
        if self.use_coord:
            self.coord = CoordAtt(out_channels, out_channels)

    def forward(self, x):
        out = self.group_conv3x3(x)
        out = self.norm(out)
        out = self.act(out)
        if self.use_coord:
            out = self.coord(out)
        out = self.projection(out)
        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class LocalityFeedForwardV3(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, expand_ratio=4., act='hs+se', reduction=4):
        super(LocalityFeedForwardV3, self).__init__()
        hidden_dim = int(in_dim * expand_ratio)
        layers = []
        layers.extend([
            nn.Conv2d(in_dim, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            h_swish() if act.find('hs') >= 0 else nn.ReLU6(inplace=True)])

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            h_swish() if act.find('hs') >= 0 else nn.ReLU6(inplace=True)
        ])

        if act.find('+se') >= 0:
            layers.append(SELayer(hidden_dim, reduction=reduction))

        layers.extend([
            nn.Conv2d(hidden_dim, out_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_dim)
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Mlp(nn.Module):
    def __init__(self, in_features, out_features=None, mlp_ratio=None, drop=0., bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_dim = _make_divisible(in_features * mlp_ratio, 32)
        self.conv1 = nn.Conv2d(in_features, hidden_dim, kernel_size=1, bias=bias)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_dim, out_features, kernel_size=1, bias=bias)
        self.drop = nn.Dropout(drop)

    def merge_bn(self, pre_norm):
        merge_pre_bn(self.conv1, pre_norm)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x

class LFPv3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, path_dropout=0.2,
                 drop=0, head_dim=32, mlp_ratio=3):
        super(LFPv3, self).__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        
        self.patch_embed = PatchEmbed(in_channels, out_channels, stride)
        self.norm1 = norm_layer(out_channels)
        
        extra_args = {"rel_pos_bias": True} if is_natten_post_017 else {"bias": True}
        self.attn = NeighborhoodAttention(
            out_channels,
            kernel_size=7,
            dilation=None,
            num_heads=(out_channels // head_dim),
            qkv_bias=True,
            qk_scale=None,
            attn_drop=drop,
            proj_drop=0.0,
            **extra_args,
        )
        self.attention_path_dropout = DropPath(path_dropout)
        
        self.norm2 = norm_layer(out_channels)
        self.ffn = LocalityFeedForwardV3(out_channels, out_channels, kernel_size, 1, mlp_ratio, reduction=out_channels)
        self.is_bn_merged = False

    def merge_bn(self):
        pass

    def forward(self, x):
        x = self.patch_embed(x)
        b, c, h, w = x.shape
        shortcut = x
        
        x = self.norm1(x)
        x = self.attn(x.reshape(b, h, w, c))
        x = shortcut + self.attention_path_dropout(x.reshape(b, c, h, w))
        
        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            out = self.norm2(x)
        else:
            out = x
        x = x + self.ffn(out)
        return x

class E_MHSA(nn.Module):
    """ Efficient Multi-Head Self Attention """
    def __init__(self, dim, out_dim=None, head_dim=32, qkv_bias=True, qk_scale=None,
                 attn_drop=0, proj_drop=0., sr_ratio=1):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else dim
        self.num_heads = self.dim // head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.v = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.proj = nn.Linear(self.dim, self.out_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        self.N_ratio = sr_ratio ** 2
        if sr_ratio > 1:
            self.sr = nn.AvgPool1d(kernel_size=self.N_ratio, stride=self.N_ratio)
            self.norm = nn.BatchNorm1d(dim, eps=NORM_EPS)
        self.is_bn_merged = False

    def merge_bn(self, pre_bn):
        merge_pre_bn(self.q, pre_bn)
        if self.sr_ratio > 1:
            merge_pre_bn(self.k, pre_bn, self.norm)
            merge_pre_bn(self.v, pre_bn, self.norm)
        else:
            merge_pre_bn(self.k, pre_bn)
            merge_pre_bn(self.v, pre_bn)
        self.is_bn_merged = True

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x)
        q = q.reshape(B, N, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2)
            x_ = self.sr(x_)
            if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
                x_ = self.norm(x_)
            x_ = x_.transpose(1, 2)
            k = self.k(x_)
            k = k.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 3, 1)
            v = self.v(x_)
            v = v.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)
        else:
            k = self.k(x)
            k = k.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 3, 1)
            v = self.v(x)
            v = v.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)
        attn = (q @ k) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class GFPv3(nn.Module):
    """
    Parallel Global-Local Transformer Block with WavKAN support
    """
    def __init__(
            self, in_channels, out_channels, path_dropout, stride=1, sr_ratio=1,
            mlp_ratio=2, head_dim=32, mix_block_ratio=0.75, attn_drop=0, drop=0,
            use_kmp_glu=False, use_coord=False, use_wavkan=False,
    ):
        super(GFPv3, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mix_block_ratio = mix_block_ratio
        self.use_kmp_glu = use_kmp_glu
        norm_func = partial(nn.BatchNorm2d, eps=NORM_EPS)

        # Calculate split channels
        self.mhsa_out_channels = _make_divisible(int(out_channels * mix_block_ratio), 32)
        self.mhca_out_channels = out_channels - self.mhsa_out_channels

        # Input projection
        self.patch_embed = PatchEmbed(in_channels, out_channels, stride)
        
        # Global Branch (MHSA)
        self.norm_global = norm_func(self.mhsa_out_channels)
        self.e_mhsa = E_MHSA(self.mhsa_out_channels, head_dim=head_dim, sr_ratio=sr_ratio,
                             attn_drop=attn_drop, proj_drop=drop)
        self.mhsa_path_dropout = DropPath(path_dropout * mix_block_ratio)

        # Local Branch (MHCA)
        self.mhca = MHCA(self.mhca_out_channels, head_dim=head_dim, use_coord=use_coord)
        self.mhca_path_dropout = DropPath(path_dropout * (1 - mix_block_ratio))

        # FFN Section
        self.norm2 = norm_func(out_channels)
        self.mlp_path_dropout = DropPath(path_dropout)
        
        hidden_dim = int(out_channels * mlp_ratio)
        
        if use_wavkan and WavKAN is not None:
            self.kan = WavKAN([out_channels, hidden_dim, out_channels])
        else:
            self.kan = FasterKAN([out_channels, hidden_dim, out_channels])
        
        if self.use_kmp_glu:
            self.mlp = Mlp(out_channels, mlp_ratio=mlp_ratio, drop=drop)
            self.kan_scale = nn.Parameter(torch.zeros(out_channels, 1, 1), requires_grad=True)
        else:
            self.mlp = None

        self.is_bn_merged = False

    def merge_bn(self):
        if not self.is_bn_merged:
            self.e_mhsa.merge_bn(self.norm_global)
            if self.use_kmp_glu and self.mlp is not None:
                self.mlp.merge_bn(self.norm2)
            self.is_bn_merged = True

    def forward(self, x):
        # 1. Patch Embedding & Stride
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        
        # 2. Split into Global and Local branches
        x_global, x_local = torch.split(x, [self.mhsa_out_channels, self.mhca_out_channels], dim=1)
        
        # 3. Global Branch Processing
        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            out_global = self.norm_global(x_global)
        else:
            out_global = x_global
        out_global = rearrange(out_global, "b c h w -> b (h w) c")
        out_global = self.e_mhsa(out_global)
        out_global = rearrange(out_global, "b (h w) c -> b c h w", h=H)
        x_global = x_global + self.mhsa_path_dropout(out_global)
        
        # 4. Local Branch Processing
        out_local = self.mhca(x_local)
        x_local = x_local + self.mhca_path_dropout(out_local)
        
        # 5. Concatenate
        x = torch.cat([x_global, x_local], dim=1)

        # 6. FFN (KAN / MLP)
        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            out = self.norm2(x)
        else:
            out = x
            
        if self.use_kmp_glu:
            b, d, h, w = out.shape
            # KAN processing
            kan_out = self.kan(out.permute(0, 2, 3, 1).reshape(-1, d))
            kan_out = kan_out.view(b, h, w, d).permute(0, 3, 1, 2)
            
            # MLP processing
            mlp_out = self.mlp(out)
            
            # Fusion
            fused = mlp_out + self.kan_scale * kan_out
            x = x + self.mlp_path_dropout(fused)
        else:
            b, d, t, _ = out.shape
            # Pure KAN FFN
            x = x + self.mlp_path_dropout(self.kan(out.reshape(-1, out.shape[1])).reshape(b, d, t, t))
            
        return x

class MedViTv3(nn.Module):
    def __init__(self, stem_chs=[64, 32, 64], depths=[2, 2, 6, 2],
                 dims=[64, 128, 320, 512], path_dropout=0.1, attn_drop=0,
                 drop=0, num_classes=1000,
                 strides=[1, 2, 2, 2], sr_ratios=[8, 4, 2, 1], head_dim=32, mix_block_ratio=0.75,
                 use_checkpoint=False, use_kmp_glu=False, use_coord=False, use_wavkan=False):
        super(MedViTv3, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.use_kmp_glu = use_kmp_glu
        self.use_coord = use_coord
        self.use_wavkan = use_wavkan

        self.stage_out_channels = [[dims[0]] * (depths[0]),
                                   [dims[1]] * (depths[1] - 1) + [dims[1]],
                                   [dims[2], dims[2], dims[2]] * (depths[2] // 3),
                                   [dims[3]] * (depths[3])]

        # Use LFPv3 and GFPv3
        self.stage_block_types = [[LFPv3] * depths[0],
                                  [LFPv3] * (depths[1] - 1) + [GFPv3],
                                  [LFPv3, LFPv3, GFPv3] * (depths[2] // 3),
                                  [GFPv3] * (depths[3])]

        self.stem = nn.Sequential(
            ConvBNReLU(3, stem_chs[0], kernel_size=3, stride=2),
            ConvBNReLU(stem_chs[0], stem_chs[1], kernel_size=3, stride=1),
            ConvBNReLU(stem_chs[1], stem_chs[2], kernel_size=3, stride=1),
            ConvBNReLU(stem_chs[2], stem_chs[2], kernel_size=3, stride=2),
        )
        input_channel = stem_chs[-1]
        features = []
        idx = 0
        dpr = [x.item() for x in torch.linspace(0, path_dropout, sum(depths))]
        
        for stage_id in range(len(depths)):
            kernel = 7 if stage_id == 0 else 3
            numrepeat = depths[stage_id]
            output_channels = self.stage_out_channels[stage_id]
            block_types = self.stage_block_types[stage_id]
            for block_id in range(numrepeat):
                if strides[stage_id] == 2 and block_id == 0:
                    stride = 2
                else:
                    stride = 1
                output_channel = output_channels[block_id]
                block_type = block_types[block_id]
                
                if block_type is LFPv3:
                    layer = LFPv3(input_channel, output_channel, stride=stride, kernel_size=kernel, path_dropout=dpr[idx + block_id],
                                drop=drop, head_dim=head_dim)
                    features.append(layer)
                elif block_type is GFPv3:
                    layer = GFPv3(input_channel, output_channel, path_dropout=dpr[idx + block_id], stride=stride,
                                sr_ratio=sr_ratios[stage_id], head_dim=head_dim, mix_block_ratio=mix_block_ratio,
                                attn_drop=attn_drop, drop=drop, use_kmp_glu=self.use_kmp_glu, use_coord=self.use_coord,
                                use_wavkan=self.use_wavkan)
                    features.append(layer)
                input_channel = output_channel
            idx += numrepeat
        self.features = nn.Sequential(*features)

        self.norm = nn.BatchNorm2d(output_channel, eps=NORM_EPS)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj_head = nn.Sequential(
            nn.Linear(output_channel, num_classes),
        )

        self._initialize_weights()

    def merge_bn(self):
        self.eval()
        for idx, module in self.named_modules():
            if isinstance(module, LFPv3) or isinstance(module, GFPv3):
                module.merge_bn()

    def _initialize_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if isinstance(m, SplineLinear):
                    continue
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        for idx, layer in enumerate(self.features):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(layer, x)
            else:
                x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.proj_head(x)
        return x

@register_model
def MedViTv3_tiny(pretrained=False, **kwargs):
    model = MedViTv3(stem_chs=[64, 32, 64], depths=[2, 2, 6, 1], dims=[64, 128, 192, 384], path_dropout=0.1, **kwargs)
    return model

@register_model
def MedViTv3_small(pretrained=False, **kwargs):
    model = MedViTv3(stem_chs=[64, 32, 64], depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], path_dropout=0.1, **kwargs)
    return model

@register_model
def MedViTv3_base(pretrained=False, **kwargs):
    model = MedViTv3(stem_chs=[64, 32, 64], depths=[2, 2, 6, 2], dims=[96, 192, 384, 768], path_dropout=0.2, **kwargs)
    return model

@register_model
def MedViTv3_large(pretrained=False, **kwargs):
    model = MedViTv3(stem_chs=[64, 32, 64], depths=[2, 2, 6, 2], dims=[96, 256, 512, 1024], path_dropout=0.2, **kwargs)
    return model