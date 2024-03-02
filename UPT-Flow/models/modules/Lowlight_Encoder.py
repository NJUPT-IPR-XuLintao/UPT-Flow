import numbers
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange, repeat
import math



class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.channel = channel
        self.k_size = k_size

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

    def flops(self):
        flops = 0
        flops += self.channel * self.channel * self.k_size

        return flops


class eca_layer_1d(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer_1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.channel = channel
        self.k_size = k_size

    def forward(self, x):
        # b hw c
        # feature descriptor on the global spatial information
        y = self.avg_pool(x.transpose(-1, -2))

        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2))

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

    def flops(self):
        flops = 0
        flops += self.channel * self.channel * self.k_size

        return flops


class SepConv2d(nn.Module):
    def __init__(self,
                 in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, act_layer=nn.ReLU):
        super(SepConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                   padding=padding, dilation=dilation, groups=in_channels)

        # self.act_layer = act_layer() if act_layer is not None else nn.Identity()
        # self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = self.depthwise(x)
        # x = self.act_layer(x)
        # x = self.pointwise(x)
        return x

    def flops(self, HW):
        flops = 0
        flops += HW * self.in_channels * self.kernel_size ** 2 / self.stride ** 2
        flops += HW * self.in_channels * self.out_channels
        print("SeqConv2d:{%.2f}" % (flops / 1e9))
        return flops


######## Embedding for q,k,v ########
class ConvProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, kernel_size=3, stride=1):
        super().__init__()

        inner_dim = dim_head * heads
        self.heads = heads
        pad = (kernel_size - stride) // 2
        # self.to_q = SepConv2d(dim, inner_dim, kernel_size, stride, pad)
        # self.to_k = SepConv2d(dim, inner_dim, kernel_size, stride, pad)
        # self.to_v = SepConv2d(dim, inner_dim, kernel_size, stride, pad)
        self.to_q = nn.Conv2d(dim, inner_dim, kernel_size=kernel_size, stride=stride, padding=pad, groups=dim)
        self.to_k = nn.Conv2d(dim, inner_dim, kernel_size=kernel_size, stride=stride, padding=pad, groups=dim)
        self.to_v = nn.Conv2d(dim, inner_dim, kernel_size=kernel_size, stride=stride, padding=pad, groups=dim)

    def forward(self, x, zero_map):
        b, n, c = x.shape
        h = self.heads
        l = int(math.sqrt(n))
        w = int(math.sqrt(n))

        attn_qk = zero_map
        attn_qk = rearrange(attn_qk, 'b (l w) c -> b c l w', l=l, w=w)
        q = self.to_q(attn_qk)
        k = self.to_k(attn_qk)

        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)

        x = rearrange(x, 'b (l w) c -> b c l w', l=l, w=w)

        # print(attn_kv)
        v = self.to_v(x)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)

        return q, k, v

    def flops(self, q_L, kv_L=None):
        kv_L = kv_L or q_L
        flops = 0
        flops += self.to_q.flops(q_L)
        flops += self.to_k.flops(kv_L)
        flops += self.to_v.flops(kv_L)
        return flops


class LinearProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_v = nn.Linear(dim, inner_dim, bias=bias)
        self.to_qk = nn.Linear(dim, inner_dim * 2, bias=False)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, zero_map):
        B_, N, C = x.shape

        N_kv = zero_map.size(1)
        v = self.to_v(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        qk = self.to_qk(zero_map).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]
        v = v[0]

        return q, k, v

    def flops(self, q_L, kv_L=None):
        kv_L = kv_L or q_L
        flops = q_L * self.dim * self.inner_dim + kv_L * self.dim * self.inner_dim * 2
        return flops

    #########################################

class WindowAttention(nn.Module):
    def __init__(self, dim, win_size, num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0])  # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1])  # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        if token_projection == 'conv':
            self.qkv = ConvProjection(dim, num_heads, dim // num_heads)
        elif token_projection == 'linear':
            self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
            # self.qkv = nn.Linear(dim, dim, bias=qkv_bias)
        else:
            raise Exception("Projection error!")

        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None, zero_map=None):
        B_, N, C = x.shape

        q, k, v = self.qkv(x, zero_map)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) * self.temperature

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1) // relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d=ratio)

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)', d=ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N * ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N * ratio)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def flops(self, H, W):
        # calculate flops for 1 window with token length of N
        # print(N, self.dim)
        flops = 0
        N = self.win_size[0] * self.win_size[1]
        nW = H * W / N
        # qkv = self.qkv(x)
        # flops += N * self.dim * 3 * self.dim
        flops += self.qkv.flops(H * W, H * W)

        # attn = (q @ k.transpose(-2, -1))

        flops += nW * self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += nW * self.num_heads * N * N * (self.dim // self.num_heads)

        # x = self.proj(x)
        flops += nW * N * self.dim * self.dim
        print("W-MSA:{%.2f}" % (flops / 1e9))
        return flops

    def flops(self, q_num, kv_num):
        # calculate flops for 1 window with token length of N
        # print(N, self.dim)
        flops = 0
        # N = self.win_size[0]*self.win_size[1]
        # nW = H*W/N
        # qkv = self.qkv(x)
        # flops += N * self.dim * 3 * self.dim
        flops += self.qkv.flops(q_num, kv_num)
        # attn = (q @ k.transpose(-2, -1))

        flops += self.num_heads * q_num * (self.dim // self.num_heads) * kv_num
        #  x = (attn @ v)
        flops += self.num_heads * q_num * (self.dim // self.num_heads) * kv_num

        # x = self.proj(x)
        flops += q_num * self.dim * self.dim
        print("MCA:{%.2f}" % (flops / 1e9))
        return flops


#########################################
class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, use_eca=True):
        super().__init__()

        self.conv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=True)
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer())
        self.conv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=True)
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.eca = eca_layer(dim) if use_eca else nn.Identity()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)

        x = self.conv1(x)

        # spatial restore
        x = self.dwconv(x)
        x = self.conv2(x)

        x = self.eca(x)

        return x

    def flops(self, H, W):
        flops = 0
        # fc1
        flops += H * W * self.dim * self.hidden_dim
        # dwconv
        flops += H * W * self.hidden_dim * 3 * 3
        # fc2
        flops += H * W * self.hidden_dim * self.dim
        print("LeFF:{%.2f}" % (flops / 1e9))
        # eca
        if hasattr(self.eca, 'flops'):
            flops += self.eca.flops()
        return flops


#########################################
########### window operation#############
def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate != 1:
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1),
                     stride=win_size)  # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0, 2, 1).contiguous().view(-1, C, win_size, win_size)  # B' ,C ,Wh ,Ww
        windows = windows.permute(0, 2, 3, 1).contiguous()  # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C)  # B' ,Wh ,Ww ,C
    return windows


def window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate != 1:
        x = windows.permute(0, 5, 3, 4, 1, 2).contiguous()  # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1),
                   stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class LeWinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, win_size=8, shift_size=0, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, token_projection='linear',
                 token_mlp='leff'):
        super().__init__()
        self.dim = dim

        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp

        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
                                    token_projection=token_projection)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LeFF(dim, mlp_hidden_dim, act_layer=act_layer)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, x, zero_map=None):
        B, C, H, W = x.shape
        # H = int(math.sqrt(L))
        # W = int(math.sqrt(L))
        # shortcut = x.view(B, -1, C)
        x = x.permute(0, 2, 3, 1)  # b h w c
        shortcut = x

        zero_map = zero_map.permute(0, 2, 3, 1)
        # x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.win_size - W % self.win_size) % self.win_size
        pad_b = (self.win_size - H % self.win_size) % self.win_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))  # [B,H,W,C]
        zero_map = F.pad(zero_map, (0, 0, pad_l, pad_r, pad_t, pad_b))  # [B,H,W,C]

        attn_mask = None
        ## shift mask
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            shift_mask = torch.zeros((1, H + pad_b, W + pad_r, 1)).type_as(x)
            h_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1
            shift_mask_windows = window_partition(shift_mask, self.win_size)  # nW, win_size, win_size, 1
            shift_mask_windows = shift_mask_windows.view(-1, self.win_size * self.win_size)  # nW, win_size*win_size
            shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(
                2)  # nW, win_size*win_size, win_size*win_size
            shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0)).masked_fill(
                shift_attn_mask == 0, float(0.0))
            attn_mask = shift_attn_mask

        x = self.norm1(x)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_zero = torch.roll(zero_map, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            shifted_zero = zero_map

        # partition windows
        x_windows = window_partition(shifted_x, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C

        zero_windows = window_partition(shifted_zero, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        zero_windows = zero_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask, zero_map=zero_windows)  # nW*B, win_size*win_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        shifted_x = window_reverse(attn_windows, self.win_size, H + pad_b, W + pad_r)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x[:, :H, :W, :]
        # x = x.contiguous().view(B, H * W, C)
        # x = x.permute(0, 3, 1, 2)
        # FFN
        x = shortcut + self.drop_path(x)
        x = x.permute(0, 3, 1, 2) + self.drop_path(self.mlp(self.norm2(x)))
        # x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W).contiguous()
        del attn_mask, zero_windows
        return x

    def flops(self):
        flops = 0
        H, W = self.input_resolution

        if self.cross_modulator is not None:
            flops += self.dim * H * W
            flops += self.cross_attn.flops(H * W, self.win_size * self.win_size)

        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        flops += self.attn.flops(H, W)
        # norm2
        flops += self.dim * H * W
        # mlp
        flops += self.mlp.flops(H, W)
        # print("LeWin:{%.2f}"%(flops/1e9))
        return flops


#########################################
########### Basic layer of Uformer ################
class BasicUformerLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, win_size, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 token_projection='linear', token_mlp='ffn', shift_flag=True):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # build blocks
        if shift_flag:
            self.blocks = nn.ModuleList([
                LeWinTransformerBlock(dim=dim, num_heads=num_heads, win_size=win_size,
                                      shift_size=0 if (i % 2 == 0) else win_size // 2,
                                      mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                      drop=drop, attn_drop=attn_drop,
                                      drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                      norm_layer=norm_layer, token_projection=token_projection, token_mlp=token_mlp)
                for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                LeWinTransformerBlock(dim=dim, num_heads=num_heads, win_size=win_size,
                                      shift_size=0, mlp_ratio=mlp_ratio,
                                      qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                                      drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                      norm_layer=norm_layer, token_projection=token_projection, token_mlp=token_mlp)
                for i in range(depth)])

    def forward(self, x, zero_map=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, zero_map)
        return x

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops


#####################################################################################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.qkv(x)
        qkv = self.qkv_dwconv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(2, 3)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))
        # self.body = nn.Sequential(nn.PixelUnshuffle(2),
        #                           nn.Conv2d(n_feat * 4, n_feat, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
#Noticed win_size in line727, LOL is 5, the others are 8
class MSFormer(nn.Module):
    def __init__(self, inp_channels=3, dim=48, num_refinement_blocks=2, heads=[2, 2, 4, 8],
                 ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias',
                 depths=[2, 2, 2, 2, 2, 2, 2], num_heads=[1, 2, 4, 8, 8, 4, 2],
                 win_size=5, drop_path_rate=0.1, token_projection='conv', token_mlp='leff', shift_flag=True):
        super(MSFormer, self).__init__()


        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.dim = dim
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(depths[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(depths[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(depths[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(depths[3])])
        self.reduce_chan_level0 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(depths[4])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(depths[5])])

        self.up2_1 = Upsample(int(dim * 2 ** 2))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.reduce_chan_level1 = nn.Conv2d(int(dim * 3), int(dim * 2), kernel_size=1, bias=bias)
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(depths[6])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        ############################################################################################################################
        self.num_enc_layers = len(depths) // 2
        self.num_dec_layers = len(depths) // 2
        self.pos_drop = nn.Dropout(p=0)

        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate] * depths[3]
        dec_dpr = enc_dpr[::-1]

        # build layers
        # Encoder
        self.encoderlayer_1 = BasicUformerLayer(dim=dim, depth=depths[0], num_heads=num_heads[0],
                                                win_size=win_size,
                                                drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                shift_flag=shift_flag)
        self.encoderlayer_2 = BasicUformerLayer(dim=dim * 2, depth=depths[1], num_heads=num_heads[1],
                                                win_size=win_size,
                                                drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                shift_flag=shift_flag)
        self.encoderlayer_3 = BasicUformerLayer(dim=dim * 4, depth=depths[2], num_heads=num_heads[2],
                                                win_size=win_size,
                                                drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])],
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                shift_flag=shift_flag)
        # Bottleneck
        self.bottlen = BasicUformerLayer(dim=dim * 8, depth=depths[3], num_heads=num_heads[3], win_size=win_size,
                                         drop_path=conv_dpr, token_projection=token_projection, token_mlp=token_mlp,
                                         shift_flag=shift_flag)

        self.decoderlayer_1 = BasicUformerLayer(dim=dim * 4, depth=depths[4], num_heads=num_heads[4],
                                                win_size=win_size,
                                                drop_path=dec_dpr[:depths[4]], token_projection=token_projection,
                                                token_mlp=token_mlp, shift_flag=shift_flag)
        self.decoderlayer_2 = BasicUformerLayer(dim=dim * 4, depth=depths[5], num_heads=num_heads[5],
                                                win_size=win_size,
                                                drop_path=dec_dpr[sum(depths[4:5]):sum(depths[4:6])],
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                shift_flag=shift_flag)
        self.decoderlayer_3 = BasicUformerLayer(dim=dim * 2, depth=depths[6], num_heads=num_heads[6],
                                                win_size=win_size,
                                                drop_path=dec_dpr[sum(depths[4:6]):sum(depths[4:7])],
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                shift_flag=shift_flag)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    ############################################################################################################################

    def forward(self, inp_img, zc):

        result = {}
        inp_enc_level1 = self.patch_embed(inp_img)
        zero_map = zc.repeat(1, self.dim, 1, 1)

        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        conv1 = self.encoderlayer_1(inp_enc_level1, zero_map=zero_map)
        # 1,48,144,144
        x1 = out_enc_level1 + conv1

        inp_enc_level2 = self.down1_2(x1)
        zero_map_down1 = self.down1_2(zero_map)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        conv2 = self.encoderlayer_2(out_enc_level2, zero_map=zero_map_down1)  # 1,96,72,72
        # conv2 = self.encoderlayer_2(inp_enc_level2, zero_map=zero_map_down1)  # 1,96,72,72
        x2 = out_enc_level2 + conv2

        inp_enc_level3 = self.down2_3(x2)
        zero_map_down2 = self.down2_3(zero_map_down1)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        conv3 = self.encoderlayer_3(inp_enc_level3, zero_map=zero_map_down2)  # 1,192,36,36
        x3 = out_enc_level3 + conv3

        inp_enc_level4 = self.down3_4(x3)
        zero_map_down3 = self.down3_4(zero_map_down2)

        latent = self.latent(inp_enc_level4)
        bottlen = self.bottlen(inp_enc_level4, zero_map=zero_map_down3)

        latent = latent + bottlen

        # 1,348,18,18
        fea_up0 = self.reduce_chan_level0(latent)
        result['fea_up0'] = fea_up0

        inp_dec_level3 = self.up4_3(latent)
        zero_map_up3 = self.up4_3(zero_map_down3)

        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        up1 = self.decoderlayer_1(inp_dec_level3, zero_map=zero_map_up3)

        out_dec_level3 = out_dec_level3 + up1
        # 1,192,36,36
        result['fea_up1'] = out_dec_level3

        inp_dec_level2 = self.up3_2(out_dec_level3)
        zero_map_up2 = self.up3_2(zero_map_up3)

        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        zero_map_up2 = zero_map_up2.repeat(1, 2, 1, 1)
        up2 = self.decoderlayer_2(inp_dec_level2, zero_map=zero_map_up2)
        out_dec_level2 = out_dec_level2 + up2
        # 1,192,72,72
        result['fea_up2'] = out_dec_level2

        inp_dec_level1 = self.up2_1(out_dec_level2)
        zero_map_up1 = self.up2_1(zero_map_up2)

        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        up3 = self.decoderlayer_3(inp_dec_level1, zero_map=zero_map_up1)
        out_dec_level1 = out_dec_level1 + up3

        out_dec_level0 = self.refinement(out_dec_level1)
        # out_dec_level0 = self.reduce_chan_level0(out_dec_level0)
        # 1,96,144,144
        result['cat_f'] = out_dec_level0

        #### For Dual-Pixel Defocus Deblurring Task ####
        # if self.dual_pixel_task:#false
        #     out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
        #     out_dec_level1 = self.output(out_dec_level1)
        ###########################

        # out = self.output(out_dec_level0) + inp_img
        # # out = self.skip(out)
        # result['color_map'] = out

        return result


if __name__ == '__main__':
    x = torch.randn((1, 3, 400, 600))
    y = torch.zeros([1, 1, 400, 600])
    model = MSFormer()

    out = model(x, y)
    print(model)
    # print(out.shape)
    print("Parameters of full network %.4f " % (sum([m.numel() for m in model.parameters()]) / 1e6))
