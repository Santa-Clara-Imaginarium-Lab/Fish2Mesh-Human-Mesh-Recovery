from layers.swin_blocks import *

import torch
import numpy as np


def calculate_attention_mask(H, W, window_size, shift_size, device):
    # Extend the dimensions to be multiples of window_size
    Hp = int(np.ceil(H / window_size)) * window_size
    Wp = int(np.ceil(W / window_size)) * window_size

    # Initialize the image mask
    img_mask = torch.zeros((1, Hp, Wp, 1), device=device)

    # Define slices for the windows
    h_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    w_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))

    # Create the mask
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    # Partition the mask into windows and create the attention mask
    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attention_mask.masked_fill(attention_mask != 0, float(-100.0)).masked_fill(attention_mask == 0,
                                                                                           float(0.0))

    return attention_mask


class EncoderBlock(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 inverse=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            swin_encoder_block(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                inverse=inverse)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        attention_mask = calculate_attention_mask(H, W, self.window_size, self.shift_size, x.device)

        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x, attention_mask)

        if self.downsample is not None:
            x_down = self.downsample(x, H, W)

            return x_down, H, W
        else:
            return x, H, W


class EncoderBlock_withoutPos(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 inverse=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            swin_encoder_block_withoutPos(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                inverse=inverse)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        attention_mask = calculate_attention_mask(H, W, self.window_size, self.shift_size, x.device)

        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x, attention_mask)

        if self.downsample is not None:
            x_down = self.downsample(x, H, W)

            return x_down, H, W
        else:
            return x, H, W