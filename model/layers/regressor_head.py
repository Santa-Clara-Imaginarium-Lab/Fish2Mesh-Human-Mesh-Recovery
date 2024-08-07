import torch
import math
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import torch.nn as nn


def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)

def CNNs(i, embed_dim, in_chans):
    # Define the CNN layers

    # output from the second encoder block: 384*13*12
    cnns3 = nn.Sequential(
        conv3x3(embed_dim * 8, embed_dim * 4),
        nn.GELU(),
        conv3x3(embed_dim * 4, embed_dim * 2),
        nn.GELU(),
        conv3x3(embed_dim * 2, embed_dim),
        nn.GELU(),
        conv3x3(embed_dim, in_chans),
    )

    # output from the second encoder block: 192*25*24
    cnns2 = nn.Sequential(
        conv3x3(embed_dim * 4, embed_dim * 2),
        nn.GELU(),
        conv3x3(embed_dim * 2, embed_dim),
        nn.GELU(),
        nn.Conv2d(embed_dim, in_chans, kernel_size=3, stride=2)
    )

    # output from the first encoder block: 96*50*48
    cnns1 = nn.Sequential(
        nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=3, stride=2),
        nn.GELU(),
        nn.Conv2d(embed_dim, in_chans, kernel_size=3, stride=2)
    )

    if i == 0:
        return cnns1
    elif i == 1:
        return cnns2
    elif i == 2:
        return cnns3
    elif i == 3:
        return cnns3


class regressor_head(nn.Module):
    def __init__(self,
                 i, embed_dim,
                 in_chans,
                 out_features
                 ):
        super().__init__()
        self.i = i
        self.cnns = CNNs(i, embed_dim, in_chans)
        self.in_features = [11 * 11, 12 * 11, 13 * 12, 13 * 12]
        self.body_regressor_head = nn.Linear(self.in_features[i], out_features)
        self.embed_dim = embed_dim
        self.in_chans = in_chans

    def forward(self, x, H, W):
        features = x

        # adaptive channel size
        # C = self.embed_dim * 2 ** (self.i + 1)
        # features = features.view(-1, H, W, C).permute(0, 3, 1, 2).contiguous()

        features = self.cnns(features)
        features = features.view(-1, self.in_chans, self.in_features[self.i])
        out = self.body_regressor_head(features)

        return out
