from swin_blocks import *


def calculate_temporal_mask(N, device='cuda'):
    mask = torch.triu(torch.ones(N, N) * float('-inf'), diagonal=1)
    # mask = torch.triu(torch.ones(N, N), diagonal=0)
    # inverted_mask = 1 - mask
    mask = mask.to(device)

    return mask


class DecoderBlock(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio,
                 num_heads,
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
        self.use_checkpoint = use_checkpoint

        self.cross_att_module = cross_attention(
            dim=dim,
            num_heads=2,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        self.norm5 = norm_layer(dim)
        self.norm6 = norm_layer(dim)
        self.norm7 = norm_layer(dim)
        self.act_layer = nn.GELU

        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp_ratio = mlp_ratio
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=self.act_layer, drop=drop)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=self.act_layer, drop=drop)
        self.mlp3 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=self.act_layer, drop=drop)

    def forward(self, encoder, decoder):
        temporal_attention_mask = calculate_temporal_mask(31)

        # first masked self-attention module
        shortcut1 = decoder
        decoder = self.norm1(decoder)

        decoder = self.cross_att_module(decoder, decoder, temporal_attention_mask)

        # FFN
        decoder = self.drop_path(shortcut1) + self.drop_path(decoder)
        decoder = decoder + self.drop_path(self.mlp1(self.norm2(decoder)))

        # # second masked self-attention module
        # shortcut2 = x
        # x = self.norm3(x)
        # x = self.cross_att_module(x, x, temporal_attention_mask)
        #
        # #FFN
        # x = shortcut2 + x
        # x = x + self.mlp2(self.norm4(x))

        # the cross-attention module

        shortcut3 = encoder
        encoder = self.norm5(encoder)
        decoder = self.norm7(decoder)

        # encoder kv , decoder q
        # the first param is for kv
        # the second param is for q
        cross_attention_output = self.cross_att_module(encoder, decoder)

        # FFN
        y = shortcut3 + cross_attention_output
        y = y + self.mlp3(self.norm6(y))

        return y
