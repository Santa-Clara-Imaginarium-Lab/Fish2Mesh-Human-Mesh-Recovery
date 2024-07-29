from torch.nn.init import trunc_normal_
from encoder_block import *
from decoder_block import *
from util import PatchEmbed, PatchMerging
from regressor_head import *


class D3Pose(nn.Module):
    def __init__(self,
                 pretrain_img_size=256,
                 patch_size=2,
                 in_chans=3,
                 embed_dim=48,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 offset=0.4,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.in_chans = in_chans

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.pos_drop2 = nn.Dropout(p=drop_rate+offset)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # initialize encoder layers
        self.encoder_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = EncoderBlock(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                inverse=False)
            self.encoder_layers.append(layer)

        # dpr2 = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # initialize decoder layers
        # self.decoder_layers = nn.ModuleList()
        # for i_layer in range(self.num_layers):
        #     layer = DecoderBlock(
        #         dim=82,
        #         mlp_ratio=mlp_ratio,
        #         num_heads=num_heads[i_layer],
        #         qkv_bias=qkv_bias,
        #         qk_scale=qk_scale,
        #         drop=drop_rate+offset,
        #         attn_drop=attn_drop_rate+offset,
        #         drop_path=drop_path_rate+offset,
        #         norm_layer=norm_layer,
        #         downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
        #         use_checkpoint=use_checkpoint,
        #         inverse=False)
        #     self.decoder_layers.append(layer)

        # initialize regression head
        # self.regressor_heads = nn.ModuleList()
        # for i_layer in range(self.num_layers):
        #     layer = regressor_head(
        #         i=i_layer,
        #         embed_dim=embed_dim,
        #         in_chans=31,
        #         out_features=82,
        #     )
        #     self.regressor_heads.append(layer)

        self.outputHead = nn.Sequential(
            conv3x3(embed_dim * 8, embed_dim * 2),
            nn.GELU(),
            conv3x3(embed_dim * 2, embed_dim // 2),
            nn.GELU(),
            # conv3x3(embed_dim // 2, embed_dim // 8),
            # nn.GELU(),
            conv3x3(embed_dim // 2 , 2),
        )
        # global_orient (b x 1 x 3 x 3), body_pose (b x 23 x 3 x 3), betas (b x 10), pred_cam (b x 3)
        # 9 + 207 + 10 + 3 = 229
        self.regressHead = nn.Linear(2 * 14 * 14, 229)
        # self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, images):
        """Forward function."""
        # images(feature) : batch x 243 x 200 x 192
        # images(frames)  : batch x 80 x 3 x 224 x 224

        # preprocess encoder input
        # images = images.view(-1, 1, 200, 192)
        images = images.view(-1, 3, 224, 224)
        encoder_input = self.patch_embed(images)
        WH, WW = encoder_input.size(2), encoder_input.size(3)

        encoder_input = encoder_input.flatten(2).transpose(1, 2)
        encoder_input = self.pos_drop(encoder_input)

        # decoder_out = gt

        # encoder_head = encoder_input

        for i in range(self.num_layers):
            encoder_block = self.encoder_layers[i]

            encoder_out, WH, WW = encoder_block(encoder_input, WH, WW)
            encoder_input = encoder_out

            C = self.embed_dim * 2 ** (i)
            if i < 3:
                WH, WW = (WH + 1) // 2, (WW + 1) // 2
                C = self.embed_dim * 2 ** (i + 1)

            # C = self.embed_dim * 2 ** (i + 1)
            # encoder_out = encoder_out.view(-1, int(WH), int(WW), C).permute(0, 3, 1, 2).contiguous()

            # regressor_head = self.regressor_heads[i]
            # encoder_head = regressor_head(encoder_out, WH, WW)

            # encoder_out as part of decoder's input
            # decoder_block = self.decoder_layers[i]
            # decoder_out = decoder_block(encoder_head, decoder_out)

        encoder_out = encoder_out.view(-1, int(WH), int(WW), C).permute(0, 3, 1, 2).contiguous()
        encoder_head = self.outputHead(encoder_out)
        encoder_head = encoder_head.view(-1, 1, 2*14*14)
        encoder_head = self.regressHead(encoder_head)
        # global_orient (b x 1 x 3 x 3), body_pose (b x 23 x 3 x 3), betas (b x 10), pred_cam (b x 3)
        # 9 + 207 + 10 + 3 = 229

        # encoder_head = self.sigmoid(encoder_head)

        return encoder_head #.view(-1, 243, 200, 192)
