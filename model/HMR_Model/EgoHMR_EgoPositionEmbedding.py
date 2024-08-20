from torch.cuda import device
from torch.nn.init import trunc_normal_
from model.layers.encoder_block import *
from model.layers.decoder_block import *
from model.util.util import PatchEmbed, PatchMerging
from model.layers.regressor_head import *
from model.util.geometry import *
from model.util.smpl_wrapper import *


class EgoHMR_pos(nn.Module):
    def __init__(self,
                 pretrain_img_size=224,
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
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # initialize encoder layers
        self.encoder_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = EncoderBlock_withoutPos(
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


        self.outputHead = nn.Sequential(
            conv3x3(embed_dim * 8, embed_dim * 2),
            nn.GELU(),
            conv3x3(embed_dim * 2, embed_dim // 2),
            nn.GELU(),
            conv3x3(embed_dim // 2 , 2),
        )
        # global_orient (b x 1 x 3 x 3), body_pose (b x 23 x 3 x 3), betas (b x 10), pred_cam (b x 3)
        # 9 + 207 + 10 + 3 = 229
        self.regressHead = nn.Linear(2 * 14 * 14, 229)

        self.pos_embedding = nn.Parameter(torch.zeros(3, self.pretrain_img_size, self.pretrain_img_size))
        SMPL_CONFIG = {'data_dir': '/home/imaginarium/.cache/4DHumans/data/',
                       'model_path': '/home/imaginarium/.cache/4DHumans/data//smpl',
                       'gender': 'neutral',
                       'num_body_joints': 23,
                       'joint_regressor_extra': '/home/imaginarium/.cache/4DHumans/data//SMPL_to_J19.pkl',
                       'mean_params': '/home/imaginarium/.cache/4DHumans/data//smpl_mean_params.npz'}
        self.smpl_model = SMPL(**SMPL_CONFIG)



    def forward(self, images):
        """Forward function."""
        # images(feature) : batch x 243 x 200 x 192
        # images(frames)  : batch x 80 x 3 x 224 x 224
        # b x 3 x 224 x 224

        # preprocess encoder input
        # images = images.view(-1, 1, 200, 192)
        images = images.view(-1, 3, 224, 224) + self.pos_embedding
        encoder_input = self.patch_embed(images)
        WH, WW = encoder_input.size(2), encoder_input.size(3)

        encoder_input = encoder_input.flatten(2).transpose(1, 2)
        encoder_input = self.pos_drop(encoder_input)

        for i in range(self.num_layers):
            encoder_block = self.encoder_layers[i]

            encoder_out, WH, WW = encoder_block(encoder_input, WH, WW)
            encoder_input = encoder_out

            C = self.embed_dim * 2 ** (i)
            if i < 3:
                WH, WW = (WH + 1) // 2, (WW + 1) // 2
                C = self.embed_dim * 2 ** (i + 1)

        encoder_out = encoder_out.view(-1, int(WH), int(WW), C).permute(0, 3, 1, 2).contiguous()
        encoder_head = self.outputHead(encoder_out)
        encoder_head = encoder_head.view(-1, 1, 2*14*14)
        encoder_head = self.regressHead(encoder_head)
        # global_orient (b x 1 x 3 x 3), body_pose (b x 23 x 3 x 3), betas (b x 10), pred_cam (b x 3)
        # 9 + 207 + 10 + 3 = 229

        out_global_orient = encoder_head[:, :, 0:3 * 3].view(-1, 1, 3, 3)
        out_body_pose = encoder_head[:, :, 3 * 3: 3 * 3 + 23 * 3 * 3].view(-1, 23, 3, 3)
        out_betas = encoder_head[:, :, 3 * 3 + 23 * 3 * 3: 10 + 3 * 3 + 23 * 3 * 3].view(-1, 10)
        out_pred_cam = encoder_head[:, :, 10 + 3 * 3 + 23 * 3 * 3:].view(-1, 3)

        # regression SMPL->joints
        smpl_output = self.smpl_model(**{'global_orient': out_global_orient, 'body_pose': out_body_pose, 'betas': out_betas},
                                 pose2rot=False)
        pred_keypoints_3d = smpl_output.joints
        pred_vertices = smpl_output.vertices

        focal_length = 5000 * torch.ones(images.shape[0], 2,device='cuda:1', dtype=torch.float32)
        focal_length = focal_length.reshape(-1, 2)
        pred_keypoints_2d = perspective_projection(pred_keypoints_3d, translation=out_pred_cam,
                                                   focal_length=focal_length / self.pretrain_img_size)

        out ={}
        out['out_global_orient'] = out_global_orient
        out['out_body_pose'] = out_body_pose
        out['out_betas'] = out_betas
        out['out_pred_cam'] = out_pred_cam
        out['pred_keypoints_3d'] = pred_keypoints_3d
        out['pred_keypoints_2d'] = pred_keypoints_2d
        out['pred_vertices'] = pred_vertices

        return out
