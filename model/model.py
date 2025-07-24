import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math

model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

def _make_fusion_block(features, use_bn, size = None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )

def homo_warping(src_fea, T_ref2src, depth_values, inv_K):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    #height, width = src_fea.shape[2], src_fea.shape[3]
    h_src, w_src = src_fea.shape[2], src_fea.shape[3]
    h_ref, w_ref = depth_values.shape[2], depth_values.shape[3]

    with torch.no_grad():
        inv_k_33 = inv_K[:, :3, :3]
        rot = T_ref2src[:, :3, :3]
        trans = T_ref2src[:, :3, 3:4]

        y, x = torch.meshgrid([torch.arange(0, h_ref, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, w_ref, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(h_ref * w_ref), x.view(h_ref * w_ref)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        xyz = torch.matmul(inv_k_33, xyz)
        depth_xyz = xyz.unsqueeze(2).repeat(1,1,num_depth,1) * depth_values.view(batch, 1, num_depth, -1)
        depth_xyz = depth_xyz.permute(0,2,1,3)

        rot_depth_xyz = torch.matmul(rot.unsqueeze(1).repeat(1, num_depth, 1, 1), depth_xyz) + trans.unsqueeze(1)
        proj_xyz = torch.matmul(torch.inverse(inv_k_33).unsqueeze(1).repeat(1,num_depth,1,1), rot_depth_xyz)
        proj_xyz = proj_xyz.permute(0,2,1,3)
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        z = proj_xyz[:, 2:3, :, :].view(batch, num_depth, h_ref, w_ref)
        proj_x_normalized = proj_xy[:, 0, :,:] / ((w_src - 1) / 2.0) - 1
        proj_y_normalized = proj_xy[:, 1, :,:] / ((h_src - 1) / 2.0) - 1
        X_mask = ((proj_x_normalized > 1)+(proj_x_normalized < -1)).detach()
        proj_x_normalized[X_mask] = 2  # make sure that no point in warped image is a combinaison of im and gray
        Y_mask = ((proj_y_normalized > 1)+(proj_y_normalized < -1)).detach()
        proj_y_normalized[Y_mask] = 2
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy
        proj_mask = ((X_mask + Y_mask) > 0).view(batch, num_depth, h_ref, w_ref)
        proj_mask = (proj_mask + (z <= 0)) > 0

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * h_ref, w_ref, 2), mode='bilinear',
                                   padding_mode='zeros', align_corners=True)

    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, h_ref, w_ref)

    #return warped_src_fea , proj_mask
    return warped_src_fea , proj_mask, grid.view(batch, num_depth, h_ref, w_ref, 2)

def compute_depth_expectation(p, depth_values):
    depth_values = depth_values.view(*depth_values.shape, 1, 1)

    depth = torch.sum(p * depth_values, 1)
    return depth


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)

def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_channels))

class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 1)
    def forward(self, x):
        x = self.conv(x)
        return x

class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

class CrossCueFusion(nn.Module):
    def __init__(self, mono_dim=256, mvs_dim=32, out_dim=64):
        super().__init__()
        self.mono_dim = mono_dim
        self.mvs_dim = mvs_dim
        self.mid_dim = mono_dim
        self.out_dim = out_dim
        self.residual_connection =True

        self.mono_expand = nn.Sequential(
            nn.Conv2d(self.mono_dim, self.mid_dim, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_dim, self.mid_dim, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_dim, self.mid_dim, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.mid_dim),
            nn.ReLU(inplace=True)
        )

        self.multi_expand = nn.Sequential(
            nn.Conv2d(self.mvs_dim, self.mid_dim, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_dim, self.mid_dim, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_dim, self.mid_dim, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.mid_dim),
            nn.ReLU(inplace=True)
        )

        self.kq_dim = self.mid_dim //4 if self.mid_dim>128 else self.mid_dim

        self.lin_mono_k = nn.Conv2d(self.mid_dim, self.kq_dim, kernel_size=1)
        self.lin_mono_q = nn.Conv2d(self.mid_dim, self.kq_dim, kernel_size=1)
        self.lin_mono_v = nn.Conv2d(self.mid_dim, self.mid_dim, kernel_size=1)

        self.lin_multi_k = nn.Conv2d(self.mid_dim, self.kq_dim, kernel_size=1)
        self.lin_multi_q = nn.Conv2d(self.mid_dim, self.kq_dim, kernel_size=1)
        self.lin_multi_v = nn.Conv2d(self.mid_dim, self.mid_dim, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

        if self.residual_connection:
            self.mono_reg = nn.Sequential(
                nn.Conv2d(self.mono_dim, self.mid_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.mid_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.mid_dim, self.mid_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.mid_dim),
                nn.ReLU(inplace=True)
            )
            self.multi_reg = nn.Sequential(
                nn.Conv2d(self.mvs_dim, self.mid_dim, kernel_size=1, padding=0),
                nn.BatchNorm2d(self.mid_dim),
                nn.ReLU(inplace=True)
            )
            self.gamma = nn.Parameter(torch.zeros(1))

        self.projects = nn.Conv2d(self.mono_dim * 2, self.out_dim, 3, 1, 1, bias=False)

    def forward(self, ref_f, src_f):
        _, _, init_h, init_w = src_f.shape
        mono_feat = self.mono_expand(ref_f)
        multi_feat = self.multi_expand(src_f)
        b,c,h,w = multi_feat.shape

        # cross-cue attention
        mono_q = self.lin_mono_q(mono_feat).view(b,-1,h*w).permute(0,2,1)
        mono_k = self.lin_mono_k(mono_feat).view(b,-1,h*w)
        mono_score = torch.bmm(mono_q, mono_k)
        mono_atten = self.softmax(mono_score)

        multi_q = self.lin_multi_q(multi_feat).view(b,-1,h*w).permute(0,2,1)
        multi_k = self.lin_multi_k(multi_feat).view(b,-1,h*w)
        multi_score = torch.bmm(multi_q, multi_k)
        multi_atten = self.softmax(multi_score)

        mono_v = self.lin_mono_v(mono_feat).view(b,-1,h*w)
        mono_out = torch.bmm(mono_v, multi_atten.permute(0,2,1))
        mono_out = mono_out.view(b,self.mid_dim, h,w)

        multi_v = self.lin_multi_v(multi_feat).view(b,-1,h*w)
        multi_out = torch.bmm(multi_v, mono_atten.permute(0,2,1))
        multi_out = multi_out.view(b,self.mid_dim, h,w)


        # concatenate and upsample
        fused = torch.cat((multi_out,mono_out), dim=1)
        fused = torch.nn.functional.interpolate(fused, size=(init_h,init_w))

        if self.residual_connection:
            mono_residual = self.mono_reg(ref_f)
            multi_residual = self.multi_reg(src_f)
            fused_cat = torch.cat((mono_residual,multi_residual), dim=1)
            fused = fused_cat + self.gamma * fused

        return self.projects(fused)

class FeatureNet(nn.Module):
    def __init__(self, in_chans=256, out_channels=[64, 64, 128, 128], mono_only=False, **kwargs):
        super(FeatureNet, self).__init__()

        self.in_chans = in_chans
        self.feat_channels = out_channels
        self.mono_only = mono_only

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_chans,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])

    def forward(self, out_features):
        ref = []
        src = []
        vis = []
        patch_h = patch_w = int(math.sqrt(out_features[0][0].shape[-2]))
        for i, x in enumerate(out_features):
            x = x[0]
            _B = x.shape[0] // 3
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            if self.mono_only:
                vis.append(x)
            else:
                vis.append(x[:_B])

            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            if self.mono_only:
                ref.append(x)
            else:
                ref.append(x[:_B])
                src.append((x[_B:-_B], x[-_B:]))

        if self.mono_only:
            return vis, ref, patch_h, patch_w
        return vis, ref, src, patch_h, patch_w

class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups=1

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        if self.bn==True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn==True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn==True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups=1

        self.expand = expand
        out_features = features
        if self.expand==True:
            out_features = features//2

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

        self.size=size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output


class Head(nn.Module):
    def __init__(self, features, use_bn, is_rel, max_depth, return_features, **kwargs):
        super(Head, self).__init__()
        self.features = features
        self.use_bn = use_bn
        self.is_rel = is_rel
        self.max_depth = max_depth
        self.return_features = return_features

        self.refinenet1 = _make_fusion_block(features, use_bn)
        self.refinenet2 = _make_fusion_block(features, use_bn)
        self.refinenet3 = _make_fusion_block(features, use_bn)
        self.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32

        self.depth_output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.conf_output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.normal_output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)

        self.depth_output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.conf_output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.normal_output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 3, kernel_size=1, stride=1, padding=0),
        )
        # 这里可替换

    def forward(self, out, patch_h, patch_w):
        f_out = []
        layer_4_rn, layer_3_rn, layer_2_rn, layer_1_rn = out
        path_4 = self.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.refinenet1(path_2, layer_1_rn)

        ## depth
        depth_out = self.depth_output_conv1(path_1)
        depth_out = F.interpolate(depth_out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        depth_out = self.depth_output_conv2(depth_out)

        ## conf
        conf_out = self.conf_output_conv1(path_1)
        conf_out = F.interpolate(conf_out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        conf_out = self.conf_output_conv2(conf_out)

        ## normal
        normal_out = self.normal_output_conv1(path_1)
        normal_out = F.interpolate(normal_out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        normal_out = self.normal_output_conv2(normal_out)

        if self.return_features:
            if self.is_rel:
                return depth_out, conf_out, normal_out, [path_1, path_2, path_3, path_4]
            return depth_out * self.max_depth, conf_out, normal_out, [path_1, path_2, path_3, path_4]

        if self.is_rel:
            return depth_out, conf_out, normal_out
        return depth_out * self.max_depth, conf_out, normal_out

class MVS2D(nn.Module):
    def __init__(self, backbone, max_depth=80.0, min_depth=0.001, nlabels=[96, 48, 24, 12], cost_dims=[32, 32, 64, 64],  posenet=None, augment=False, is_rel=False, mono_only=False,  return_depth_feature=False, pretrained_source=None, **kwargs):
        super(MVS2D, self).__init__()
        self.iters = 3
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.nlabels = nlabels
        self.cost_dims = cost_dims
        self.is_rel = is_rel
        self.backbone = backbone
        self.encoder = backbone.encoder
        self.cost_dims = cost_dims
        self.mono_only = mono_only
        self.augment = augment
        self.return_depth_feature = return_depth_feature
        self.pretrained_source = pretrained_source

        self.model_config = {
            'vits': {'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        self.dim_feature = self.model_config[self.encoder]['features']

        if self.is_rel:
            self.max_depth = max_depth + 10.0 # 为了扩大搜索的范围

        self.feature = FeatureNet(in_chans=self.model_config[self.encoder]['out_channels'][-1], out_channels=self.model_config[self.encoder]['out_channels'], mono_only=self.mono_only)

        if self.mono_only:
            self.convs = nn.ModuleList([
                nn.Sequential(
                    convbn(self.model_config[self.encoder]['out_channels'][i], self.model_config[self.encoder]['out_channels'][i] // 4, 3, 1, 1, 1),
                    nn.ReLU(True),
                    nn.Conv2d(self.model_config[self.encoder]['out_channels'][i] // 4, self.dim_feature, 1, 1, 0, bias=False)
                ) for i in range(len(self.cost_dims))
            ])
            if self.is_rel:
                d_model = self.model_config[self.encoder]['out_channels'][-1]
                self.pooling = nn.AdaptiveAvgPool2d((1, 1))
                self.scale_regressor = nn.Sequential(
                    nn.Linear(4*d_model, d_model),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(d_model, d_model),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(d_model, 1),
                    nn.Softplus()
                )
        else:
            self.convs = nn.ModuleList([
                nn.Sequential(
                    convbn(self.model_config[self.encoder]['out_channels'][i], self.model_config[self.encoder]['out_channels'][i] // 4, 3, 1, 1, 1),
                    nn.ReLU(True),
                    nn.Conv2d(self.model_config[self.encoder]['out_channels'][i] // 4, self.cost_dims[i], 1, 1, 0, bias=False)
                ) for i in range(len(self.cost_dims))
            ])
            # 投影到convs
            self.downsamples = nn.ModuleList([
                nn.Sequential(
                    nn.Conv3d(self.cost_dims[i],self.cost_dims[i] // 4,3,1,padding=1,bias=False),
                    nn.Conv3d(self.cost_dims[i] // 4,1,3,1,padding=1,bias=False)
                ) for i in range(len(self.cost_dims))
            ])

            self.cross_cues = nn.ModuleList([
                CrossCueFusion(self.cost_dims[i], self.nlabels[i], self.dim_feature) for i in range(len(self.cost_dims))
            ]) # 64, 64, 128, 128

            self.pose_net = posenet

            self.mvs_depths = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.nlabels[i],
                            self.nlabels[i] * 2,
                            kernel_size=3,
                            padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.nlabels[i] * 2,
                            self.nlabels[i] * 4,
                            kernel_size=1),
                ) for i in range(len(self.cost_dims))
            ])

        self.output_head = Head(features=self.dim_feature, use_bn=False, is_rel=self.is_rel, max_depth=max_depth, return_features=return_depth_feature)

        if self.pretrained_source is not None:
            ckpt = torch.load(pretrained_source, map_location='cpu')['state_dict']
            state_dict = {}
            for k in ckpt.keys():
                state_dict[k[4:]] = ckpt[k]
            self.load_state_dict(state_dict, strict=False)
            del ckpt
            del state_dict

    def get_bins(self, nlabel):
        depth_bins = np.linspace(math.log(self.min_depth), math.log(self.max_depth), nlabel)
        depth_bins = np.array([math.exp(x) for x in depth_bins])
        depth_values = torch.from_numpy(depth_bins).float()
        return depth_values

    def compute_scale(self, pred, target, mask):
        a_00 = torch.sum(mask * pred * pred, (1, 2, 3))
        a_01 = torch.sum(mask * pred * target, (1, 2, 3))
        scale = torch.zeros_like(a_00)
        valid = a_00 > 0
        scale[valid] = a_01[valid] / (a_00[valid] + 1e-7)
        scale[~valid] = self.max_depth - 10.0

        return scale

    def process(self, image):
        do_augment = random.random()
        if do_augment > 0.5:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(image.device)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(image.device)
            image = image * std + mean
            # Gamma augmentation
            gamma = random.uniform(0.9, 1.1)
            image_aug = image ** gamma
            # Brightness augmentation
            brightness = random.uniform(0.9, 1.1)
            image_aug = image_aug * brightness
            # Color augmentation
            colors = torch.rand((3, 1, 1), device=image.device) * 0.2 + 0.9
            image_aug *= colors
            # Clip the values to maintain them between 0 and 1
            image_aug = torch.clamp(image_aug, 0, 1)
            image_aug = (image_aug - mean) / std
            return image_aug.unsqueeze(0)
        return image.unsqueeze(0)

    def augment_image(self, image):
        image_aug = []
        B = image.size(0)
        for i in range(B):
            image_aug.append(self.process(image[i]))
        return torch.cat(image_aug, dim=0)


    def depth_regression(self, p, depth_values):
        depth_values = depth_values.view(*depth_values.shape, 1, 1)
        depth = torch.sum(p * depth_values, 1).unsqueeze(1)
        return depth

    def upsample(self, x, size=None, mode='bilinear'):
        """Upsample input tensor by a factor of 2
        """
        if size is None:
            return F.interpolate(x, scale_factor=self.scale_factor, mode=mode)
        return F.interpolate(x, size, mode=mode)

    def get_extrinsics(self, p):
        R = p[..., :3]
        T = p[..., 3:]

        return transformation_from_parameters(R, T)

    def normalize_depth(self, d):
        B = d.shape[0]
        min_d = torch.min(d.view(B, -1), dim=1)[0].view(B, 1, 1, 1)
        max_d = torch.max(d.view(B, -1), dim=1)[0].view(B, 1, 1, 1)
        norm_d = (d - min_d + 1e-7) /  (max_d - min_d + 1e-7)
        return norm_d

    def get_inv_K(self, K, new_size):
        old_size = (2 * (K[..., 0, 2]), 2 * (K[..., 1, 2]))
        ratio = (old_size[1] / new_size[0], old_size[0] / new_size[1])
        K_ = K.clone()
        K_[..., 0, 0] = K[..., 0, 0] / ratio[0]
        K_[..., 0, 2] = K[..., 0, 2] / ratio[0]
        K_[..., 1, 1] = K[..., 1, 1] / ratio[1]
        K_[..., 1, 2] = K[..., 1, 2] / ratio[1]

        return torch.inverse(K_)

    def compute_depth_expectation(self, p, depth_values):
        depth = torch.sum(p * depth_values, 1)
        return depth

    def get_forward_data(self, sample):
        preds = {}
        ref_img = sample['image']
        B = ref_img.shape[0]
        visual_features, ref_features, patch_h, patch_w = self.feature(self.backbone(ref_img))
        out = []
        for i in range(len(ref_features)):
            idx = -(i+1)
            f = self.convs[idx](ref_features[idx])
            out.append(f)

        preds['scale'] = torch.ones([B, 1]).to(ref_img.device)

        if self.return_depth_feature:
            preds['rel_depth'], preds['conf'], preds['normal'], f = self.output_head(out, patch_h, patch_w)
            return preds, visual_features, f
        preds['rel_depth'], preds['conf'], preds['normal'] = self.output_head(out, patch_h, patch_w)

        return preds, visual_features, None


    def forward(self, sample):
        preds, _, _ = self.get_forward_data(sample)

        for k in preds.keys():
            if len(preds[k].shape) == 4:
                if sample['depth'].shape[-2:] != preds[k].shape[-2:]:
                    preds[k] = nn.functional.interpolate(preds[k], sample['depth'].shape[-2:], mode='bilinear', align_corners=True)

        if self.is_rel:
            valid_mask = ((sample['depth'] > 0.001) & (sample['depth'] < self.max_depth - 10.0))
            valid_mask = (valid_mask & ~sample['sem_mask']).detach()
            B = valid_mask.shape[0]
            s = self.compute_scale(preds['rel_depth'], sample['depth'], valid_mask)
            preds['rel_scale'] = s
        else:
            B = preds['rel_depth'].shape[0]

        preds['metric_depth'] = preds['rel_depth'] * preds['scale'].view(B, 1, 1, 1)

        return preds

def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M

def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot

def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.3)
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.leaky_relu(layer(x), 0.2, True) if i < self.num_layers - 1 else layer(x)
        return x


class MVSEstimate(nn.Module):
    def __init__(self, backbone, is_rel, depth_encoder, prompt_encoder, visual_encoder, object_decoder, num_queries, d_model, max_depth=80.0, has_mask=False, has_depth=True, l_query=True, multi_scale=False, has_cls=False, dino=None, **kwargs):
        super(MVSEstimate, self).__init__()
        self.backbone = backbone
        self.is_rel = is_rel
        self.max_depth = max_depth
        self.has_mask = has_mask
        self.d_model = d_model
        self.l_query = l_query
        self.multi_scale = multi_scale
        self.has_depth = has_depth
        self.has_cls = has_cls
        self.dino = dino

        ## 这个地方需要把几何特征进行编码，例如Depth，Normal
        self.visual_encoder = visual_encoder
        ## 这个地方把位置进行编码
        self.object_decoder = object_decoder
        ## 这个地方需要把conf编码进去
        # query：记得要编码进行位置（对称性的位置
        # 先尝试用真实的深度估计scale扔进去能达到的理论值
        ## 在尝试回归scale

        if self.has_depth:
            self.depth_encoder = depth_encoder

        self.div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / 32))
        self.pos = None

        if self.l_query:
            self.query_embed = nn.Linear(d_model, d_model)
            self.prompt_encoder = prompt_encoder
        else:
            self.query_embed = nn.Parameter(torch.randn(num_queries, d_model), requires_grad=True)
            self.prompt_encoder = MLP(4, d_model, d_model, 3)

        if self.has_cls:
            self.size_regressor = nn.Sequential(
                MLP(d_model, d_model, 3 + 61, 2),
            )
        else:
            self.size_regressor = nn.Sequential(
                MLP(d_model, d_model, 3, 2),
                nn.ReLU(inplace=True),
            )

        if self.has_mask:
            self.mask_embed = MLP(d_model, d_model // 2, d_model, 3)
            self.mask_conv = nn.Sequential(
                nn.Conv2d(d_model * 2, d_model, 1, 1, 0),
                nn.ConvTranspose2d(d_model, d_model, 4, 4, 0)
            )

        for p in self.backbone.parameters():
            p.requires_grad = False

    def compute_scale(self, pred, target, mask):
        a_00 = torch.sum(mask * pred * pred, (1, 2, 3))
        a_01 = torch.sum(mask * pred * target, (1, 2, 3))
        scale = torch.zeros_like(a_00)
        valid = a_00 > 0
        scale[valid] = a_01[valid] / (a_00[valid] + 1e-7)
        scale[~valid] = self.max_depth

        return scale

    def select_tgt(self, position, f):
        B, N, C = f.shape
        patch = int(math.sqrt(N))
        p = position[..., :2] * patch
        index = torch.clamp(p[..., 1] - 1, min=0).int() * patch + p[..., 0].int()
        index = index.unsqueeze(-1).expand(-1, -1, C).long()
        pos = torch.gather(self.pos.unsqueeze(0).expand(B, -1, -1), 1, index)
        if self.l_query:
            tgt = torch.gather(f, 1, index)
            tgt = tgt + self.query_embed(tgt)
        else:
            tgt = self.query_embed.unsqueeze(0).expand(B, -1, -1)

        return tgt, pos

    def forward(self, sample):
        with torch.no_grad():
            preds, visual_features, depth_features = self.backbone.get_forward_data(sample)

        reldepth = preds['rel_depth']
        normal = preds['normal']
        conf = preds['conf']
        B = reldepth.shape[0]
        # img_sz = sample['image'].shape[-2:]
        img_sz = (1024, 1024)
        reldepth = reldepth * preds['scale'].view(B, 1, 1, 1)
        device = conf.device

        if self.pos is None:
            self.div_term = self.div_term.to(device)
            seq_len = img_sz[0] * img_sz[1]
            position = torch.arange(0, seq_len, dtype=torch.float32, device=device).unsqueeze(1)
            pos_encoding = torch.zeros((seq_len, self.d_model), dtype=torch.float32,
                                       device=device)
            pos_encoding[:, 0::2] = torch.sin(position * self.div_term)
            pos_encoding[:, 1::2] = torch.cos(position * self.div_term)
            self.pos = pos_encoding

        # 设计树形注意力
        v_f = self.visual_encoder(visual_features)

        if self.has_depth:
            d_f = self.depth_encoder(depth_features, reldepth, normal, conf)
        else:
            d_f = v_f

        obj_position = sample['position']
        tgt, pos = self.select_tgt(obj_position, v_f)

        if self.l_query:
            query_pos = self.prompt_encoder(obj_position, pos)
        else:
            query_pos = self.prompt_encoder(obj_position)

        key_padding_mask = sample['mask_2d'].bool()

        tgt = self.object_decoder(v_f, d_f, query_pos, tgt, key_padding_mask)

        if self.has_cls:
            f_head = self.size_regressor(tgt)
            preds['obj'] = F.relu(f_head[..., :3])
            cls_logist = f_head[..., 3:]
            cls_pred = torch.argmax(cls_logist, dim=-1)
            preds['cls_logist'] = cls_logist
            preds['cls_pred'] = cls_pred
        else:
            preds['obj'] = self.size_regressor(tgt)
        preds['rel_depth'] = nn.functional.interpolate(preds['rel_depth'], img_sz, mode='bilinear', align_corners=True)
        preds['metric_depth'] = preds['rel_depth'] * preds['scale'].view(B, 1, 1, 1)

        return preds
