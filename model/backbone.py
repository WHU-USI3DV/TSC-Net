import torch
import torch.nn as nn
import os
from utils.model_utils import PrepForMidas, denormalize
from plugins.zoedepth.base_models.dpt_dinov2.dinov2 import DINOv2

class MiDasCore(nn.Module):
    def __init__(self, encoder='vitl', pretrained_source=None, trainable=False, keep_aspect_ratio=True, img_size=384, denorm=False, freeze_bn=False, **kwargs):
        super(MiDasCore, self).__init__()

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23],
            'vitg': [9, 19, 29, 39]
        }

        self.encoder = encoder
        pretrained = DINOv2(model_name=encoder)

        if pretrained_source is not None:
            _, file_extension = os.path.splitext(pretrained_source)
            if file_extension == '.pth':
                state_dict = {}
                ckpt = torch.load(pretrained_source, map_location='cpu')
                #  add metric3d
                if 'dino' in pretrained_source:
                    state_dict = ckpt
                else:
                    for k in ckpt.keys():
                        if 'pretrained' in k:
                            state_dict[k.replace('pretrained.','')] = ckpt[k]
            else:
                # for zoe
                ckpt = torch.load(pretrained_source, map_location='cpu')['model']
                state_dict = {}
                for k in ckpt.keys():
                    if 'core.core.pretrained' in k:
                        state_dict[k.replace('core.core.pretrained.','')] = ckpt[k]
            pretrained.load_state_dict(state_dict)
            del state_dict
            del ckpt

        self.pretrained = pretrained
        self.trainable = trainable
        self.denorm = denorm

        self.set_trainable(trainable)

        self.prep = PrepForMidas(keep_aspect_ratio=keep_aspect_ratio,
                                 img_size=img_size, do_resize=kwargs.get('do_resize', True))

        if freeze_bn:
            self.freeze_bn()

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self


    def set_trainable(self, trainable):
        self.trainable = trainable
        if trainable:
            self.unfreeze()
        else:
            self.freeze()
        return self

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.trainable = False
        return self

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True
        self.trainable = True
        return self

    def forward(self, x):

        with torch.no_grad():
            if self.denorm:
                x = denormalize(x)
                x = self.prep(x)
            h, w = x.shape[-2:]
            self.patch_h, self.patch_w = h // 14, w // 14

        with torch.set_grad_enabled(self.trainable):

            features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)

        return features

    def get_rel_pos_params(self):
        for name, p in self.pretrained.named_parameters():
            if "pos_embed" in name:
                yield p

    def get_enc_params_except_rel_pos(self):
        for name, p in self.pretrained.named_parameters():
            if "pos_embed" not in name:
                yield p

