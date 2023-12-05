# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torchvision
from torchvision.models._utils import IntermediateLayerGetter

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class Backbone(nn.Module):
    def __init__(self, args):
        super(Backbone, self).__init__()

        if args.frozen_batch_norm:
            backbone = getattr(torchvision.models, args.backbone)(
                replace_stride_with_dilation=[False, False, args.dilation],
                pretrained=True, norm_layer=FrozenBatchNorm2d)
        else:
            backbone = getattr(torchvision.models, args.backbone)(
                replace_stride_with_dilation=[False, False, args.dilation],
                pretrained=True)

        self.num_frames = args.num_frame
        self.num_channels = 512 if args.backbone in ('resnet18', 'resnet34') else 2048

        self.body = IntermediateLayerGetter(backbone, return_layers={'layer4': "0"})

    def forward(self, x):
        x = self.body(x)["0"]

        return x


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, x):
        bs, t, _, h, w = x.shape
        x = x.reshape(bs * t, 3, h, w)

        features = self[0](x)
        _, c, oh, ow = features.shape

        pos = self[1](features).to(x.dtype)

        return features, pos


def build_backbone(args):
    pos_embed = build_position_encoding(args)
    backbone = Backbone(args)
    model = Joiner(backbone, pos_embed)
    model.num_channels = backbone.num_channels
    return model
