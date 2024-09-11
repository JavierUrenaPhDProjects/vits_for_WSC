# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_


class VisionTransformer_token(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))

        trunc_normal_(self.pos_embed, std=.02)

        self.output1 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 1)
        )
        self.output1.apply(self._init_weights)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        x = self.output1(x)
        return x


class VisionTransformer_gap(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))

        trunc_normal_(self.pos_embed, std=.02)

        self.output1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(6912 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
        self.output1.apply(self._init_weights)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x[:, 1:]
        return x

    def forward(self, x):
        with torch.no_grad():
            x = self.forward_features(x)
            # x = self.head(x)
            x = F.adaptive_avg_pool1d(x, (48))
            x = x.view(x.shape[0], -1)
        x = self.output1(x)
        return x


@register_model
def transcrowd_token(args, **kwargs):
    """
    download from https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth
    """
    model = VisionTransformer_token(
        img_size=args['img_size'],
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        in_chans=args['channels'],
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def transcrowd_gap(args, **kwargs):
    """
    download from https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth
    """
    model = VisionTransformer_gap(
        img_size=args['img_size'],
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        in_chans=args['channels'],
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def TC_384_16_token(args, **kwargs):
    """
    download from https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth
    """
    model = VisionTransformer_token(
        img_size=args['img_size'],
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        in_chans=args['channels'],
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def TC_384_16_gap(args, **kwargs):
    """
    download from https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth
    """
    model = VisionTransformer_gap(
        img_size=args['img_size'],
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        in_chans=args['channels'],
        **kwargs)
    model.default_cfg = _cfg()
    return model


if __name__ == '__main__':
    args = {'img_size': 384, 'channels': 3}
    model_gap = transcrowd_gap(args)
    model_token = transcrowd_token(args)
    x = torch.randn(1, 3, 384, 384)
    y = model_gap(x)
    print(y)
    print(sum(p.numel() for p in model_gap.parameters()))
    y = model_token(x)
    print(y)
    print(sum(p.numel() for p in model_token.parameters()))
