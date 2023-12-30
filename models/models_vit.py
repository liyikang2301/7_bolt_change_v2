# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs): #Vision Transformer 模型的初始化
        super(VisionTransformer, self).__init__(**kwargs) #调用父类的初始化方法，确保子类继承了父类的属性和方法。

        self.global_pool = global_pool
        if self.global_pool: #将传入的 global_pool 参数赋值给类的属性 self.global_pool。这个参数用于指示是否使用全局平均池化。
            norm_layer = kwargs['norm_layer'] #从 kwargs 字典中获取键为 'norm_layer' 的值，通常这是一个规范化层（normalization layer），比如 Batch Normalization。
            embed_dim = kwargs['embed_dim'] #从 kwargs 字典中获取键为 'embed_dim' 的值，表示嵌入的维度。
            self.fc_norm = norm_layer(embed_dim) #创建一个规范化层，并将其赋值给 self.fc_norm。这里使用了嵌入的维度来初始化规范化层。

            del self.norm  # remove the original norm

    def forward_features(self, x):
        #这段代码描述了 Vision Transformer 模型的一次前向传播过程，其中包括输入的处理、位置编码、多块的处理以及最终的输出。
        B = x.shape[0] #获取输入张量 x 的批次大小（batch size）
        x = self.patch_embed(x) #将输入图像分割成小块，然后进行嵌入（embedding）,运行后维度变为(1，196，768)

        #在每个批次中，复制一份 cls_token，并将其与嵌入后的小块拼接
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        #添加位置编码，并进行位置编码的 dropout
        x = x + self.pos_embed
        x = self.pos_drop(x)

        #遍历多块处理
        for blk in self.blocks:
            x = blk(x)

        if self.global_pool: #如果使用全局平均池化（global_pool=True），则对去除 cls token 后的嵌入进行平均池化，并经过全连接层和 LayerNorm。
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else: #否则，对最后一块的输出进行 LayerNorm，并提取 cls token 对应的输出。
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model