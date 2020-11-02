#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""ResNe(X)t 3D stem helper."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_stem_func(name):
    """
    Retrieves the transformation module by name.
    """
    trans_funcs = {
        "resnet_stem": ResNetBasicStem,
        "x3d_stem": X3DStem,
    }
    assert (
        name in trans_funcs.keys()
    ), "Stem function '{}' not supported".format(name)
    return trans_funcs[name]


class VideoModelStem(nn.Module):
    """
    Video 3D stem module. Provides stem operations of Conv, BN, ReLU, MaxPool
    on input data tensor for one or multiple pathways.
    """

    def __init__(
        self,
        cfg,
        dim_in,
        dim_out,
        kernel,
        stride,
        padding,
        stem_func_name,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        norm_module=nn.BatchNorm3d,
        pool_pad=True,
    ):
        """
        The `__init__` method of any subclass should also contain these
        arguments. List size of 1 for single pathway models (C2D, I3D, Slow
        and etc), list size of 2 for two pathway models (SlowFast).

        Args:
            dim_in (list): the list of channel dimensions of the inputs.
            dim_out (list): the output dimension of the convolution in the stem
                layer.
            kernel (list): the kernels' size of the convolutions in the stem
                layers. Temporal kernel size, height kernel size, width kernel
                size in order.
            stride (list): the stride sizes of the convolutions in the stem
                layer. Temporal kernel stride, height kernel size, width kernel
                size in order.
            padding (list): the paddings' sizes of the convolutions in the stem
                layer. Temporal padding size, height padding size, width padding
                size in order.
            stem_func_name (str):
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(VideoModelStem, self).__init__()

        assert (
            len(
                {
                    len(dim_in),
                    len(dim_out),
                    len(kernel),
                    len(stride),
                    len(padding),
                }
            )
            == 1
        ), "Input pathway dimensions are not consistent."
        self.cfg = cfg
        self.num_pathways = len(dim_in)
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.eps = eps
        self.bn_mmt = bn_mmt
        # Construct the stem layer.
        self._construct_stem(dim_in, dim_out, stem_func_name, norm_module, pool_pad)
        if cfg.PGT.ENABLE:
            # NOTE: only fast pathway has cache
            self.cache = [None] * self.num_pathways
            self.nframes = self.cfg.PGT.STEP_LEN
            self.ov = self.cfg.PGT.OVERLAP

    def clear_cache(self):
        assert self.cfg.PGT.ENABLE, "Only used when PGT enables!"
        self.cache = [None] * self.num_pathways

    def _construct_stem(self, dim_in, dim_out, stem_func_name, norm_module, pool_pad):
        stem_func = get_stem_func(stem_func_name)
        for pathway in range(len(dim_in)):
            stem = stem_func(
                dim_in[pathway],
                dim_out[pathway],
                self.kernel[pathway],
                self.stride[pathway],
                self.padding[pathway],
                self.inplace_relu,
                self.eps,
                self.bn_mmt,
                norm_module,
                is_pool=True,
                pool_pad=pool_pad,
            )
            self.add_module("pathway{}_stem".format(pathway), stem)

    def forward(self, x):
        assert (
            len(x) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        for pathway in range(len(x)):
            # Progress padding when temp kernel
            if self.cfg.PGT.ENABLE and self.kernel[pathway][0] > 1:
                t = x[pathway].size(2)
                if t < self.nframes[pathway]:
                    x[pathway] = torch.cat([self.cache[pathway], x[pathway]], dim=2)
                    assert x[pathway].size(2) == self.nframes[pathway]
                else: # reset cache
                    self.cache[pathway] = None
                if self.cfg.PGT.CACHE == "last":
                    cache = x[pathway][:, :, -self.ov[pathway]:, ...]
                elif self.cfg.PGT.CACHE == "max":
                    cache = F.adaptive_max_pool3d(x[pathway], (self.ov[pathway], None, None))
                elif self.cfg.PGT.CACHE == "avg":
                    cache = F.adaptive_avg_pool3d(x[pathway], (self.ov[pathway], None, None))
            m = getattr(self, "pathway{}_stem".format(pathway))
            x[pathway] = m(x[pathway])

            # Remove progress padding and update cache
            if self.cfg.PGT.ENABLE and self.kernel[pathway][0] > 1:
                if isinstance(self.cache[pathway], torch.Tensor):
                    x[pathway] = x[pathway][:, :, self.ov[pathway]:, ...]
                    momentum = self.cfg.PGT.CACHE_MOMENTUM
                    cache = (1 - momentum) * cache + momentum * self.cache[pathway]
                self.cache[pathway] = cache if self.cfg.PGT.TRUNCATE_GRAD else cache.detach()
        return x


class ResNetBasicStem(nn.Module):
    """
    ResNe(X)t 3D stem module.
    Performs spatiotemporal Convolution, BN, and Relu following by a
        spatiotemporal pooling.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel,
        stride,
        padding,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        norm_module=nn.BatchNorm3d,
        is_pool=True,
        pool_pad=True,
    ):
        """
        The `__init__` method of any subclass should also contain these arguments.

        Args:
            dim_in (int): the channel dimension of the input. Normally 3 is used
                for rgb input, and 2 or 3 is used for optical flow input.
            dim_out (int): the output dimension of the convolution in the stem
                layer.
            kernel (list): the kernel size of the convolution in the stem layer.
                temporal kernel size, height kernel size, width kernel size in
                order.
            stride (list): the stride size of the convolution in the stem layer.
                temporal kernel stride, height kernel size, width kernel size in
                order.
            padding (int): the padding size of the convolution in the stem
                layer, temporal padding size, height padding size, width
                padding size in order.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
            is_pool (bool): pooling or not.
            pool_pad (bool): padding in pooling or not.
        """
        super(ResNetBasicStem, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.eps = eps
        self.bn_mmt = bn_mmt
        self.is_pool = is_pool
        self.pool_pad = pool_pad

        # Construct the stem layer.
        self._construct_stem(dim_in, dim_out, norm_module)

    def _construct_stem(self, dim_in, dim_out, norm_module):
        self.conv = nn.Conv3d(
            dim_in,
            dim_out,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            bias=False,
        )
        self.bn = norm_module(
            num_features=dim_out, eps=self.eps, momentum=self.bn_mmt
        )
        self.relu = nn.ReLU(self.inplace_relu)
        if self.is_pool:
            padding = [0, 1, 1] if self.pool_pad else 0
            self.pool_layer = nn.MaxPool3d(
                kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=padding
            )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.is_pool:
            x = self.pool_layer(x)
        return x


class X3DStem(nn.Module):
    """
    X3D's 3D stem module.
    Performs a spatial followed by a depthwise temporal Convolution, BN, and Relu following by a
        spatiotemporal pooling.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel,
        stride,
        padding,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        norm_module=nn.BatchNorm3d,
        is_pool=None,
        pool_pad=None,
    ):
        """
        The `__init__` method of any subclass should also contain these arguments.
        Args:
            dim_in (int): the channel dimension of the input. Normally 3 is used
                for rgb input, and 2 or 3 is used for optical flow input.
            dim_out (int): the output dimension of the convolution in the stem
                layer.
            kernel (list): the kernel size of the convolution in the stem layer.
                temporal kernel size, height kernel size, width kernel size in
                order.
            stride (list): the stride size of the convolution in the stem layer.
                temporal kernel stride, height kernel size, width kernel size in
                order.
            padding (int): the padding size of the convolution in the stem
                layer, temporal padding size, height padding size, width
                padding size in order.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(X3DStem, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.eps = eps
        self.bn_mmt = bn_mmt
        # Construct the stem layer.
        self._construct_stem(dim_in, dim_out, norm_module)

    def _construct_stem(self, dim_in, dim_out, norm_module):
        self.conv_xy = nn.Conv3d(
            dim_in,
            dim_out,
            kernel_size=(1, self.kernel[1], self.kernel[2]),
            stride=(1, self.stride[1], self.stride[2]),
            padding=(0, self.padding[1], self.padding[2]),
            bias=False,
        )
        self.conv = nn.Conv3d(
            dim_out,
            dim_out,
            kernel_size=(self.kernel[0], 1, 1),
            stride=(self.stride[0], 1, 1),
            padding=(self.padding[0], 0, 0),
            bias=False,
            groups=dim_out,
        )

        self.bn = norm_module(
            num_features=dim_out, eps=self.eps, momentum=self.bn_mmt
        )
        self.relu = nn.ReLU(self.inplace_relu)

    def forward(self, x):
        x = self.conv_xy(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
