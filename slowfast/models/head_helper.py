#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""ResNe(X)t Head helper."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetRoIHead(nn.Module):
    """
    ResNe(X)t RoI head.
    """

    def __init__(
        self,
        cfg,
        dim_in,
        num_classes,
        pool_size,
        resolution,
        scale_factor,
        pool_type="avg",
        dropout_rate=0.0,
        act_func="softmax",
        aligned=True,
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetRoIHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            resolution (list): the list of spatial output size from the ROIAlign.
            scale_factor (list): the list of ratio to the input boxes by this
                number.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            aligned (bool): if False, use the legacy implementation. If True,
                align the results more perfectly.
        Note:
            Given a continuous coordinate c, its two neighboring pixel indices
            (in our pixel model) are computed by floor (c - 0.5) and ceil
            (c - 0.5). For example, c=1.3 has pixel neighbors with discrete
            indices [0] and [1] (which are sampled from the underlying signal at
            continuous coordinates 0.5 and 1.5). But the original roi_align
            (aligned=False) does not subtract the 0.5 when computing neighboring
            pixel indices and therefore it uses pixels with a slightly incorrect
            alignment (relative to our pixel model) when performing bilinear
            interpolation.
            With `aligned=True`, we first appropriately scale the ROI and then
            shift it by -0.5 prior to calling roi_align. This produces the
            correct neighbors; It makes negligible differences to the model's
            performance if ROIAlign is used together with conv layers.
        """
        super(ResNetRoIHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.cfg = cfg
        self.num_pathways = len(pool_size)
        if self.cfg.PGT.ENABLE:
            self.cache = [None] * self.num_pathways
            self.nframes = self.cfg.PGT.STEP_LEN
            self.ov = self.cfg.PGT.OVERLAP

        from .detection_helper import ROIAlign
        for pathway in range(self.num_pathways):
            if pool_type == "avg":
                temporal_pool = nn.AvgPool3d(
                    [pool_size[pathway][0], 1, 1], stride=1
                )
            elif pool_type == "max":
                temporal_pool = nn.MaxPool3d(
                    [pool_size[pathway][0], 1, 1], stride=1
                )
            self.add_module("s{}_tpool".format(pathway), temporal_pool)

            roi_align = ROIAlign(
                resolution[pathway],
                spatial_scale=1.0 / scale_factor[pathway],
                sampling_ratio=0,
            )
            self.add_module("s{}_roi".format(pathway), roi_align)
            spatial_pool = nn.MaxPool2d(resolution[pathway], stride=1)
            self.add_module("s{}_spool".format(pathway), spatial_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)

        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs, bboxes, slices=None):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        if bboxes.nelement() == 0:  # Empty bboxes
            return None
        if slices is not None:
            inputs = [x[:, :, s] for x, s in zip(inputs, slices)]
        pool_out = []
        for pathway in range(self.num_pathways):
            if self.cfg.PGT.ENABLE and self.cfg.PGT.HEAD:
                t = inputs[pathway].size(2)
                if t < self.nframes[pathway]:
                    inputs[pathway] = torch.cat([self.cache[pathway], inputs[pathway]], dim=2)
                    assert inputs[pathway].size(2) == self.nframes[pathway]
                else: # reset cache
                    self.cache[pathway] = None
                if self.cfg.PGT.CACHE == "last":
                    cache = inputs[pathway][:, :, -self.ov[pathway]:, ...]
                elif self.cfg.PGT.CACHE == "max":
                    cache = F.adaptive_max_pool3d(inputs[pathway], (self.ov[pathway], None, None))
                elif self.cfg.PGT.CACHE == "avg":
                    cache = F.adaptive_avg_pool3d(inputs[pathway], (self.ov[pathway], None, None))

            t_pool = getattr(self, "s{}_tpool".format(pathway))
            out = t_pool(inputs[pathway])
            assert out.shape[2] == 1
            out = torch.squeeze(out, 2)

            # update cache
            if self.cfg.PGT.ENABLE and self.cfg.PGT.HEAD:
                if isinstance(self.cache[pathway], torch.Tensor):
                    momentum = self.cfg.PGT.CACHE_MOMENTUM
                    cache = (1 - momentum) * cache + momentum * self.cache[pathway]
                self.cache[pathway] = cache if self.cfg.PGT.TRUNCATE_GRAD else cache.detach()

            roi_align = getattr(self, "s{}_roi".format(pathway))
            out = roi_align(out, bboxes)

            s_pool = getattr(self, "s{}_spool".format(pathway))
            pool_out.append(s_pool(out))

        # B C H W.
        x = torch.cat(pool_out, 1)

        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        x = x.view(x.shape[0], -1)
        x = self.projection(x)
        x = self.act(x)
        return x


class ResNetBasicHead(nn.Module):
    """
    ResNe(X)t 3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(
        self,
        cfg,
        dim_in,
        num_classes,
        pool_size,
        pool_type=["avg", "avg"],
        dropout_rate=0.0,
        act_func="softmax",
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetBasicHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(ResNetBasicHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.cfg = cfg
        self.num_pathways = len(pool_size)
        if self.cfg.PGT.ENABLE:
            self.cache = [None] * self.num_pathways
            self.nframes = self.cfg.PGT.STEP_LEN
            self.ov = self.cfg.PGT.OVERLAP

        for pathway in range(self.num_pathways):
            if pool_type[0] == "avg":
                spatial_pool = nn.AvgPool3d(
                    [1, pool_size[pathway][1], pool_size[pathway][2]], stride=1
                )
            elif pool_type[0] == "max":
                spatial_pool = nn.MaxPool3d(
                    [1, pool_size[pathway][1], pool_size[pathway][2]], stride=1
                )
            self.add_module("s{}_spool".format(pathway), spatial_pool)

            if pool_type[1] == "avg":
                temporal_pool = nn.AvgPool3d(
                    [pool_size[pathway][0], 1, 1], stride=1
                )
            elif pool_type[1] == "max":
                temporal_pool = nn.MaxPool3d(
                    [pool_size[pathway][0], 1, 1], stride=1
                )
            self.add_module("s{}_tpool".format(pathway), temporal_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def clear_cache(self):
        assert self.cfg.PGT.ENABLE, "Only used when PGT enables!"
        self.cache = [None] * self.num_pathways

    def forward(self, inputs):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            # Progress padding
            if self.cfg.PGT.ENABLE and self.cfg.PGT.HEAD:
                t = inputs[pathway].size(2)
                if t < self.nframes[pathway]:
                    inputs[pathway] = torch.cat([self.cache[pathway], inputs[pathway]], dim=2)
                    assert inputs[pathway].size(2) == self.nframes[pathway]
                else: # reset cache
                    self.cache[pathway] = None
                if self.cfg.PGT.CACHE == "last":
                    cache = inputs[pathway][:, :, -self.ov[pathway]:, ...]
                elif self.cfg.PGT.CACHE == "max":
                    cache = F.adaptive_avg_pool3d(inputs[pathway], (self.ov[pathway], None, None))
                elif self.cfg.PGT.CACHE == "avg":
                    cache = F.adaptive_avg_pool3d(inputs[pathway], (self.ov[pathway], None, None))

            s_pool = getattr(self, "s{}_spool".format(pathway))
            t_pool = getattr(self, "s{}_tpool".format(pathway))
            pool_out.append(t_pool(s_pool(inputs[pathway])))

            # update cache
            if self.cfg.PGT.ENABLE and self.cfg.PGT.HEAD:
                if isinstance(self.cache[pathway], torch.Tensor):
                    momentum = self.cfg.PGT.CACHE_MOMENTUM
                    cache = (1 - momentum) * cache + momentum * self.cache[pathway]
                self.cache[pathway] = cache if self.cfg.PGT.TRUNCATE_GRAD else cache.detach()
        x = torch.cat(pool_out, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        # Performs fully convlutional inference.
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2, 3])

        x = x.view(x.shape[0], -1)
        return x


class X3DHead(nn.Module):
    """
    X3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(
        self,
        cfg,
        dim_in,
        dim_inner,
        dim_out,
        num_classes,
        pool_size,
        pool_type=["avg", "avg"],
        dropout_rate=0.0,
        act_func="softmax",
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        norm_module=nn.BatchNorm3d,
        bn_lin5_on=False,
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        X3DHead takes a 5-dim feature tensor (BxCxTxHxW) as input.
        Args:
            dim_in (float): the channel dimension C of the input.
            num_classes (int): the channel dimensions of the output.
            pool_size (float): a single entry list of kernel size for
                spatiotemporal pooling for the TxHxW dimensions.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
            bn_lin5_on (bool): if True, perform normalization on the features
                before the classifier.
        """
        super(X3DHead, self).__init__()
        self.cfg = cfg
        self.pool_size = pool_size
        self.pool_type = pool_type
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.act_func = act_func
        self.eps = eps
        self.bn_mmt = bn_mmt
        self.inplace_relu = inplace_relu
        self.bn_lin5_on = bn_lin5_on
        # X3D is single pathway
        if self.cfg.PGT.ENABLE:
            self.cache = None
            self.nframes = self.cfg.PGT.STEP_LEN[0]
            self.ov = self.cfg.PGT.OVERLAP[0]
        self._construct_head(dim_in, dim_inner, dim_out, norm_module)

    def _construct_head(self, dim_in, dim_inner, dim_out, norm_module):

        self.conv_5 = nn.Conv3d(
            dim_in,
            dim_inner,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=False,
        )
        self.conv_5_bn = norm_module(
            num_features=dim_inner, eps=self.eps, momentum=self.bn_mmt
        )
        self.conv_5_relu = nn.ReLU(self.inplace_relu)

        if self.pool_type[0] == "avg":
            s_pool = nn.AvgPool3d(
                [1, self.pool_size[1], self.pool_size[2]], stride=1)
        elif self.pool_type[0] == "max":
            s_pool = nn.MaxPool3d(
                [1, self.pool_size[1], self.pool_size[2]], stride=1)
        self.s_pool = s_pool

        if self.pool_type[1] == "avg":
            t_pool = nn.AvgPool3d([self.pool_size[0], 1, 1], stride=1)
        elif self.pool_type[1] == "max":
            t_pool = nn.MaxPool3d([self.pool_size[0], 1, 1], stride=1)
        self.t_pool = t_pool

        self.lin_5 = nn.Conv3d(
            dim_inner,
            dim_out,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=False,
        )
        if self.bn_lin5_on:
            self.lin_5_bn = norm_module(
                num_features=dim_out, eps=self.eps, momentum=self.bn_mmt
            )
        self.lin_5_relu = nn.ReLU(self.inplace_relu)

        if self.dropout_rate > 0.0:
            self.dropout = nn.Dropout(self.dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(dim_out, self.num_classes, bias=True)

        # Softmax for evaluation and testing.
        if self.act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif self.act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(self.act_func)
            )

    def clear_cache(self):
        assert self.cfg.PGT.ENABLE, "Only used when PGT enables!"
        self.cache = None

    def forward(self, inputs):
        # In its current design the X3D head is only useable for a single
        # pathway input.
        assert len(inputs) == 1, "Input tensor does not contain 1 pathway"
        x = self.conv_5(inputs[0])
        x = self.conv_5_bn(x)
        x = self.conv_5_relu(x)

        # Progress padding
        if self.cfg.PGT.ENABLE and self.cfg.PGT.HEAD:
            t = x.size(2)
            if t < self.nframes:
                x = torch.cat([self.cache, x], dim=2)
                assert x.size(2) == self.nframes
            else: # reset cache
                self.cache = None
            if self.cfg.PGT.CACHE == "last":
                cache = x[:, :, -self.ov:, ...]
            elif self.cfg.PGT.CACHE == "max":
                cache = F.adaptive_max_pool3d(x, (self.ov, None, None))
            elif self.cfg.PGT.CACHE == "avg":
                cache = F.adaptive_avg_pool3d(x, (self.ov, None, None))

        # Temporal op
        x = self.t_pool(self.s_pool(x))

        # Update cache
        if self.cfg.PGT.ENABLE and self.cfg.PGT.HEAD:
            if isinstance(self.cache, torch.Tensor):
                momentum = self.cfg.PGT.CACHE_MOMENTUM
                cache = (1 - momentum) * cache + momentum * self.cache
            self.cache = cache if self.cfg.PGT.TRUNCATE_GRAD else cache.detach()

        x = self.lin_5(x)
        if self.bn_lin5_on:
            x = self.lin_5_bn(x)
        x = self.lin_5_relu(x)

        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        # Performs fully convlutional inference.
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2, 3])

        x = x.view(x.shape[0], -1)
        return x
