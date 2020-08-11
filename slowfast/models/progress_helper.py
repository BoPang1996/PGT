import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel.distributed as dist

import slowfast.utils.misc as misc
import slowfast.models.optimizer as optim
from slowfast.models.head_helper import ResNetBasicHead, ResNetRoIHead


class PGT(object):
    def __init__(self, model, cfg, optimizer=None, loss_fun=None):
        self.padding = []
        self.padding_selfatt = []
        self.zero_padding = []
        self.inter_results = []
        self.inter_results_selfatt = []
        self.model = model
        self.cfg = cfg
        self.optimizer = optimizer
        self.loss_fun = loss_fun
        self.progress_steps = 1

        self.hook_handles = []
        self.register_hook()

        # pooling
        pg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        if cfg.MODEL.FINAL_POOL[1] == "avg":
            t_pool = nn.AdaptiveAvgPool3d((1, None, None))
        elif cfg.MODEL.FINAL_POOL[1] == "max":
            t_pool = nn.AdaptiveMaxPool3d((1, None, None))
        ms = self.model.module if cfg.NUM_GPUS > 1 else self.model
        if isinstance(ms.head, (ResNetRoIHead, ResNetBasicHead)):
            ms.head.s0_tpool = t_pool
        else:  # regnet
            ms.avgpool = pg_pool

        # For PrgSelfATT padding
        if self.cfg.PGT.SELFATT:
            theta_len = self.cfg.PGT.STEP_LEN if self.model.training else None
            for module in self.model.modules():
                if isinstance(module, PrgSelfAtt):
                    module.theta_len = theta_len

    def register_hook(self):
        def temp_cnn_forward_hook(module, input, output):
            assert isinstance(module, torch.nn.Conv3d) and module.kernel_size[0] == 3, \
                "wrong hook on non-temporal conv layers"
            if isinstance(input, tuple):
                input = input[0]
            self.inter_results.append(input[:, :, -3].unsqueeze(2).detach())

        def temp_cnn_forward_pre_hook(module, input):
            assert isinstance(module, torch.nn.Conv3d) and module.kernel_size[0] == 3, \
                "wrong hook on non-temporal conv layers"
            if isinstance(input, tuple):
                input = input[0]
            if len(self.padding) > 0:
                inter_s = self.padding[0].pop(0)
                inter_e = self.padding[1].pop(0)
                input = torch.cat((inter_s, input, inter_e), dim=2)
            return input

        def self_att_forward_hook(module, input, output):
            assert isinstance(
                module, PrgSelfAtt), "wrong hook on non self-attention layers"
            if isinstance(input, tuple):
                input = input[0]
            self.inter_results_selfatt.append(
                input[:, :, -self.cfg.PGT.STEP_LEN:-1].detach())

        def self_att_forward_pre_hook(module, input):
            assert isinstance(
                module, PrgSelfAtt), "wrong hook on non self-attention layers"
            if isinstance(input, tuple):
                input = input[0]
            if len(self.padding_selfatt) > 0:
                inter = self.padding_selfatt.pop(0)
                input = torch.cat((inter, input), dim=2)
            return input

        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv3d) and module.kernel_size[0] == 3:
                module.padding = (0, 0, 0)
                self.hook_handles.append(
                    module.register_forward_hook(temp_cnn_forward_hook))
                self.hook_handles.append(
                    module.register_forward_pre_hook(temp_cnn_forward_pre_hook))
            elif isinstance(module, PrgSelfAtt):
                self.hook_handles.append(
                    module.register_forward_hook(self_att_forward_hook))
                self.hook_handles.append(
                    module.register_forward_pre_hook(self_att_forward_pre_hook))

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.hook_handles:
            handle.remove()

    def padding_fn(self, inputs):
        padding = []
        b, c, t, h, w = inputs.shape
        if not isinstance(self.model, dist.DistributedDataParallel):
            padding_shape = self.model.padding_shape
        else:
            padding_shape = self.model.module.padding_shape
        for shape in padding_shape:
            padding.append(torch.zeros((b,
                                        shape[0],
                                        1,
                                        h // shape[1],
                                        w // shape[1])).cuda())
        return padding

    def step_train(self, inputs, labels, bboxes=None):
        loss_mean = []
        preds = []

        # get first padding
        self.inter_results = self.padding_fn(inputs[0])
        self.inter_results_selfatt = []    # first clip theta comes from itself
        self.zero_padding = self.padding_fn(inputs[0])

        # progress steps
        for i in range(0, inputs[0].shape[2] - 1, self.cfg.PGT.STEP_LEN - 1):
            self.padding = [
                self.inter_results.copy(), self.zero_padding.copy()]
            self.padding_selfatt = self.inter_results_selfatt
            self.inter_results = []
            self.inter_results_selfatt = []
            idx = i // (self.cfg.PGT.STEP_LEN - 1)
            progress_input = [inputs[0][:, :, i:i + self.cfg.PGT.STEP_LEN]]

            # Forward
            if bboxes != None:
                pred = self.model(progress_input, bboxes)
            else:
                pred = self.model(progress_input)

            if len(labels.size()) > 2:  # for charades
                loss = self.loss_fun(pred, labels[:,idx])
            else:  # for kinetics and ava
                # FIXME: ava step labels
                loss = self.loss_fun(pred, labels)

            # Check Nan Loss.
            misc.check_nan_losses(loss)

            # Perform the backward pass.
            self.optimizer.zero_grad()
            loss.backward()
            # Update the parameters.
            self.optimizer.step()

            preds.append(pred)
            loss_mean.append(loss.detach())

        preds = pred
        loss_mean = torch.stack(loss_mean, dim=0).mean()
        return preds, loss_mean

    def step_eval(self, inputs, bboxes=None):
        if not self.cfg.PGT.PG_EVAL:
            # remove inter results
            self.inter_results = []
            self.inter_results_selfatt = []

            # get frist padding
            self.padding = [self.padding_fn(inputs[0]),
                            self.padding_fn(inputs[0])]

            if bboxes != None:
                preds = self.model(inputs, bboxes)
            else:
                preds = self.model(inputs)

        else:
            preds = []

            # get first padding
            self.inter_results = self.padding_fn(inputs[0])
            self.inter_results_selfatt = []
            self.zero_padding = self.padding_fn(inputs[0])

            # progress steps
            for i in range(0, inputs[0].shape[2] - 1, self.cfg.PGT.STEP_LEN - 1):
                self.padding = [self.inter_results.copy(), self.zero_padding.copy()]
                self.padding_selfatt = self.inter_results_selfatt
                self.inter_results = []
                self.inter_results_selfatt = []
                progress_input = [inputs[0][:, :, i:i + self.cfg.PGT.STEP_LEN]]

                # forward
                if bboxes != None:
                    pred = self.model(progress_input, bboxes)
                else:
                    pred = self.model(progress_input)
                preds.append(pred)

            if self.cfg.PGT.ENSEMBLE_METHOD == "avg":
                preds = torch.stack(preds, dim=1).mean(dim=1)
            elif self.cfg.PGT.ENSEMBLE_METHOD == "max":
                preds = torch.stack(preds, dim=1).max(dim=1)[0]

        return preds


class PrgSelfAtt(nn.Module):
    def __init__(
        self,
        dim,
        dim_inner,
        pool_size=None,
        theta_len=None,
        instantiation="softmax",
        norm_type="batchnorm",
        zero_init_final_conv=False,
        zero_init_final_norm=True,
        norm_eps=1e-5,
        norm_momentum=0.1,
    ):
        super(PrgSelfAtt, self).__init__()
        self.dim = dim
        self.dim_inner = dim_inner
        self.pool_size = pool_size
        self.theta_len = theta_len
        self.instantiation = instantiation
        self.norm_type = norm_type
        self.hidden_state = None
        self.use_pool = (
            False
            if pool_size is None
            else True
        )
        self.norm_eps = norm_eps
        self.norm_momentum = norm_momentum
        self._construct_nonlocal(zero_init_final_conv, zero_init_final_norm)

    def _construct_nonlocal(self, zero_init_final_conv, zero_init_final_norm):
        # Three convolution heads: theta, phi, and g.
        self.conv_theta = nn.Conv3d(
            self.dim, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        self.conv_phi = nn.Conv3d(
            self.dim, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        self.conv_g = nn.Conv3d(
            self.dim, self.dim_inner, kernel_size=1, stride=1, padding=0
        )

        self.conv_f = nn.Conv3d(
            self.dim_inner, self.dim_inner, kernel_size=1, stride=1, padding=0
        )

        self.conv_z = nn.Conv3d(
            self.dim_inner, self.dim_inner, kernel_size=1, stride=1, padding=0
        )

        # Final convolution output.
        self.conv_out = nn.Conv3d(
            self.dim_inner, self.dim, kernel_size=1, stride=1, padding=0
        )
        # Zero initializing the final convolution output.
        self.conv_out.zero_init = zero_init_final_conv

        if self.norm_type == "batchnorm":
            self.bn = nn.BatchNorm3d(
                self.dim, eps=self.norm_eps, momentum=self.norm_momentum
            )
            # Zero initializing the final bn.
            self.bn.transform_final_bn = zero_init_final_norm
        elif self.norm_type == "layernorm":
            # In Caffe2 the LayerNorm op does not contain the scale an bias
            # terms described in the paper:
            # https://caffe2.ai/docs/operators-catalogue.html#layernorm
            # Builds LayerNorm as GroupNorm with one single group.
            # Setting Affine to false to align with Caffe2.
            self.ln = nn.GroupNorm(
                1, self.dim, eps=self.norm_eps, affine=False)
        elif self.norm_type == "none":
            # Does not use any norm.
            pass
        else:
            raise NotImplementedError(
                "Norm type {} is not supported".format(self.norm_type)
            )

        # Optional to add the spatial-temporal pooling.
        if self.use_pool:
            self.pool = nn.MaxPool3d(
                kernel_size=self.pool_size,
                stride=self.pool_size,
                padding=[0, 0, 0],
            )

    def forward(self, x):
        if self.theta_len is not None:
            # current clip
            x_identity = x[:, :, -self.theta_len:]
            x_for_theta = x[:, :, -self.theta_len:]
            x_for_kv = x[:, :, :self.theta_len]
        else:
            x_identity = x
            x_for_theta = x
            x_for_kv = x
        N, C, T, H, W = x_for_theta.size()

        theta = self.conv_theta(x_for_theta)

        # Perform temporal-spatial pooling to reduce the computation.
        if self.use_pool:
            x_for_kv = self.pool(x_for_kv)

        phi = self.conv_phi(x_for_kv)
        g = self.conv_g(x_for_kv)

        theta = theta.view(N, self.dim_inner, -1)
        phi = phi.view(N, self.dim_inner, -1)
        g = g.view(N, self.dim_inner, -1)

        # (N, C, TxHxW) * (N, C, TxHxW) => (N, TxHxW, TxHxW).
        theta_phi = torch.einsum("nct,ncp->ntp", (theta, phi))

        if self.instantiation == "softmax":
            # Normalizing the affinity tensor theta_phi before softmax.
            theta_phi = theta_phi * (self.dim_inner ** -0.5)
            theta_phi = nn.functional.softmax(theta_phi, dim=2)
        elif self.instantiation == "dot_product":
            spatial_temporal_dim = theta_phi.shape[2]
            theta_phi = theta_phi / spatial_temporal_dim
        else:
            raise NotImplementedError(
                "Unknown norm type {}".format(self.instantiation)
            )

        # (N, TxHxW, TxHxW) * (N, C, TxHxW) => (N, C, TxHxW).
        theta_phi_g = torch.einsum("ntg,ncg->nct", (theta_phi, g))

        # (N, C, TxHxW) => (N, C, T, H, W).
        theta_phi_g = theta_phi_g.view(N, self.dim_inner, T, H, W)

        # output
        p = self.conv_out(theta_phi_g)
        if self.norm_type == "batchnorm":
            p = self.bn(p)
        elif self.norm_type == "layernorm":
            p = self.ln(p)

        return x_identity + p

    def init_hidden(self, batch, size):
        frame, height, width = size
        self.hidden_state = torch.zeros((batch, self.dim_inner, frame, height, width),
                                        device=self.conv_theta.weight.device)
