import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel.distributed as dist

from slowfast.config.defaults import _C
from slowfast.utils import misc as misc
from slowfast.models import optimizer as optim
from slowfast.models.head_helper import ResNetBasicHead, ResNetRoIHead


class ProgressTrainer(object):
    def __init__(self, model, cfg, optimizer=None, loss_fun=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fun = loss_fun
        self.steps = cfg.PGT.STEPS
        self.overlap = cfg.PGT.OVERLAP
        self.num_frames = cfg.PGT.STEP_LEN
        self.progress_eval = cfg.PGT.PG_EVAL
        self.ensemble_method = cfg.PGT.ENSEMBLE_METHOD

        if self.model.training or cfg.PGT.PG_EVAL:
            tpool_size = 1
        else:
            tpool_size = cfg.DATA.NUM_FRAMES // cfg.PGT.STEP_LEN

        pg_pool = nn.AdaptiveAvgPool3d((tpool_size, 1, 1))
        if cfg.MODEL.FINAL_POOL[1] == "avg":
            t_pool = nn.AdaptiveAvgPool3d((tpool_size, None, None))
        elif cfg.MODEL.FINAL_POOL[1] == "max":
            t_pool = nn.AdaptiveMaxPool3d((tpool_size, None, None))
        ms = self.model.module if cfg.NUM_GPUS > 1 else self.model
        if hasattr(ms, "head"):
            ms.head.s0_tpool = t_pool
        else:  # regnet
            ms.avgpool = pg_pool

    def step_train(self, inputs, labels, bboxes=None):
        losses = []
        preds = []

        for step in range(self.steps):
            if step == 0:
                start_idx = 0
                end_idx = self.num_frames
            else:
                start_idx = step * self.num_frames - (step - 1) * self.overlap
                end_idx = start_idx + self.num_frames - self.overlap
            pg_input = [inputs[0][:, :, start_idx:end_idx]]

            # Forward
            if bboxes != None:
                pred = self.model(pg_input, bboxes)
            else:
                pred = self.model(pg_input)

            if len(labels.size()) > 2:  # for charades
                loss = self.loss_fun(pred, labels[:, step])
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
            losses.append(loss.detach())

        preds = pred  # take the last step for train acc/map calucaltion 
        loss_mean = torch.stack(losses, dim=0).mean()
        return preds, loss_mean

    @torch.no_grad()
    def step_eval(self, inputs, bboxes=None):
        if not self.progress_eval:
            if bboxes != None:
                preds = self.model(inputs, bboxes)
            else:
                preds = self.model(inputs)

        else:
            preds = []

            for step in range(self.steps):
                if step == 0:
                    start_idx = 0
                    end_idx = self.num_frames
                else:
                    start_idx = step * self.num_frames - (step - 1) * self.overlap
                    end_idx = start_idx + self.num_frames - self.overlap
                pg_input = [inputs[0][:, :, start_idx:end_idx]]

                # Forward
                if bboxes != None:
                    pred = self.model(pg_input, bboxes)
                else:
                    pred = self.model(pg_input)

                preds.append(pred)

            if self.ensemble_method == "sum":
                preds = torch.stack(preds, dim=1).sum(dim=1)
            elif self.ensemble_method == "max":
                preds = torch.stack(preds, dim=1).max(dim=1)[0]

        return preds


class ProgressNL(nn.Module):
    def __init__(
        self,
        dim,
        dim_inner,
        pool_size=None,
        instantiation="softmax",
        norm_type="layernorm",
        zero_init_final_conv=False,
        zero_init_final_norm=True,
        norm_eps=1e-5,
        norm_momentum=0.1,
    ):
        super(ProgressNL, self).__init__()
        self.dim = dim
        self.dim_inner = dim_inner
        self.pool_size = pool_size
        self.instantiation = instantiation
        self.norm_type = norm_type
        self.use_pool = (
            False
            if pool_size is None
            else True
        )
        self.norm_eps = norm_eps
        self.norm_momentum = norm_momentum
        self._construct_nonlocal(zero_init_final_conv, zero_init_final_norm)
        self.step = 0
        self.cache = None

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
        # cache
        y = x if self.step == 0 else self.cache
        if self.step == _C.PGT.STEPS - 1:
            self.step = 0
            self.cache = None
        else:
            self.step += 1
            self.cache = x.detach()

        x_identity = x
        N, C, T, H, W = x.size()

        theta = self.conv_theta(x)

        # Perform temporal-spatial pooling to reduce the computation.
        if self.use_pool:
            y = self.pool(y)

        phi = self.conv_phi(y)
        g = self.conv_g(y)

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
