import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel.distributed as dist

from slowfast.models import optimizer as optim
from slowfast.models.batchnorm_helper import FrozenBatchNorm3d
from slowfast.models.head_helper import ResNetBasicHead, ResNetRoIHead, X3DHead
from slowfast.utils import distributed as du
from slowfast.utils import logging as logging
from slowfast.utils import misc as misc


class ProgressTrainer(object):
    def __init__(
        self,
        model,
        cfg,
        epoch,
        optimizer=None,
        loss_fun=None,
        tblogger=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fun = loss_fun

        self.steps = cfg.PGT.STEPS
        self.overlap = cfg.PGT.OVERLAP
        self.num_frames = cfg.PGT.STEP_LEN
        self.train_together = cfg.PGT.TRAIN_TOGETHER
        self.progress_eval = cfg.PGT.PG_EVAL
        self.ensemble_method = cfg.PGT.ENSEMBLE_METHOD
        self.truncate_grad = cfg.PGT.TRUNCATE_GRAD
        self.logger = logging.get_logger(__name__)
        self.tblogger = tblogger

        self.multigrid = cfg.PGT.MGRID
        self.mgrid_steps = cfg.PGT.MGRID_STEPS
        self.mgrid_step_len = cfg.PGT.MGRID_STEP_LEN
        self.mgrid_lr_scales = cfg.PGT.MGRID_LRSCALES
        self.mgrid_noft = cfg.PGT.MGRID_NO_FINETUNE
        self.max_epoch = cfg.SOLVER.MAX_EPOCH

        self.single_pathway = cfg.MODEL.ARCH in cfg.MODEL.SINGLE_PATHWAY_ARCH

        if self.multigrid and self.model.training:
            # keep last epoch hyper-params align with non-multigrid setting
            # to finetune (in cfg.SOLVER)
            if epoch != cfg.SOLVER.MAX_EPOCH - 1 or self.mgrid_noft:
                self.n_schedule = len(self.mgrid_steps)
                cur_idx = epoch % self.n_schedule
                self.steps = self.mgrid_steps[cur_idx]
                self.num_frames = self.mgrid_step_len[cur_idx]
            else:
                self.steps = cfg.PGT.STEPS
                self.num_frames = cfg.PGT.STEP_LEN

            for m in self.model.modules():
                if m._get_name() == "ResBlock":
                    m.update_nframes(self.num_frames)

            if self.tblogger:
                self.tblogger.add_scalar("mgrid/steps", self.steps, epoch + 1)
                self.tblogger.add_scalar("mgrid/nframes", self.num_frames, epoch + 1)

        if self.model.training or cfg.PGT.PG_EVAL:
            tpool_size = [1] if self.single_pathway else [1, 1]
            if cfg.PGT.TPOOL_SIZE != tpool_size:
                self.logger.warn(f"You set PGT.TPOOL_SIZE {cfg.PGT.TPOOL_SIZE} but this setting is modified to {tpool_size} because PGT.PG_EVAL is on.")
        else:
            tpool_size = cfg.PGT.TPOOL_SIZE

        if cfg.MODEL.FINAL_POOL[1] == "avg":
            t_pool = [
                nn.AdaptiveAvgPool3d((t, None, None))
                for t in tpool_size
            ]
        elif cfg.MODEL.FINAL_POOL[1] == "max":
            t_pool = [
                nn.AdaptiveMaxPool3d((t, None, None))
                for t in tpool_size
            ]
        ms = self.model.module if cfg.NUM_GPUS > 1 else self.model
        if hasattr(ms, "head"):
            if isinstance(ms.head, ResNetBasicHead) or isinstance(ms.head, ResNetRoIHead):
                ms.head.s0_tpool = t_pool[0]
                if not self.single_pathway:
                    ms.head.s1_tpool = t_pool[1]
            elif isinstance(ms.head, X3DHead):
                ms.head.t_pool = t_pool[0]
        else:  # regnet
            ms.avgpool = nn.AdaptiveAvgPool3d((tpool_size[0], 1, 1))

        # log model for 1st epoch and test
        if du.is_master_proc() and (epoch == 0 or epoch is None):
            self.logger.info(str(ms.head))

    def step_train(self, inputs, labels, bboxes=None, step_idxes=None):
        losses = []
        preds = []
        last_labels = None

        for step in range(self.steps):
            if step_idxes != None:
                assert (step_idxes == step).any()
            if step == 0:
                start_idx = [0, 0]
                end_idx = self.num_frames

                pg_modules = ["ResBlock", "VideoModelStem", "ResNetBasicHead"]
                for m in self.model.modules():
                    if m._get_name() in pg_modules:
                        m.clear_cache()
            else:
                start_idx, end_idx = [], []
                for nf, ov in zip(self.num_frames, self.overlap):
                    start_idx.append(step * nf - (step - 1) * ov)
                    end_idx.append(start_idx[-1] + nf - ov)
            if self.single_pathway:
                pg_input = [inputs[0][:, :, start_idx[0]:end_idx[0]]]
            else:
                pg_input = [
                    inputs[0][:, :, start_idx[0]:end_idx[0]],
                    inputs[1][:, :, start_idx[1]:end_idx[1]]
                ]

            # Forward
            if bboxes != None:
                pred = self.model(pg_input, bboxes[step_idxes == step])
            else:
                pred = self.model(pg_input)

            if len(labels.size()) > 2:  # for charades
                loss = self.loss_fun(pred, labels[:, step])
            elif bboxes is not None:  # for AVA
                last_labels = labels[step_idxes == step]
                loss = self.loss_fun(pred, last_labels)
            else:  # for kinetics
                loss = self.loss_fun(pred, labels)

            # Check Nan Loss.
            misc.check_nan_losses(loss)

            # Perform the backward pass.
            loss.backward(retain_graph=self.truncate_grad)

            if not self.train_together:
                # Update the parameters for each iter.
                self.optimizer.step()
                self.optimizer.zero_grad()

            preds.append(pred)
            losses.append(loss.detach())

        if self.train_together:
            # Update the parameters together.
            self.optimizer.step()
            self.optimizer.zero_grad()

        preds = pred  # take the last step for train acc/map calucaltion
        loss_mean = torch.stack(losses, dim=0).mean()
        return preds, last_labels, loss_mean  # return labels of last step for AVA

    @torch.no_grad()
    def step_eval(self, inputs, bboxes=None, step_idxes=None):
        if not self.progress_eval:
            if bboxes is not None:  # AVA
                # slice inputs to the last step
                step = self.steps - 1
                start_idx, end_idx = [], []
                for nf, ov in zip(self.num_frames, self.overlap):
                    start_idx.append(step * nf - (step - 1) * ov)
                    end_idx.append(start_idx[-1] + nf - ov)
                if self.single_pathway:  # Add one frame to last step
                    slices = [slice(start_idx[0] - 1, end_idx[0])]
                else:
                    slices = [
                        slice(start_idx[0] - 1, end_idx[0]),
                        slice(start_idx[1] - 1, end_idx[1])
                    ]
                
                preds = self.model(inputs, bboxes, slices)
            else:
                preds = self.model(inputs)

        else:
            preds = []

            for step in range(self.steps):
                if step_idxes is not None:
                    assert (step_idxes == step).any()
                if step == 0:
                    start_idx = [0, 0]
                    end_idx = self.num_frames
                else:
                    start_idx, end_idx = [], []
                    for nf, ov in zip(self.num_frames, self.overlap):
                        start_idx.append(step * nf - (step - 1) * ov)
                        end_idx.append(start_idx[-1] + nf - ov)
                if self.single_pathway:
                    pg_input = [inputs[0][:, :, start_idx[0]:end_idx[0]]]
                else:
                    pg_input = [
                        inputs[0][:, :, start_idx[0]:end_idx[0]],
                        inputs[1][:, :, start_idx[1]:end_idx[1]]
                    ]

                # Forward
                if bboxes != None:
                    if step_idxes is None:
                        pred = self.model(pg_input, bboxes)
                    else:
                        pred = self.model(pg_input, bboxes[step_idxes == step])
                else:
                    pred = self.model(pg_input)

                preds.append(pred)
            
            if bboxes is not None:
                if step_idxes is None:
                    preds = pred  # Take the result of last step
                else:  # Return result of all steps
                    preds = torch.cat(preds, dim=0)
            else:
                if self.ensemble_method == "sum":
                    preds = torch.stack(preds, dim=1).sum(dim=1)
                elif self.ensemble_method == "max":
                    preds = torch.stack(preds, dim=1).max(dim=1)[0]

        return preds

    def set_lr(self, lr, epoch, global_step):
        if self.multigrid:
            if epoch != self.max_epoch - 1 or self.mgrid_noft:
                self.n_schedule = len(self.mgrid_steps)
                cur_idx = epoch % self.n_schedule
                scale = self.mgrid_lr_scales[cur_idx]
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr * scale
        else:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
        if self.tblogger:
            lr = self.optimizer.param_groups[0]["lr"]
            self.tblogger.add_scalar("mgrid/lr", lr, global_step)


class ProgressNL(nn.Module):
    def __init__(
        self,
        cfg,
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
        self.cfg = cfg
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
        elif self.norm_type == "frozen_batchnorm":
            self.bn = FrozenBatchNorm3d(
                self.dim, eps=self.norm_eps, momentum=self.norm_momentum)
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
        if self.step == self.cfg.PGT.STEPS - 1:
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
        elif self.norm_type == "frozen_batchnorm":
            p = self.bn(p)
        elif self.norm_type == "layernorm":
            p = self.ln(p)

        return x_identity + p
