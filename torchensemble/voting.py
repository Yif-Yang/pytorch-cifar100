"""
  In voting-based ensemble, each base estimator is trained independently,
  and the final prediction takes the average over predictions from all base
  estimators.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
from joblib import Parallel, delayed

from ._base import BaseClassifier, BaseRegressor
from ._base import torchensemble_model_doc
from .utils import io
from .utils import set_module
from .utils import operator as op
from utils import WarmUpLR, accuracy, AverageMeter, ProgressMeter
import time
__all__ = ["VotingClassifier", "VotingRegressor"]
dis_criterion = torch.nn.L1Loss(reduce=False)


look_up_map = [[] for _ in range(50001)]
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements the knowledge distillation loss
"""
import torch
from torch.nn import functional as F


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self,
                 distillation_type: str, tau: float):
        super().__init__()
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.tau = tau

    def forward(self, teacher_outputs, outputs_kd):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        teacher_outputs = teacher_outputs.detach()
        if self.distillation_type == 'none':
            return 0

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = torch.mean(F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                # We provide the teacher's targets in log probability because we use log_target=True
                # (as recommended in pytorch https://github.com/pytorch/pytorch/blob/9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
                # but it is possible to give just the probabilities and set log_target=False. In our experiments we tried both.
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='none',
                log_target=True
            ) * (T * T), dim=-1)
            # We divide by outputs_kd.numel() to have the legacy PyTorch behavior.
            # But we also experiments output_kd.size(0)
            # see issue 61(https://github.com/facebookresearch/deit/issues/61) for more details
        elif self.distillation_type == 'hard':
            distillation_loss = torch.mean(F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1), reduction='none'),
                                           dim=-1)

        return distillation_loss

def _parallel_fit_per_epoch(
    train_loader,
    estimator,
    cur_lr,
    optimizer,
    criterion,
    idx,
    epoch,
    log_interval,
    device,
    logger,
    args
):

    """
    Private function used to fit base estimators in parallel.

    WARNING: Parallelization when fitting large base estimators may cause
    out-of-memory error.
    """
    if cur_lr:
        # Parallelization corrupts the binding between optimizer and scheduler
        set_module.update_lr(optimizer, cur_lr)
    if epoch == 0:
        iter_per_epoch = len(train_loader)
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch)

    estimator_now = estimator[idx]
    Batch_time = AverageMeter('batch_time', ':6.3f')
    Data_time = AverageMeter('Data time', ':6.3f')
    Train_loss = AverageMeter('Train_loss', ':6.3f')
    Loss_cls_1 = AverageMeter('Loss_cls_1', ':6.3f')
    Loss_cls_2 = AverageMeter('Loss_cls_2', ':6.3f')
    Loss_cls_ens = AverageMeter('Loss_cls_ens', ':6.3f')
    Loss_distill = AverageMeter('Loss_distill', ':6.3f')
    Loss_dis = AverageMeter('Loss_dis', ':6.3f')
    Acc1 = AverageMeter('Acc1@1', ':6.2f')
    Acc2 = AverageMeter('Acc2@1', ':6.2f')
    Acc_aux_ens = AverageMeter('Acc_aux_ens@1', ':6.2f')

    Acc_ens = AverageMeter('Acc_ens@1', ':6.2f')
    Acc_distill = AverageMeter('Acc_distill@1', ':6.2f')
    Acc_same = AverageMeter('Acc_same@1', ':6.2f')
    Exit_rate = AverageMeter('Exit_rate@1', ':6.2f')
    Acc_ens_sf = AverageMeter('Acc_ens_sf@1', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [Batch_time, Data_time, Train_loss, Loss_cls_1, Loss_cls_2, Loss_cls_ens, Loss_dis, Loss_distill, Exit_rate, Acc_same, Acc1, Acc2, Acc_aux_ens, Acc_distill, Acc_ens, Acc_ens_sf],
        prefix=f"Epoch: [{epoch}] Ens:[{idx}] Lr:[{optimizer.state_dict()['param_groups'][0]['lr']:.5f}]",
    logger = logger)
    end = time.time()
    for batch_idx, elem in enumerate(train_loader):
        Data_time.update(time.time() - end)

        data_id, data, target = io.split_data_target(elem, device)
        batch_size = target.size(0)

        optimizer.zero_grad()
        cls1, cls2, distill_out = estimator_now(*data)
        cls_aux = (cls1 + cls2) / 2
        ens = cls_aux * (1 - args.distillation_alpha) + distill_out * args.distillation_alpha
        ens_sf = ((F.softmax(cls1, 1) + F.softmax(cls2, 1)) / 2 + F.softmax(distill_out, 1)) / 2

        acc_1, correct_1 = accuracy(cls1, target)
        acc_2, correct_2 = accuracy(cls2, target)
        acc_aux_ens, correct_aux_ens = accuracy(cls_aux, target)
        acc_distill, correct_distill = accuracy(distill_out, target)
        acc_ens, correct_ens = accuracy(ens, target)
        acc_ens_sf, correct_ens_sf = accuracy(ens_sf, target)
        _, pred_1 = cls1.topk(1, 1, True, True)
        _, pred_2 = cls2.topk(1, 1, True, True)
        _, pred_aux = cls_aux.topk(1, 1, True, True)
        _, pred_distill = distill_out.topk(1, 1, True, True)

        exit_mask = pred_1.view(-1) == pred_2.view(-1)
        exit_mask = exit_mask.__and__(pred_aux.view(-1) == pred_distill.view(-1))
        exit_rate = torch.sum(exit_mask) / batch_size * 100
        acc_same, _ = accuracy(ens[exit_mask], target[exit_mask]) if exit_rate > 0 else (ens.new_tensor([1]), 0)

        loss_cls_1 = criterion(cls1, target)
        loss_cls_2 = criterion(cls2, target)

        loss_cls_ens = criterion(ens, target)

        loss_dis = torch.mean(dis_criterion(F.softmax(cls1, 1), F.softmax(cls2, 1)), dim=-1)

        cls_loss = (loss_cls_1 + loss_cls_2) / 2


        dis_w = (torch.max(cls_loss.detach()) - cls_loss.detach()) / (
        torch.max(cls_loss.detach()) - torch.mean(cls_loss.detach())) if args.add_dis_w else 1

        cls_w = loss_dis.detach() / torch.mean(loss_dis.detach()) if args.add_cls_w else 1

        loss = (cls_w if args.aux_dis_lambda > 0 else 1) * cls_loss - loss_dis * dis_w * args.aux_dis_lambda


        if idx == 0:
            loss = torch.mean(loss)
            distillation_loss = criterion(distill_out, target)
            distillation_loss = torch.mean(distillation_loss)
        else:
            exit_mask_before, ens_old = look_up(data_id)
            loss_weight = F.softmax(exit_mask_before / args.div_tau) * batch_size
            loss = torch.mean(loss_weight * loss)

            distill_criterion = DistillationLoss(
               args.distillation_type, args.distillation_tau
            )
            distillation_loss = distill_criterion(torch.mean(ens_old, dim=1), distill_out)
            distillation_loss = torch.mean(loss_weight * distillation_loss)

        loss = loss * (1 - args.distillation_alpha) + distillation_loss * args.distillation_alpha

        # loss_weight = pred_1.new_ones(target.size()) * 1.0 - hm_value
            # loss_weight[exit_mask_before] = 1.0
            # loss_weight = loss_weight * batch_size / torch.sum(loss_weight)

        loss_cls_1 = torch.mean(loss_cls_1)
        loss_cls_2 = torch.mean(loss_cls_2)
        loss_cls_ens = torch.mean(loss_cls_ens)
        loss_dis = torch.mean(loss_dis)

        loss.backward()
        optimizer.step()

        Train_loss.update(loss.item(), batch_size)
        Loss_cls_1.update(loss_cls_1.item(), batch_size)
        Loss_cls_2.update(loss_cls_2.item(), batch_size)
        Loss_cls_ens.update(loss_cls_ens.item(), batch_size)
        Loss_dis.update(loss_dis.item(), batch_size)
        Loss_distill.update(distillation_loss.item(), batch_size)

        Exit_rate.update(exit_rate.item(), batch_size)

        Acc1.update(acc_1[0].item(), batch_size)
        Acc2.update(acc_2[0].item(), batch_size)
        Acc_aux_ens.update(acc_aux_ens[0].item(), batch_size)
        Acc_distill.update(acc_distill[0].item(), batch_size)
        Acc_ens.update(acc_ens[0].item(), batch_size)
        Acc_same.update(acc_same[0].item(), batch_size)
        Acc_ens_sf.update(acc_ens_sf[0].item(), batch_size)

        Batch_time.update(time.time() - end)
        end = time.time()
        # Print training status
        if batch_idx % log_interval == 0:
            progress.display(batch_idx)

        if epoch == 0:
            warmup_scheduler.step()

    return estimator_now, optimizer

@torch.no_grad()
def _parallel_test_per_epoch(
    train_loader,
    estimator,
    criterion,
    idx,
    epoch,
    device,
    logger,
    args
):
    """
    Private function used to fit base estimators in parallel.

    WARNING: Parallelization when fitting large base estimators may cause
    out-of-memory error.
    """

    Batch_time = AverageMeter('batch_time', ':6.3f')
    Data_time = AverageMeter('Data time', ':6.3f')
    Train_loss = AverageMeter('Train_loss', ':6.3f')
    Loss_cls_1 = AverageMeter('Loss_cls_1', ':6.3f')
    Loss_cls_2 = AverageMeter('Loss_cls_2', ':6.3f')
    Loss_cls_ens = AverageMeter('Loss_cls_ens', ':6.3f')
    Loss_distill = AverageMeter('Loss_distill', ':6.3f')
    Loss_dis = AverageMeter('Loss_dis', ':6.3f')
    Acc1 = AverageMeter('Acc1@1', ':6.2f')
    Acc2 = AverageMeter('Acc2@1', ':6.2f')
    Acc_aux_ens = AverageMeter('Acc_aux_ens@1', ':6.2f')

    Acc_ens = AverageMeter('Acc_ens@1', ':6.2f')
    Acc_distill = AverageMeter('Acc_distill@1', ':6.2f')
    Acc_same = AverageMeter('Acc_same@1', ':6.2f')

    Exit_rate = AverageMeter('Exit_rate@1', ':6.2f')
    Acc_ens_sf = AverageMeter('Acc_ens_sf@1', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [Batch_time, Data_time, Train_loss, Loss_cls_1, Loss_cls_2, Loss_cls_ens, Loss_dis, Loss_distill, Exit_rate, Acc_same, Acc1, Acc2, Acc_aux_ens, Acc_distill, Acc_ens, Acc_ens_sf],
        prefix=f"Epoch: [{epoch}] Ens:[{idx}]",
    logger = logger)

    end = time.time()
    for batch_idx, elem in enumerate(train_loader):
        Data_time.update(time.time() - end)

        data_id, data, target = io.split_data_target(elem, device)
        batch_size = target.size(0)

        cls1, cls2, distill_out = estimator(*data)
        cls_aux = (cls1 + cls2) / 2
        ens = cls_aux * (1 - args.distillation_alpha) + distill_out * args.distillation_alpha
        ens_sf = ((F.softmax(cls1, 1) + F.softmax(cls2, 1)) / 2 + F.softmax(distill_out, 1)) / 2

        acc_1, correct_1 = accuracy(cls1, target)
        acc_2, correct_2 = accuracy(cls2, target)
        acc_aux_ens, correct_aux_ens = accuracy(cls_aux, target)
        acc_distill, correct_distill = accuracy(distill_out, target)
        acc_ens, correct_ens = accuracy(ens, target)
        acc_ens_sf, correct_ens_sf = accuracy(ens_sf, target)
        _, pred_1 = cls1.topk(1, 1, True, True)
        _, pred_2 = cls2.topk(1, 1, True, True)
        _, pred_aux = cls_aux.topk(1, 1, True, True)
        _, pred_distill = distill_out.topk(1, 1, True, True)

        exit_mask = pred_1.view(-1) == pred_2.view(-1)
        exit_mask = exit_mask.__and__(pred_aux.view(-1) == pred_distill.view(-1))
        exit_rate = torch.sum(exit_mask) / batch_size * 100
        acc_same, _ = accuracy(ens[exit_mask], target[exit_mask]) if exit_rate > 0 else (ens.new_tensor([1]), 0)

        loss_cls_1 = criterion(cls1, target)
        loss_cls_2 = criterion(cls2, target)

        loss_cls_ens = criterion(ens, target)

        loss_dis = torch.mean(dis_criterion(F.softmax(cls1, 1), F.softmax(cls2, 1)), dim=-1)

        cls_loss = (loss_cls_1 + loss_cls_2) / 2


        dis_w = (torch.max(cls_loss.detach()) - cls_loss.detach()) / (
        torch.max(cls_loss.detach()) - torch.mean(cls_loss.detach())) if args.add_dis_w else 1

        cls_w = loss_dis.detach() / torch.mean(loss_dis.detach()) if args.add_cls_w else 1

        loss = (cls_w if args.aux_dis_lambda > 0 else 1) * cls_loss - loss_dis * dis_w * args.aux_dis_lambda


        if idx == 0:
            loss = torch.mean(loss)
            distillation_loss = criterion(distill_out, target)
            distillation_loss = torch.mean(distillation_loss)
        else:
            exit_mask_before, ens_old = look_up(data_id)
            loss_weight = F.softmax(exit_mask_before, 0) * batch_size
            loss = torch.mean(loss_weight * loss)

            distillation_loss = criterion(distill_out, target)
            distillation_loss = torch.mean(distillation_loss)

            # distill_criterion = DistillationLoss(
            #    args.distillation_type, args.distillation_tau
            # )
            # distillation_loss = distill_criterion(torch.mean(ens_old, dim=1), distill_out)

        loss = loss * (1 - args.distillation_alpha) + distillation_loss * args.distillation_alpha

        # loss_weight = pred_1.new_ones(target.size()) * 1.0 - hm_value
            # loss_weight[exit_mask_before] = 1.0
            # loss_weight = loss_weight * batch_size / torch.sum(loss_weight)

        loss_cls_1 = torch.mean(loss_cls_1)
        loss_cls_2 = torch.mean(loss_cls_2)
        loss_cls_ens = torch.mean(loss_cls_ens)
        loss_dis = torch.mean(loss_dis)

        Train_loss.update(loss.item(), batch_size)
        Loss_cls_1.update(loss_cls_1.item(), batch_size)
        Loss_cls_2.update(loss_cls_2.item(), batch_size)
        Loss_cls_ens.update(loss_cls_ens.item(), batch_size)

        Loss_dis.update(loss_dis.item(), batch_size)
        Loss_distill.update(distillation_loss.item(), batch_size)

        Exit_rate.update(exit_rate.item(), batch_size)

        Acc1.update(acc_1[0].item(), batch_size)
        Acc2.update(acc_2[0].item(), batch_size)
        Acc_aux_ens.update(acc_aux_ens[0].item(), batch_size)
        Acc_distill.update(acc_distill[0].item(), batch_size)
        Acc_ens.update(acc_ens[0].item(), batch_size)
        Acc_same.update(acc_same[0].item(), batch_size)
        Acc_ens_sf.update(acc_ens_sf[0].item(), batch_size)

        Batch_time.update(time.time() - end)
    logger.info(progress.display_avg())

    return ens

@torch.no_grad()
def build_map_for_training(
        train_loader,
        estimator,
        idx,
        device,
        logger,
        args
):
    """
    Private function used to fit base estimators in parallel.

    WARNING: Parallelization when fitting large base estimators may cause
    out-of-memory error.
    """
    Batch_time = AverageMeter('batch_time', ':6.3f')
    Data_time = AverageMeter('Data time', ':6.3f')
    progress = ProgressMeter(
        len(train_loader),
        [Batch_time, Data_time],
        prefix=f"Build Feat Map Ens:[{idx}]",
        logger = logger)
    end = time.time()
    for batch_idx, elem in enumerate(train_loader):
        Data_time.update(time.time() - end)
        data_id, data, target = io.split_data_target(elem, device)

        cls1, cls2, distill_out = estimator(*data)
        cls_aux = (cls1 + cls2) / 2
        ens = cls_aux * (1 - args.distillation_alpha) + distill_out * args.distillation_alpha

        _, pred_1 = cls1.topk(1, 1, True, True)
        _, pred_2 = cls2.topk(1, 1, True, True)
        _, pred_ens = ens.topk(1, 1, True, True)
        # exit_mask = (pred_1 == pred_2).view(-1)
        exit_mask = torch.mean(dis_criterion(F.softmax(cls1, 1), F.softmax(cls2, 1)), dim=-1)
        # exit_mask = F.softmax()
        for id, _mask, _ens in zip(data_id, exit_mask, ens):
            look_up_map[id].append((_mask, _ens))

        Batch_time.update(time.time() - end)
    logger.info(progress.display_avg())

def look_up(indexs):
    exit_mask, ens = [], []
    for id in indexs:
        ens_now = []
        exit_now = []
        for idx, (_exit_distence, _ens) in enumerate(look_up_map[id]):
            ens_now.append(_ens.view(1, -1))
            exit_now.append(_exit_distence)

        exit_mask.append(torch.mean(torch.tensor(exit_now)))
        ens_now = torch.cat(ens_now)
        ens.append(ens_now)
    ens = torch.stack(ens)
    exit_mask = _exit_distence.new_tensor(exit_mask)
    return exit_mask, ens
@torchensemble_model_doc(
    """Implementation on the VotingClassifier.""", "model"
)
class VotingClassifier(BaseClassifier):
    @torchensemble_model_doc(
        """Implementation on the data forwarding in VotingClassifier.""",
        "classifier_forward",
    )
    def __init__(
            self,
            estimator,
            n_estimators,
            estimator_args=None,
            cuda=True,
            n_jobs=None,
            logger=None,
            args=None
    ):
        super(VotingClassifier, self).__init__(estimator,
                                               n_estimators,
                                               estimator_args,
                                               cuda,
                                               n_jobs,
                                               logger,
                                               )
        self.args = args

    def forward(self, *x):
        # Average over class distributions from all base estimators.
        outputs = []
        for idx, estimator in enumerate(self.estimators_):

            cls1, cls2, distill_out = estimator(*x)
            cls_aux = (cls1 + cls2) / 2
            ens = cls_aux * (1 - self.args.distillation_alpha) + distill_out * self.args.distillation_alpha
            ret = F.softmax(ens, dim=1)
            outputs.append(ret)

        proba = op.average(outputs)

        return proba
    @torch.no_grad()
    def evaluate(self, test_loader, return_loss=False):
        """Docstrings decorated by downstream models."""
        self.eval()

        Acc1 = AverageMeter('Acc1@1', ':6.2f')
        Acc1_sf = AverageMeter('Acc1_sf@1', ':6.2f')
        Acc_same = AverageMeter('Acc_same@1', ':6.2f')

        Exit_rate = AverageMeter('Exit_rate@1', ':6.2f')
        progress = ProgressMeter(
            len(test_loader),
            [Acc1, Acc1_sf, Acc_same, Exit_rate],
            prefix=f"Eval: ",
            logger = self.logger)
        outputs = []
        outputs_sf = []
        outputs_dis = []
        for _, elem in enumerate(test_loader):
            idx, data, target = io.split_data_target(
                elem, self.device
            )
            batch_size = target.size(0)
            for idx, estimator in enumerate(self.estimators_):

                cls1, cls2, distill_out = estimator(*data)
                cls_aux = (cls1 + cls2) / 2
                ens = cls_aux * (1 - self.args.distillation_alpha) + distill_out * self.args.distillation_alpha
                ens_sf = (F.softmax(cls1, 1) + F.softmax(cls2, 1)) / 2 * (1 - self.args.distillation_alpha) + F.softmax(distill_out, 1) * self.args.distillation_alpha
                _, pred_1 = cls1.topk(1, 1, True, True)
                _, pred_2 = cls2.topk(1, 1, True, True)
                _, pred_aux = cls_aux.topk(1, 1, True, True)
                _, pred_distill = distill_out.topk(1, 1, True, True)
                _, pred_ens = ens.topk(1, 1, True, True)
                ret = F.softmax(ens, dim=1)
                if idx > 0:
                    ret[mask] = outputs[-1][mask]
                    ens_sf[mask] = outputs_sf[-1][mask]
                outputs.append(ret)
                outputs_sf.append(ens_sf)
                mask_now = pred_1.view(-1) == pred_2.view(-1)
                mask_now = mask_now.__and__(pred_aux.view(-1) == pred_distill.view(-1))
                mask_now = mask_now.__and__(pred_ens.view(-1) == pred_distill.view(-1))
                if idx == 0:
                    mask = mask_now
                else:
                    mask_now = mask_now.__and__(pred_ens.view(-1) == old_pred.view(-1))
                    mask_now = mask_now.__and__(pred_1.view(-1) == old_pred.view(-1))
                    mask_now = mask_now.__and__(pred_2.view(-1) == old_pred.view(-1))
                    mask += mask_now
                old_pred = pred_ens
                exit_rate = torch.sum(mask) / data[0].size(0) * 100
                acc_same, _ = accuracy(ens[mask], target[mask]) if exit_rate > 0 else (ens.new_tensor([1]), 0)
                print(exit_rate, acc_same)
            proba = op.average(outputs)
            proba_sf = op.average(outputs_sf)

            acc_1, correct_1 = accuracy(proba, target)
            acc_sf, correct_1_sf = accuracy(proba_sf, target)
            Acc1.update(acc_1[0].item(), batch_size)
            Acc1_sf.update(acc_sf[0].item(), batch_size)
            Exit_rate.update(exit_rate.item(), batch_size)
            Acc_same.update(acc_same[0].item(), batch_size)
        print(progress.display_avg())
        return Acc1.avg
    @torchensemble_model_doc(
        """Set the attributes on optimizer for VotingClassifier.""",
        "set_optimizer",
    )
    def set_optimizer(self, optimizer_name, **kwargs):
        super().set_optimizer(optimizer_name, **kwargs)

    @torchensemble_model_doc(
        """Set the attributes on scheduler for VotingClassifier.""",
        "set_scheduler",
    )
    def set_scheduler(self, scheduler_name, **kwargs):
        super().set_scheduler(scheduler_name, **kwargs)

    @torchensemble_model_doc(
        """Set the training criterion for VotingClassifier.""",
        "set_criterion",
    )
    def set_criterion(self, criterion):
        super().set_criterion(criterion)

    @torchensemble_model_doc(
        """Implementation on the training stage of VotingClassifier.""", "fit"
    )
    # Internal helper function on pesudo forward
    def _forward(self, estimators, target, *x):
        outputs = []
        outputs_sf = []
        outputs_dis = []
        for idx, estimator in enumerate(estimators):

            cls1, cls2, distill_out = estimator(*x)
            cls_aux = (cls1 + cls2) / 2
            ens = cls_aux * (1 - self.args.distillation_alpha) + distill_out * self.args.distillation_alpha
            ens_sf = (F.softmax(cls1, 1) + F.softmax(cls2, 1)) / 2 * (1 - self.args.distillation_alpha) + F.softmax(distill_out, 1) * self.args.distillation_alpha
            _, pred_1 = cls1.topk(1, 1, True, True)
            _, pred_2 = cls2.topk(1, 1, True, True)
            _, pred_aux = cls_aux.topk(1, 1, True, True)
            _, pred_distill = distill_out.topk(1, 1, True, True)
            _, pred_ens = ens.topk(1, 1, True, True)
            ret = F.softmax(ens, dim=1)
            if idx > 0:
                ret[mask] = outputs[-1][mask]
                ens_sf[mask] = outputs_sf[-1][mask]
            outputs.append(ret)
            outputs_sf.append(ens_sf)
            mask_now = pred_1.view(-1) == pred_2.view(-1)
            mask_now = mask_now.__and__(pred_aux.view(-1) == pred_distill.view(-1))
            mask_now = mask_now.__and__(pred_ens.view(-1) == pred_distill.view(-1))
            if idx == 0:
                mask = mask_now
            else:
                mask_now = mask_now.__and__(pred_ens.view(-1) == old_pred.view(-1))
                mask_now = mask_now.__and__(pred_1.view(-1) == old_pred.view(-1))
                mask_now = mask_now.__and__(pred_2.view(-1) == old_pred.view(-1))
                mask += mask_now
            old_pred = pred_ens
            exit_rate = torch.sum(mask) / x[0].size(0) * 100
            acc_same, _ = accuracy(ens[mask], target[mask]) if exit_rate > 0 else (ens.new_tensor([1]), 0)
            print(exit_rate, acc_same)
        proba = op.average(outputs)
        proba_sf = op.average(outputs_sf)

        return proba, proba_sf
    def fit(
        self,
        train_loader,
        epochs=100,
        log_interval=100,
        test_loader=None,
        save_model=True,
        save_dir=None,
    ):

        self._validate_parameters(epochs, log_interval)
        self.n_outputs = self._decide_n_outputs(train_loader)

        # Instantiate a pool of base estimators, optimizers, and schedulers.
        estimators = []
        for _ in range(self.n_estimators):
            estimators.append(self._make_estimator())

        optimizers = []
        for i in range(self.n_estimators):
            optimizers.append(
                set_module.set_optimizer(
                    estimators[i], self.optimizer_name, **self.optimizer_args
                )
            )



        # Check the training criterion
        if not hasattr(self, "_criterion"):
            self._criterion = nn.CrossEntropyLoss()

        # Utils

        # Internal helper function on pesudo forward
        def _forward_ex(estimators, *x):
            outputs = []
            outputs_sf = []
            for idx, estimator in enumerate(estimators):

                cls1, cls2, distill_out = estimator(*x)
                cls_aux = (cls1 + cls2) / 2
                ens = cls_aux * (1 - self.args.distillation_alpha) + distill_out * self.args.distillation_alpha
                ens_sf = (F.softmax(cls1, 1) + F.softmax(cls2, 1)) / 2 * (1 - self.args.distillation_alpha) + F.softmax(distill_out, 1) * self.args.distillation_alpha
                _, pred_1 = cls1.topk(1, 1, True, True)
                _, pred_2 = cls2.topk(1, 1, True, True)
                _, pred_aux = cls_aux.topk(1, 1, True, True)
                _, pred_distill = distill_out.topk(1, 1, True, True)

                ret = F.softmax(ens, dim=1)
                if idx > 0:
                    ret[mask] = outputs[-1][mask]
                    ens_sf[mask] = outputs_sf[-1][mask]
                outputs.append(ret)
                outputs_sf.append(ens_sf)
                mask_now = pred_1.view(-1) == pred_2.view(-1)
                mask_now = mask_now.__and__(pred_aux.view(-1) == pred_distill.view(-1))
                if idx == 0:
                    mask = mask_now
                else:
                    mask += mask_now
            proba = op.average(outputs)
            proba_sf = op.average(outputs_sf)

            return proba, proba_sf
        # Internal helper function on pesudo forward
        def _forward(estimators, *x):
            outputs = []
            outputs_sf = []
            for idx, estimator in enumerate(estimators):

                cls1, cls2, distill_out = estimator(*x)
                cls_aux = (cls1 + cls2) / 2
                ens = cls_aux * (1 - self.args.distillation_alpha) + distill_out * self.args.distillation_alpha
                ens_sf = (F.softmax(cls1, 1) + F.softmax(cls2, 1)) / 2 * (1 - self.args.distillation_alpha) + F.softmax(distill_out, 1) * self.args.distillation_alpha

                ret = F.softmax(ens, dim=1)
                outputs.append(ret)
                outputs_sf.append(ens_sf)

            proba = op.average(outputs)
            proba_sf = op.average(outputs_sf)

            return proba, proba_sf
        # Maintain a pool of workers

        # Training loop
        for train_idx in range(self.n_estimators):
            if self.use_scheduler_:
                scheduler_ = set_module.set_scheduler(
                    optimizers[train_idx], self.scheduler_name, **self.scheduler_args
                )
            best_acc = 0.0
            if train_idx > 0:
                estimators[train_idx - 1].load_state_dict(self.estimators_dic[train_idx - 1])
            for epoch in range(epochs):
                self.train()

                if self.use_scheduler_:
                    cur_lr = scheduler_.get_last_lr()[0]
                else:
                    cur_lr = None

                if self.n_jobs and self.n_jobs > 1:
                    msg = "Parallelization on the training epoch: {:03d}"
                    self.logger.info(msg.format(epoch))

                estimators[train_idx], optimizers[train_idx] = _parallel_fit_per_epoch(
                    train_loader,
                    estimators,
                    cur_lr,
                    optimizers[train_idx],
                    self._criterion,
                    train_idx,
                    epoch,
                    log_interval,
                    self.device,
                    self.logger,
                    self.args
                )

                # Validation
                if test_loader:
                    with torch.no_grad():
                        self.eval()
                        _parallel_test_per_epoch(test_loader,
                                                 estimators[train_idx],
                                                 self._criterion,
                                                 train_idx,
                                                 epoch,
                                                 self.device,
                                                 self.logger,
                                                 self.args
                                                 )

                        Acc1 = AverageMeter('Acc1@1', ':6.2f')
                        Acc1_sf = AverageMeter('Acc1_sf@1', ':6.2f')
                        Acc1_ex = AverageMeter('Acc1_ex@1', ':6.2f')
                        Acc1_ex_sf = AverageMeter('Acc1_ex_sf@1', ':6.2f')
                        progress = ProgressMeter(
                            len(test_loader),
                            [Acc1, Acc1_sf, Acc1_ex, Acc1_ex_sf],
                            prefix=f"Epoch: [{epoch}] Ens:[{train_idx}] ",
                        logger = self.logger)
                        for _, elem in enumerate(test_loader):
                            idx, data, target = io.split_data_target(
                                elem, self.device
                            )
                            batch_size = target.size(0)
                            output, output_sf = _forward(estimators[:train_idx + 1], *data)
                            output_ex, output_ex_sf = _forward_ex(estimators[:train_idx + 1], *data)
                            acc_1, _ = accuracy(output, target)
                            acc_sf, _ = accuracy(output_sf, target)
                            acc1_ex, _ = accuracy(output_ex, target)
                            acc1_ex_sf, _ = accuracy(output_ex_sf, target)
                            Acc1.update(acc_1[0].item(), batch_size)
                            Acc1_sf.update(acc_sf[0].item(), batch_size)
                            Acc1_ex.update(acc1_ex[0].item(), batch_size)
                            Acc1_ex_sf.update(acc1_ex_sf[0].item(), batch_size)
                        acc = Acc1.avg
                        print(progress.display_avg())
                        if acc > best_acc:
                            best_acc = acc
                            self.estimators_dic = [[] for _ in range(self.n_estimators)]
                            self.estimators_dic[train_idx] = estimators[train_idx].state_dict()
                            self.estimators_ = nn.ModuleList()
                            self.estimators_.extend(estimators)
                            if save_model and train_idx + 1 == self.n_estimators:
                                io.save(self, save_dir, self.logger)

                        msg = (
                            "Train_idx: {:03d} | Epoch: {:03d} | Validation Acc: {:.3f}"
                            " % | Historical Best: {:.3f} %"
                        )
                        self.logger.info(msg.format(train_idx, epoch, acc, best_acc))
                        if self.tb_logger:
                            self.tb_logger.add_scalar(
                                f"voting/Validation_Acc_{train_idx}", acc, epoch
                            )

                # Update the scheduler
                with warnings.catch_warnings():

                    # UserWarning raised by PyTorch is ignored because
                    # scheduler does not have a real effect on the optimizer.
                    warnings.simplefilter("ignore", UserWarning)

                    if self.use_scheduler_:
                        scheduler_.step()
            build_map_for_training(train_loader, estimators[train_idx], train_idx,self.device,
                                   self.logger, self.args)
        self.estimators_ = nn.ModuleList()
        self.estimators_.extend(estimators)
        if save_model and not test_loader:
            io.save(self, save_dir, self.logger)

    # @torchensemble_model_doc(item="classifier_evaluate")
    # def evaluate(self, test_loader, return_loss=False):
    #     return super().evaluate(test_loader, return_loss)

    @torchensemble_model_doc(item="predict")
    def predict(self, *x):
        return super().predict(*x)


@torchensemble_model_doc("""Implementation on the VotingRegressor.""", "model")
class VotingRegressor(BaseRegressor):
    @torchensemble_model_doc(
        """Implementation on the data forwarding in VotingRegressor.""",
        "regressor_forward",
    )
    def forward(self, *x):
        # Average over predictions from all base estimators.
        outputs = [estimator(*x) for estimator in self.estimators_]
        pred = op.average(outputs)

        return pred

    @torchensemble_model_doc(
        """Set the attributes on optimizer for VotingRegressor.""",
        "set_optimizer",
    )
    def set_optimizer(self, optimizer_name, **kwargs):
        super().set_optimizer(optimizer_name, **kwargs)

    @torchensemble_model_doc(
        """Set the attributes on scheduler for VotingRegressor.""",
        "set_scheduler",
    )
    def set_scheduler(self, scheduler_name, **kwargs):
        super().set_scheduler(scheduler_name, **kwargs)

    @torchensemble_model_doc(
        """Set the training criterion for VotingRegressor.""",
        "set_criterion",
    )
    def set_criterion(self, criterion):
        super().set_criterion(criterion)

    @torchensemble_model_doc(
        """Implementation on the training stage of VotingRegressor.""", "fit"
    )
    def fit(
        self,
        train_loader,
        epochs=100,
        log_interval=100,
        test_loader=None,
        save_model=True,
        save_dir=None,
    ):

        self._validate_parameters(epochs, log_interval)
        self.n_outputs = self._decide_n_outputs(train_loader)

        # Instantiate a pool of base estimators, optimizers, and schedulers.
        estimators = []
        for _ in range(self.n_estimators):
            estimators.append(self._make_estimator())

        optimizers = []
        for i in range(self.n_estimators):
            optimizers.append(
                set_module.set_optimizer(
                    estimators[i], self.optimizer_name, **self.optimizer_args
                )
            )

        if self.use_scheduler_:
            scheduler_ = set_module.set_scheduler(
                optimizers[0], self.scheduler_name, **self.scheduler_args
            )

        # Check the training criterion
        if not hasattr(self, "_criterion"):
            self._criterion = nn.MSELoss()

        # Utils
        best_loss = float("inf")

        # Internal helper function on pesudo forward
        def _forward(estimators, *x):
            outputs = [estimator(*x) for estimator in estimators]
            pred = op.average(outputs)

            return pred

        # Maintain a pool of workers
        with Parallel(n_jobs=self.n_jobs) as parallel:

            # Training loop
            for epoch in range(epochs):
                self.train()

                if self.use_scheduler_:
                    cur_lr = scheduler_.get_last_lr()[0]
                else:
                    cur_lr = None

                if self.n_jobs and self.n_jobs > 1:
                    msg = "Parallelization on the training epoch: {:03d}"
                    self.logger.info(msg.format(epoch))

                rets = parallel(
                    delayed(_parallel_fit_per_epoch)(
                        train_loader,
                        estimator,
                        cur_lr,
                        optimizer,
                        self._criterion,
                        idx,
                        epoch,
                        log_interval,
                        self.device,
                        False,
                        self.logger
                    )
                    for idx, (estimator, optimizer) in enumerate(
                        zip(estimators, optimizers)
                    )
                )

                estimators, optimizers = [], []
                for estimator, optimizer in rets:
                    estimators.append(estimator)
                    optimizers.append(optimizer)

                # Validation
                if test_loader:
                    self.eval()
                    with torch.no_grad():
                        val_loss = 0.0
                        for _, elem in enumerate(test_loader):
                            data, target = io.split_data_target(
                                elem, self.device
                            )
                            output = _forward(estimators, *data)
                            val_loss += self._criterion(output, target)
                        val_loss /= len(test_loader)

                        if val_loss < best_loss:
                            best_loss = val_loss
                            self.estimators_ = nn.ModuleList()
                            self.estimators_.extend(estimators)
                            if save_model:
                                io.save(self, save_dir, self.logger)

                        msg = (
                            "Epoch: {:03d} | Validation Loss:"
                            " {:.5f} | Historical Best: {:.5f}"
                        )
                        self.logger.info(
                            msg.format(epoch, val_loss, best_loss)
                        )
                        if self.tb_logger:
                            self.tb_logger.add_scalar(
                                "voting/Validation_Loss", val_loss, epoch
                            )

                # Update the scheduler
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)

                    if self.use_scheduler_:
                        scheduler_.step()

        self.estimators_ = nn.ModuleList()
        self.estimators_.extend(estimators)
        if save_model and not test_loader:
            io.save(self, save_dir, self.logger)

    @torchensemble_model_doc(item="regressor_evaluate")
    def evaluate(self, test_loader):
        return super().evaluate(test_loader)

    @torchensemble_model_doc(item="predict")
    def predict(self, *x):
        return super().predict(*x)
