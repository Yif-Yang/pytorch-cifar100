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

cls_cls_map = {}
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
    is_classification,
    logger,
    aux_dis_lambda,
    hm_value
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
    Loss_dis = AverageMeter('Loss_dis', ':6.3f')
    Acc1 = AverageMeter('Acc1@1', ':6.2f')
    Acc2 = AverageMeter('Acc2@1', ':6.2f')
    Acc_ens = AverageMeter('Acc_ens@1', ':6.2f')
    Acc_same = AverageMeter('Acc_same@1', ':6.2f')
    Exit_rate = AverageMeter('Exit_rate@1', ':6.2f')
    Acc_ens_sf = AverageMeter('Acc_ens_sf@1', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [Batch_time, Data_time, Train_loss, Loss_cls_1, Loss_cls_2, Loss_cls_ens, Loss_dis, Exit_rate, Acc_same, Acc1, Acc2, Acc_ens, Acc_ens_sf],
        prefix=f"Epoch: [{epoch}] Ens:[{idx}] Lr:[{optimizer.state_dict()['param_groups'][0]['lr']:.5f}]",
    logger = logger)

    end = time.time()
    for batch_idx, elem in enumerate(train_loader):
        Data_time.update(time.time() - end)

        data_id, data, target = io.split_data_target(elem, device)
        batch_size = target.size(0)

        optimizer.zero_grad()
        cls1, cls2 = estimator_now(*data)
        ens = (cls1 + cls2) / 2
        ens_sf = (F.softmax(cls1, 1) + F.softmax(cls2, 1)) / 2

        acc_1, correct_1 = accuracy(cls1, target)
        acc_2, correct_2 = accuracy(cls2, target)
        acc_ens, correct_ens = accuracy(ens, target)
        acc_ens_sf, correct_ens_sf = accuracy(ens_sf, target)
        _, pred_1 = cls1.topk(1, 1, True, True)
        _, pred_2 = cls2.topk(1, 1, True, True)
        _, pred_ens = ens.topk(1, 1, True, True)
        exit_mask = (pred_1 == pred_2).view(-1)

        exit_rate = torch.sum(exit_mask) / batch_size * 100
        acc_same, _ = accuracy(ens[exit_mask], target[exit_mask]) if exit_rate > 0 else (ens.new_tensor([1]), 0)

        loss_cls_1 = criterion(cls1, target)
        loss_cls_2 = criterion(cls2, target)
        loss_cls_ens = criterion(ens, target)
        dis_criterion = torch.nn.L1Loss()

        loss_dis = dis_criterion(F.softmax(cls1, 1), F.softmax(cls2, 1))

        loss = (loss_cls_1 + loss_cls_2) / 2 - loss_dis * aux_dis_lambda

        loss.backward()
        optimizer.step()

        Train_loss.update(loss.item(), batch_size)
        Loss_cls_1.update(loss_cls_1.item(), batch_size)
        Loss_cls_2.update(loss_cls_2.item(), batch_size)
        Loss_cls_ens.update(loss_cls_ens.item(), batch_size)
        Loss_dis.update(loss_dis.item(), batch_size)

        Exit_rate.update(exit_rate.item(), batch_size)

        Acc1.update(acc_1[0].item(), batch_size)
        Acc2.update(acc_2[0].item(), batch_size)
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
    aux_dis_lambda
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
    Loss_dis = AverageMeter('Loss_dis', ':6.3f')
    Acc1 = AverageMeter('Acc1@1', ':6.2f')
    Acc2 = AverageMeter('Acc2@1', ':6.2f')
    Acc_ens = AverageMeter('Acc_ens@1', ':6.2f')
    Acc_same = AverageMeter('Acc_same@1', ':6.2f')

    Exit_rate = AverageMeter('Exit_rate@1', ':6.2f')
    Acc_ens_sf = AverageMeter('Acc_ens_sf@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [Batch_time, Data_time, Train_loss, Loss_cls_1, Loss_cls_2, Loss_cls_ens, Loss_dis, Exit_rate, Acc_same, Acc1, Acc2, Acc_ens, Acc_ens_sf],
        prefix=f"Epoch: [{epoch}] Ens:[{idx}]",
    logger = logger)

    end = time.time()
    for batch_idx, elem in enumerate(train_loader):
        Data_time.update(time.time() - end)

        data_id, data, target = io.split_data_target(elem, device)
        batch_size = target.size(0)

        cls1, cls2 = estimator(*data)
        ens = (cls1 + cls2) / 2
        ens_sf = (F.softmax(cls1, 1) + F.softmax(cls2, 1)) / 2

        acc_1, correct_1 = accuracy(cls1, target)
        acc_2, correct_2 = accuracy(cls2, target)
        acc_ens, correct_ens = accuracy(ens, target)
        acc_ens_sf, correct_ens_sf = accuracy(ens_sf, target)
        _, pred_1 = cls1.topk(1, 1, True, True)
        _, pred_2 = cls2.topk(1, 1, True, True)
        _, pred_ens = ens.topk(1, 1, True, True)
        exit_mask = (pred_1 == pred_2).view(-1)
        exit_rate = torch.sum(exit_mask) / batch_size * 100
        acc_same, _ = accuracy(ens[exit_mask], target[exit_mask]) if exit_rate > 0 else (ens.new_tensor([1]), 0)

        loss_cls_1 = criterion(cls1, target)
        loss_cls_2 = criterion(cls2, target)
        loss_cls_ens = criterion(ens, target)
        dis_criterion = torch.nn.L1Loss(reduce=False)

        loss_dis = torch.mean(dis_criterion(F.softmax(cls1, 1), F.softmax(cls2, 1)), dim=-1)

        cls_loss = (loss_cls_1 + loss_cls_2) / 2
        loss = torch.mean(loss_dis.detach() / torch.mean(loss_dis.detach()) * cls_loss) - torch.mean(
            loss_dis * (torch.max(cls_loss.detach()) - cls_loss.detach()) / (
                        torch.max(cls_loss.detach()) - torch.mean(cls_loss.detach()))) * aux_dis_lambda
        loss_cls_1 = torch.mean(loss_cls_1)
        loss_cls_2 = torch.mean(loss_cls_2)
        loss_cls_ens = torch.mean(loss_cls_ens)
        loss_dis = torch.mean(loss_dis)

        Train_loss.update(loss.item(), batch_size)
        Loss_cls_1.update(loss_cls_1.item(), batch_size)
        Loss_cls_2.update(loss_cls_2.item(), batch_size)
        Loss_cls_ens.update(loss_cls_ens.item(), batch_size)

        Loss_dis.update(loss_dis.item(), batch_size)

        Exit_rate.update(exit_rate.item(), batch_size)

        Acc1.update(acc_1[0].item(), batch_size)
        Acc2.update(acc_2[0].item(), batch_size)
        Acc_same.update(acc_same[0].item(), batch_size)
        Acc_ens.update(acc_ens[0].item(), batch_size)
        Acc_ens_sf.update(acc_ens_sf[0].item(), batch_size)
        Batch_time.update(time.time() - end)
    logger.info(progress.display_avg())

    return ens

@torchensemble_model_doc(
    """Implementation on the VotingClassifier.""", "model"
)
class VotingClassifier(BaseClassifier):
    @torchensemble_model_doc(
        """Implementation on the data forwarding in VotingClassifier.""",
        "classifier_forward",
    )
    def forward(self, *x):
        # Average over class distributions from all base estimators.
        outputs = []
        for idx, estimator in enumerate(self.estimators_):

            cls1, cls2 = estimator(*x)
            ens = (cls1 + cls2) / 2
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
        progress = ProgressMeter(
            len(test_loader),
            [Acc1, Acc1_sf],
            prefix=f"Eval: ",
            logger = self.logger)
        for _, elem in enumerate(test_loader):
            idx, data, target = io.split_data_target(
                elem, self.device
            )
            batch_size = target.size(0)
            output, output_sf = self._forward(self.estimators_, *data)
            acc_1, correct_1 = accuracy(output, target)
            acc_sf, correct_1_sf = accuracy(output_sf, target)
            Acc1.update(acc_1[0].item(), batch_size)
            Acc1_sf.update(acc_sf[0].item(), batch_size)
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
    def _forward(self, estimators, *x):
        outputs = []
        outputs_sf = []
        for idx, estimator in enumerate(estimators):

            cls1, cls2 = estimator(*x)
            ens = (cls1 + cls2) / 2
            ens_sf = (F.softmax(cls1, 1) + F.softmax(cls2, 1)) / 2
            _, pred_1 = cls1.topk(1, 1, True, True)
            _, pred_2 = cls2.topk(1, 1, True, True)
            ret = F.softmax(ens, dim=1)
            if len(outputs) > 0:
                ret[mask] = outputs[-1][mask]
                ens_sf[mask] = outputs_sf[-1][mask]
            outputs.append(ret)
            outputs_sf.append(ens_sf)
            mask = pred_1.view(-1) == pred_2.view(-1)
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
        aux_dis_lambda=0,
        hm_value=5
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

                cls1, cls2 = estimator(*x)
                ens = (cls1 + cls2) / 2
                ens_sf = (F.softmax(cls1, 1) + F.softmax(cls2, 1)) / 2
                _, pred_1 = cls1.topk(1, 1, True, True)
                _, pred_2 = cls2.topk(1, 1, True, True)
                ret = F.softmax(ens, dim=1)
                if len(outputs) > 0:
                    ret[mask] = outputs[-1][mask]
                    ens_sf[mask] = outputs_sf[-1][mask]
                outputs.append(ret)
                outputs_sf.append(ens_sf)
                mask = pred_1.view(-1) == pred_2.view(-1)
            proba = op.average(outputs)
            proba_sf = op.average(outputs_sf)

            return proba, proba_sf
        # Internal helper function on pesudo forward
        def _forward(estimators, *x):
            outputs = []
            outputs_sf = []
            for idx, estimator in enumerate(estimators):

                cls1, cls2 = estimator(*x)
                ens = (cls1 + cls2) / 2
                ens_sf = (F.softmax(cls1, 1) + F.softmax(cls2, 1)) / 2

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
                    True,
                    self.logger,
                    aux_dis_lambda,
                    hm_value
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
                                                 aux_dis_lambda
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
                            # output, output_sf = _forward(estimators[:train_idx + 1], *data)
                            output, output_sf = _forward(estimators[:train_idx + 1], *data)
                            # output_ex, output_ex_sf = _forward_ex(estimators[:train_idx + 1], *data)
                            acc_1, _ = accuracy(output, target)
                            acc_sf, _ = accuracy(output_sf, target)
                            # acc1_ex, _ = accuracy(output_ex, target)
                            # acc1_ex_sf, _ = accuracy(output_ex_sf, target)
                            Acc1.update(acc_1[0].item(), batch_size)
                            Acc1_sf.update(acc_sf[0].item(), batch_size)
                            # Acc1_ex.update(acc1_ex[0].item(), batch_size)
                            # Acc1_ex_sf.update(acc1_ex_sf[0].item(), batch_size)
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

        self.estimators_ = nn.ModuleList()
        self.estimators_.extend(estimators)
        if save_model and not test_loader:
            io.save(self, save_dir, self.logger)

    @torchensemble_model_doc(item="classifier_evaluate")
    def evaluate(self, test_loader, return_loss=False):
        return super().evaluate(test_loader, return_loss)

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
