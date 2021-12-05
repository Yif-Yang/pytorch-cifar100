# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import shutil
from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
from torch.nn.functional import softmax
import logging

def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('USNet')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger
def train(epoch, aux_dis_lambda=1, main_dis_lambda=1):
    optimizer = optimizer_fc

    start = time.time()
    net.train()
    Batch_time = AverageMeter('batch_time', ':6.3f')
    Data_time = AverageMeter('Data time', ':6.3f')
    Train_loss = AverageMeter('Train_loss', ':6.3f')
    Loss_cls_1 = AverageMeter('Loss_cls_1', ':6.3f')
    Loss_cls_2 = AverageMeter('Loss_cls_2', ':6.3f')
    Loss_cls_3 = AverageMeter('Loss_cls_3', ':6.3f')
    Loss_ensemble = AverageMeter('Loss_ensemble', ':6.3f')
    Loss_dis = AverageMeter('Loss_dis', ':6.3f')
    Acc1 = AverageMeter('Acc1@1', ':6.2f')
    Acc2 = AverageMeter('Acc2@1', ':6.2f')
    Acc3 = AverageMeter('Acc3@1', ':6.2f')
    Acc_ens = AverageMeter('Acc_ens@1', ':6.2f')
    progress = ProgressMeter(
        len(cifar100_training_loader),
        [Batch_time, Data_time, Train_loss, Loss_cls_1, Loss_cls_2, Loss_cls_3, Loss_dis, Acc1, Acc2, Acc3, Acc_ens], prefix="Epoch: [{}]".format(epoch))
    end = time.time()
    logger.info(f"epoch: {epoch} LR: {optimizer.param_groups[0]['lr']}")
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        Data_time.update(time.time() - end)

        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        res1, res2, res3 = outputs
        ens = 0
        for k in outputs:
            ens += k
        ens /= len(outputs)
        loss_cls_1 = loss_function(res1, labels)
        loss_cls_2 = loss_function(res2, labels)
        loss_cls_3 = loss_function(res3, labels)
        loss_ensemble = loss_function(ens, labels)
        loss_dis = dis_criterion(softmax(res1, 1), softmax(res2, 1)) + dis_criterion(softmax(res1, 1), softmax(res3, 1)) + dis_criterion(softmax(res2, 1), softmax(res3, 1)) / 3
        loss = - loss_dis * aux_dis_lambda if epoch // 10 % 2 == 0 else 0
        if args.loss_aux_single:
            loss += (loss_cls_1 + loss_cls_2 + loss_cls_3) / 3
        if args.loss_aux_ensemble:
            loss += loss_ensemble
        loss.backward()
        optimizer.step()

        Train_loss.update(loss.item(), labels.size(0))
        Loss_cls_1.update(loss_cls_1.item(), labels.size(0))
        Loss_cls_2.update(loss_cls_2.item(), labels.size(0))
        Loss_cls_3.update(loss_cls_3.item(), labels.size(0))
        Loss_ensemble.update(loss_ensemble.item(), labels.size(0))
        Loss_dis.update(loss_dis.item(), labels.size(0))

        acc_1 = accuracy(res1, labels)[0]
        acc_2 = accuracy(res2, labels)[0]
        acc_3 = accuracy(res3, labels)[0]
        acc_ens = accuracy(ens, labels)[0]
        Acc1.update(acc_1[0], labels.size(0))
        Acc2.update(acc_2[0], labels.size(0))
        Acc3.update(acc_3[0], labels.size(0))
        Acc_ens.update(acc_ens[0], labels.size(0))

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        # last_layer = list(net.children())[-1]
        # for name, para in last_layer.named_parameters():
        #     if 'weight' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
        #     if 'bias' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)
        writer.add_scalar('Train/loss_cls_1', loss_cls_1.item(), n_iter)
        writer.add_scalar('Train/loss_cls_2', loss_cls_2.item(), n_iter)
        writer.add_scalar('Train/loss_cls_3', loss_cls_3.item(), n_iter)
        writer.add_scalar('Train/loss_aux_ensemble', loss_ensemble.item(), n_iter)
        writer.add_scalar('Train/loss_dis', loss_dis.item(), n_iter)
        Batch_time.update(time.time() - end)
        end = time.time()
        if batch_index % args.print_freq == 0:
            progress.display(batch_index)
        if epoch <= args.warm:
            warmup_scheduler.step()

    # for name, param in net.named_parameters():
    #     layer, attr = os.path.splitext(name)
    #     attr = attr[1:]
    #     writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    logger.info('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(epoch=0, tb=True, output_num=3):

    net.eval()
    Batch_time = AverageMeter('batch_time', ':6.3f')
    Data_time = AverageMeter('Data time', ':6.3f')
    Test_loss = AverageMeter('Train_loss', ':6.3f')
    Loss_cls_1 = AverageMeter('Loss_cls_1', ':6.3f')
    Loss_cls_2 = AverageMeter('Loss_cls_2', ':6.3f')
    Loss_cls_3 = AverageMeter('Loss_cls_3', ':6.3f')
    Loss_ensemble = AverageMeter('Loss_ensemble', ':6.3f')
    Loss_dis = AverageMeter('Loss_dis', ':6.3f')
    Acc1 = AverageMeter('Acc1@1', ':6.2f')
    Acc2 = AverageMeter('Acc2@1', ':6.2f')
    Acc3 = AverageMeter('Acc3@1', ':6.2f')
    Acc_ens = AverageMeter('Acc_ens@1', ':6.2f')
    t5_Acc1 = AverageMeter('t5_Acc1@1', ':6.2f')
    t5_Acc2 = AverageMeter('t5_Acc2@1', ':6.2f')
    t5_Acc3 = AverageMeter('t5_Acc3@1', ':6.2f')
    t5_Acc_ens = AverageMeter('t5_Acc_ens@1', ':6.2f')
    progress = ProgressMeter(
        len(cifar100_test_loader),
        [Batch_time, Data_time, Loss_cls_1, Loss_cls_2, Loss_cls_3, Loss_dis, Acc1, Acc2, Acc3, Acc_ens, t5_Acc1, t5_Acc2, t5_Acc3, t5_Acc_ens, ], prefix="Test Epoch: [{}]".format(epoch))
    end = time.time()
    import numpy as np
    all_res_1, all_res_2, all_res_3, all_res_ens, label_all = [], [], [], [], []

    for (images, labels) in cifar100_test_loader:
        Data_time.update(time.time() - end)

        images = images.cuda()
        labels = labels.cuda()

        outputs = net(images)
        res1, res2, res3 = outputs
        ens = 0
        for k in outputs:
            ens += k
        ens /= len(outputs)
        all_res_1.append(res1)
        all_res_2.append(res2)
        all_res_3.append(res3)
        all_res_ens.append(ens)
        label_all.append(labels)
        loss_cls_1 = loss_function(res1, labels)
        loss_cls_2 = loss_function(res2, labels)
        loss_cls_3 = loss_function(res3, labels)
        loss_ensemble = loss_function(ens, labels)
        loss_dis = (dis_criterion(softmax(res1, 1), softmax(res2, 1)) + dis_criterion(softmax(res1, 1), softmax(res3, 1)) + dis_criterion(softmax(res2, 1), softmax(res3, 1))) / 3
        loss = - loss_dis * args.aux_dis_lambda
        if args.loss_aux_single:
            loss += (loss_cls_1 + loss_cls_2 + loss_cls_3) / 3
        if args.loss_aux_ensemble:
            loss += loss_ensemble
        Test_loss.update(loss.item(), labels.size(0))
        Loss_cls_1.update(loss_cls_1.item(), labels.size(0))
        Loss_cls_2.update(loss_cls_2.item(), labels.size(0))
        Loss_cls_3.update(loss_cls_3.item(), labels.size(0))
        Loss_ensemble.update(loss_ensemble.item(), labels.size(0))
        Loss_dis.update(loss_dis.item(), labels.size(0))

        acc_1, t5_acc_1 = accuracy(res1, labels, topk=(1, 5))
        acc_2, t5_acc_2 = accuracy(res2, labels, topk=(1, 5))
        acc_3, t5_acc_3 = accuracy(res3, labels, topk=(1, 5))
        acc_ens, t5_acc_ens = accuracy(ens, labels, topk=(1, 5))
        Acc1.update(acc_1[0], labels.size(0))
        Acc2.update(acc_2[0], labels.size(0))
        Acc3.update(acc_3[0], labels.size(0))
        Acc_ens.update(acc_ens[0], labels.size(0))
        t5_Acc1.update(t5_acc_1[0], labels.size(0))
        t5_Acc2.update(t5_acc_2[0], labels.size(0))
        t5_Acc3.update(t5_acc_3[0], labels.size(0))
        t5_Acc_ens.update(t5_acc_ens[0], labels.size(0))
        Batch_time.update(time.time() - end)
    logger.info(progress.display_avg())
    all_res_1 = torch.cat(all_res_1, dim=0)
    all_res_2 = torch.cat(all_res_2, dim=0)
    all_res_3 = torch.cat(all_res_3, dim=0)
    all_res_ens = torch.cat(all_res_ens, dim=0)
    label_all = torch.cat(label_all, dim=0)
    _, pred_1 = all_res_1.topk(1, 1, True, True)
    _, pred_2 = all_res_2.topk(1, 1, True, True)
    _, pred_3 = all_res_3.topk(1, 1, True, True)
    _, pred_ens = all_res_ens.topk(1, 1, True, True)
    aux_cls_map = torch.cat((pred_1 == label_all.view(-1, 1), pred_2 == label_all.view(-1, 1), pred_3 == label_all.view(-1, 1)),dim=1)
    emsemble_miscls = torch.sum(torch.sum(pred_ens == label_all.view(-1, 1), dim=-1) == 1)

    ensemble_0_aux_1 = torch.sum(torch.sum(aux_cls_map[torch.sum(pred_ens == label_all.view(-1, 1), dim=-1) == 1], dim=-1) == 1)
    ensemble_0_aux_2 = torch.sum(torch.sum(aux_cls_map[torch.sum(pred_ens == label_all.view(-1, 1), dim=-1) == 1], dim=-1) == 2)
    ensemble_0_aux_3 = torch.sum(torch.sum(aux_cls_map[torch.sum(pred_ens == label_all.view(-1, 1), dim=-1) == 1], dim=-1) == 3)
    print(f'ensemble shotted {emsemble_miscls}, in which {ensemble_0_aux_1} sample in aux_cls shotted 1 time, {ensemble_0_aux_2} shotted 2 time, {ensemble_0_aux_3} shotted 3 time')
    emsemble_miscls = torch.sum(torch.sum(pred_ens == label_all.view(-1, 1), dim=-1) == 0)

    ensemble_0_aux_1 = torch.sum(torch.sum(aux_cls_map[torch.sum(pred_ens == label_all.view(-1, 1), dim=-1) == 0], dim=-1) == 1)
    ensemble_0_aux_2 = torch.sum(torch.sum(aux_cls_map[torch.sum(pred_ens == label_all.view(-1, 1), dim=-1) == 0], dim=-1) == 2)
    ensemble_0_aux_3 = torch.sum(torch.sum(aux_cls_map[torch.sum(pred_ens == label_all.view(-1, 1), dim=-1) == 0], dim=-1) == 3)
    print(f'ensemble missed {emsemble_miscls}, but {ensemble_0_aux_1} sample in aux_cls shotted 1 time, {ensemble_0_aux_2} shotted 2 time, {ensemble_0_aux_3} shotted 3 time')
    #see label
    # label_all[torch.sum(pred_ens == label_all.view(-1, 1), dim=-1) == 0][
    #     [torch.sum(aux_cls_map[torch.sum(pred_ens == label_all.view(-1, 1), dim=-1) == 0], dim=-1) == 1]][
    #     5].cpu().numpy()
    # torch.stack((softmax(all_res_1, 1), softmax(all_res_2, 1), softmax(all_res_3, 1)), dim=1)[
    #     torch.sum(pred_ens == label_all.view(-1, 1), dim=-1) == 0][
    #     [torch.sum(aux_cls_map[torch.sum(pred_ens == label_all.view(-1, 1), dim=-1) == 0], dim=-1) == 1]][
    #     5].cpu().numpy()
    return Acc_ens.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        # print('\t'.join(entries))
        logger.info('\t'.join(entries))

    def display_avg(self):
        entries = [self.prefix ]
        entries += [f"{meter.name}:{meter.avg:6.3f}" for meter in self.meters]
        # print('\t'.join(entries))
        logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(os.path.split(filename)[0], 'model_best.pth.tar'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-work_dir', type=str, default='./work_dir/debug_embedding_head', help='dir name')
    parser.add_argument('-blob_dir', type=str, default='/blob_aml_k8s_skt_australiav100data/output/ensemble/cifar100', help='dir name')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-start_epoch', type=int, default=1, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-print_freq', type=int, default=100, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-aux_dis_lambda', type=float, default=1, help='aux_dis_lambda loss rate')
    parser.add_argument('-main_dis_lambda', type=float, default=1, help='main_dis_lambda loss rate')
    parser.add_argument('-resume', type=str, default=None, help='dir name')
    parser.add_argument('-loss_aux_ensemble', action='store_true', default=False, help='loss_aux_ensemble')
    parser.add_argument('-loss_aux_single', action='store_true', default=False, help='loss_aux_ensemble')
    parser.add_argument('-seed', type=int, default=-1, metavar='S', help='random seed (default: 1)')

    args = parser.parse_args()
    if args.seed > 0:
        print(f'set seed {args.seed}')
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        import random
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        print('not set seed')
    net = get_network(args)
    net = net.cuda(args.gpu)
    if not os.path.exists(args.work_dir):
        os.mkdir(args.work_dir)
    logger = get_logger(os.path.join(args.work_dir, 'train.log'))
    logger.info(args)
    logger.info(net)
    settings.LOG_DIR = os.path.join(args.work_dir, 'pb')
    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )
    dis_criterion = torch.nn.L1Loss()
    # fc_params = []
    # encoder_params = []
    # for name, para in net.named_parameters():
    #     if para.requires_grad:
    #         if "fc" in name or 'linear_aux' in name:
    #             fc_params += [para]
    #         else:
    #             encoder_params += [para]
    # fc_params_list = [
    #     {"params": fc_params, "lr": args.lr},
    #     # {"params": encoder_params, "lr": args.lr},
    # ]
    # encoder_params_list = [
    #     # {"params": fc_params, "lr": args.lr},
    #     {"params": encoder_params, "lr": args.lr},
    # ]
    for name, param in net.named_parameters():
        if param.requires_grad:
            if not ('linear_aux' in name or 'fc' in name):
                print(f'close {name}')
                param.requires_grad = False

    loss_function = nn.CrossEntropyLoss()
    # optimizer_fc = optim.SGD(fc_params_list, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # optimizer_encoder = optim.SGD(encoder_params_list, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.AdamW(net.parameters(), lr=args.lr)
    optimizer_fc = optimizer
    train_scheduler_encoder = optim.lr_scheduler.MultiStepLR(optimizer_fc, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer_fc, iter_per_epoch * args.warm)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            # args.start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'], strict=False)
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            eval_training(checkpoint['epoch'], output_num=3)
            print('-------------------- test for resumed ckpt --------------------')
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 3, 32, 32)
    input_tensor = input_tensor.cuda()
    writer.add_graph(net, input_tensor)

    #create checkpoint folder to save model
    checkpoint_path = os.path.join(args.work_dir, 'ckpt')
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    best_acc = 0.0
    best_ep = 0

    for epoch in range(args.start_epoch, settings.EPOCH + 1):
        if epoch > args.warm:
            # train_scheduler_fc.step(epoch)
            train_scheduler_encoder.step(epoch)

        train(epoch, aux_dis_lambda=args.aux_dis_lambda, main_dis_lambda=args.main_dis_lambda)
        acc = eval_training(epoch, output_num=3)
        #start to save best performance model after learning rate decay to 0.01
        if best_acc < acc:
            best_acc = acc
            best_ep = epoch
            save_checkpoint({
                'epoch': epoch,
                'arch': args.net,
                'state_dict': net.state_dict(),
                # 'optimizer' : optimizer.state_dict(),
            }, is_best=True, filename=checkpoint_path+'checkpoint_{:04d}.pth.tar'.format(epoch))
        elif epoch % 10 == 0:
                save_checkpoint({
                    'epoch': epoch,
                    'arch': args.net,
                    'state_dict': net.state_dict(),
                    # 'optimizer': optimizer.state_dict(),
                }, is_best=False, filename=checkpoint_path + 'checkpoint_{:04d}.pth.tar'.format(epoch))
        logger.info(f'epoch({epoch}): best acc-{best_acc:6.3f} from ep {best_ep}')


    os.system(f'cp -r {args.work_dir} {args.blob_dir}')
    writer.close()
