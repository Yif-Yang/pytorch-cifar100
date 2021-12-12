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
def train(epoch):

    start = time.time()
    net.train()
    Batch_time = AverageMeter('batch_time', ':6.3f')
    Data_time = AverageMeter('Data time', ':6.3f')
    Train_loss = AverageMeter('Train_loss', ':6.3f')
    Acc1 = AverageMeter('Acc1@1', ':6.2f')
    progress = ProgressMeter(
        len(cifar100_training_loader),
        [Batch_time, Data_time, Train_loss, Acc1], prefix="Epoch: [{}]".format(epoch))
    end = time.time()
    logger.info(f"epoch: {epoch} LR: {optimizer.param_groups[0]['lr']}")
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        Data_time.update(time.time() - end)

        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        Train_loss.update(loss.item(), labels.size(0))
        acc_1 = accuracy(outputs, labels)[0]
        Acc1.update(acc_1[0], labels.size(0))

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        # last_layer = list(net.children())[-1]
        # for name, para in last_layer.named_parameters():
        #     if 'weight' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
        #     if 'bias' in name:
        #         writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)
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
def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()
    Batch_time = AverageMeter('batch_time', ':6.3f')
    Data_time = AverageMeter('Data time', ':6.3f')
    Test_loss = AverageMeter('Train_loss', ':6.3f')
    Acc1 = AverageMeter('Acc1@1', ':6.2f')
    Acc5 = AverageMeter('Acc5@1', ':6.2f')
    progress = ProgressMeter(
        len(cifar100_training_loader),
        [Batch_time, Data_time, Test_loss, Acc1, Acc5], prefix="Test Epoch: [{}]".format(epoch))
    end = time.time()
    for (images, labels) in cifar100_test_loader:
        Data_time.update(time.time() - end)

        images = images.cuda()
        labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        acc_1, t5_acc = accuracy(outputs, labels, topk=(1, 5))
        Test_loss.update(loss.item(), labels.size(0))

        Acc1.update(acc_1[0], labels.size(0))
        Acc5.update(t5_acc[0], labels.size(0))
        Batch_time.update(time.time() - end)
    logger.info(progress.display_avg())

    return Acc1.avg
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
    parser.add_argument('-work_dir', type=str, default='./work_dir', help='dir name')
    parser.add_argument('-blob_dir', type=str, default='/blob_aml_k8s_skt_yif_australiav100data/output/ensemble/cifar100', help='dir name')
    parser.add_argument('-gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-start_epoch', type=int, default=1, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-print_freq', type=int, default=100, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', type=str, default=None, help='dir name')
    parser.add_argument('-nesterov', action='store_true', default=False, help='resume training')
    parser.add_argument('-seed', type=int, default=-1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('-n_estimators', type=int, default=3, metavar='S', help='random seed (default: 1)')

    args = parser.parse_args()
    if args.seed > -1:
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

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=args.nesterov)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
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
            train_scheduler.step(epoch)

        train(epoch)
        acc = eval_training(epoch)
        if best_acc < acc:
            best_acc = acc
            best_ep = epoch
            save_checkpoint({
                'epoch': epoch,
                'arch': args.net,
                'state_dict': net.state_dict(),
                # 'optimizer' : optimizer.state_dict(),
            }, is_best=True, filename=checkpoint_path+'checkpoint_{:04d}.pth.tar'.format(epoch))
        elif epoch % 20 == 0:
                save_checkpoint({
                    'epoch': epoch,
                    'arch': args.net,
                    'state_dict': net.state_dict(),
                    # 'optimizer': optimizer.state_dict(),
                }, is_best=False, filename=checkpoint_path + 'checkpoint_{:04d}.pth.tar'.format(epoch))
        logger.info(f'epoch({epoch}): best acc-{best_acc:6.3f} from ep {best_ep}')


    os.system(f'cp -r {args.work_dir} {args.blob_dir}')
    writer.close()
