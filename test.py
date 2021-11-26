#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse

from matplotlib import pyplot as plt
import time
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from pycm import *
from conf import settings
from utils import get_network, get_test_dataloader, get_training_dataloader
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
        print('\t'.join(entries))

    def display_avg(self):
        entries = [self.prefix ]
        entries += [f"{meter.name}:{meter.avg:6.3f}" for meter in self.meters]
        # print('\t'.join(entries))
        print('\t'.join(entries))

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
def draw_softmax_img(out, labels):
    for idx, pred in enumerate(out):
        pred = pred[labels.sort()[1]]
        image_array = softmax(pred, 1).cpu().numpy() * 256
        from PIL import Image
        im = Image.fromarray(image_array)
        im = im.convert('L')
        im.save(f'plot/acc_{idx}.jpg')

def ensemble(outputs):
    ens = 0
    for k in outputs:
        ens += k
    ens /= len(outputs)
    return ens
def ensemble_max(outputs):
    res1, res2, res3 = outputs
    ens = []
    return ens
@torch.no_grad()
def test(epoch=0, tb=True, output_num=3):

    net.eval()
    Batch_time = AverageMeter('batch_time', ':6.3f')
    Data_time = AverageMeter('Data time', ':6.3f')
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
        [Batch_time, Data_time, Loss_cls_1, Loss_cls_2, Loss_cls_3, Loss_dis, Acc1, Acc2, Acc3, Acc_ens, t5_Acc1, t5_Acc2, t5_Acc3, t5_Acc_ens], prefix="Test Epoch: [{}]".format(epoch))
    end = time.time()
    import numpy as np
    all_res_1, all_res_2, all_res_3, all_res_ens, label_all = [], [], [], [], []

    for (images, labels) in cifar100_test_loader:
        Data_time.update(time.time() - end)

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        res1, res2, res3 = outputs

        ens = ensemble(outputs)

        all_res_1.append(res1)
        all_res_2.append(res2)
        all_res_3.append(res3)
        all_res_ens.append(ens)
        label_all.append(labels)
        loss_cls_1 = loss_function(res1, labels)
        loss_cls_2 = loss_function(res2, labels)
        loss_cls_3 = loss_function(res3, labels)
        loss_ensemble = loss_function(ens, labels)
        loss_dis = dis_criterion(softmax(res1, 1), softmax(res2, 1)) + dis_criterion(softmax(res1, 1), softmax(res3, 1)) + dis_criterion(softmax(res2, 1), softmax(res3, 1))
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
    print(progress.display_avg())
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    args = parser.parse_args()
    dis_criterion = torch.nn.L1Loss()

    loss_function = torch.nn.CrossEntropyLoss()
    net = get_network(args)

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        #settings.CIFAR100_PATH,
        num_workers=4,
        batch_size=args.b,
    )

    net.load_state_dict(torch.load(args.weights))
    print(net)
    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    test(net, tb=False, output_num=3)
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
