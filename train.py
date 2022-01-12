# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""
import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import shutil
from conf import settings
from utils import get_training_dataloader, get_test_dataloader
import logging
from torchensemble.voting import VotingClassifier
from torchensemble.utils.logging import set_logger
from torchensemble.utils import io
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

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(os.path.split(filename)[0], 'model_best.pth.tar'))
def display_records(records, logger):
    msg = (
        "{:<28} | Testing Acc: {:.2f} % | Training Time: {:.2f} s |"
        " Evaluating Time: {:.2f} s"
    )

    print("\n")
    for method, training_time, evaluating_time, acc in records:
        logger.info(msg.format(method, acc, training_time, evaluating_time))
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
    parser.add_argument('-aux_dis_lambda', type=float, default=0, help='aux_dis_lambda loss rate')
    parser.add_argument('-hm_value', type=float, default=2, help='hm_value loss rate')
    parser.add_argument('-resume', type=str, default=None, help='dir name')
    parser.add_argument('-nesterov', action='store_true', default=False, help='resume training')
    parser.add_argument('-seed', type=int, default=-1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('-n_estimators', type=int, default=3, metavar='S', help='random seed (default: 1)')
    parser.add_argument('-n_jobs', type=int, default=1, metavar='S', help='n_jobs ')
    parser.add_argument('-add_cls_w', action='store_true', default=False, help='add_cls_w training')
    parser.add_argument('-add_dis_w', action='store_true', default=False, help='add_dis_w training')
    parser.add_argument('-distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('-distillation-tau', default=1.0, type=float, help="")
    parser.add_argument('-distillation-type', default='soft', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('-div-tau', default=1.0, type=float, help="")

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
    loss_function = nn.CrossEntropyLoss(reduce=False)
    from models.resnet_new import ResNet, BasicBlock
    net = VotingClassifier(
        estimator=ResNet, n_estimators=args.n_estimators, estimator_args={"block": BasicBlock, "num_blocks": [2, 2, 2, 2], 'num_classes':100}, cuda=True, n_jobs=args.n_jobs, args=args
    )
    net.set_optimizer('SGD', lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=args.nesterov)
    net.set_scheduler('MultiStepLR', milestones=settings.MILESTONES, gamma=0.2)
    net.set_criterion(loss_function)
    if not os.path.exists(args.work_dir):
        os.mkdir(args.work_dir)
    logger = get_logger(os.path.join(args.work_dir, 'train.log'))
    logger.info(args)
    logger.info(net)
    settings.LOG_DIR = os.path.join(args.work_dir, 'pb')
    #create checkpoint folder to save model
    checkpoint_path = os.path.join(args.work_dir, 'ckpt')
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
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
    logger = set_logger(
        use_tb_logger=True, log_path=args.work_dir
    )
    tic = time.time()
    # net.fit(cifar100_training_loader,
    #     epochs=settings.EPOCH,
    #     log_interval=args.print_freq,
    #     test_loader=cifar100_test_loader,
    #     save_model=True,
    #     save_dir=checkpoint_path,
    #     )
    toc = time.time()
    training_time = toc - tic
    if args.resume:
        io.load(net, args.resume)
    # Evaluating
    tic = time.time()
    # testing_acc, testing_loss = net.evaluate(cifar100_test_loader, return_loss=True)
    testing_acc = net.evaluate(cifar100_test_loader, return_loss=True)
    toc = time.time()
    evaluating_time = toc - tic
    # records = []
    # records.append(
    #     ("VotingClassifier", training_time, evaluating_time, testing_acc)
    # )
    # display_records(records, logger)
    cp_cmd = f'cp -r {args.work_dir} {args.blob_dir}'
    logger.info(cp_cmd)
    os.system(cp_cmd)
