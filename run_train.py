import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('-net', type=str, default='resnet18', help='net type')
parser.add_argument('-work_dir', type=str, default='./work_dir', help='dir name')
parser.add_argument('-exp_name', type=str, default='baseline_nesterov', help='exp_name ')
parser.add_argument('-runs', type=str, default='1', help='exp_name ')
parser.add_argument('-blob_dir', type=str, default='/blob_aml_k8s_skt_australiav100data/output/ensemble/cifar100',
                    help='dir name')
parser.add_argument('-gpu',  type=int, default=0, help='batch size for dataloader')
parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('-resume', action='store_true', default=False, help='resume training')
parser.add_argument('-seed', type=int, default=-1, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()
if not os.path.exists('./work_dir'):
      os.mkdir('./work_dir')
exp_name = f"{args.exp_name}_{args.net}_lr{args.lr}_{f'seed_{args.seed}' if args.seed > -1 else ''}_run-{args.runs}"
cmd = f"CUDA_VISIBLE_DEVICES={args.gpu} " \
      "nohup python train.py " \
      f"-net {args.net} " \
      f"-lr {args.lr} " \
      f"-work_dir={os.path.join(args.work_dir, exp_name)} " \
      f"-blob_dir={args.blob_dir} " \
      f"-seed={args.seed} " \
      f"> {exp_name}.out &"
print(cmd)
os.system(cmd)