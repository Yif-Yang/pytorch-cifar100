import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('-net', type=str, default='resnet18_new', help='net type')
parser.add_argument('-work_dir', type=str, default='./work_dir', help='dir name')
parser.add_argument('-exp_name', type=str, default='early_exit_cls-as-disw', help='exp_name ')
parser.add_argument('-runs', type=str, default='1', help='exp_name ')
parser.add_argument('-blob_dir', type=str, default='/blob_aml_k8s_skt_yif_australiav100data/output/ensemble/cifar100',
                    help='dir name')
parser.add_argument('-gpu',  type=int, default=0, help='batch size for dataloader')
parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('-resume', action='store_true', default=False, help='resume training')
parser.add_argument('-seed', type=int, default=-1, metavar='S', help='random seed (default: 1)')
parser.add_argument('-nesterov', action='store_false', default=True, help='nesterov training')
parser.add_argument('-n_estimators', type=int, default=3, metavar='S', help='random seed (default: 1)')
parser.add_argument('-aux_dis_lambda', type=float, default=0, help='aux_dis_lambda loss rate')

args = parser.parse_args()
if not os.path.exists('./work_dir'):
      os.mkdir('./work_dir')
exp_name = f"{args.exp_name}_{args.net}_nEns-{args.n_estimators}_lr{args.lr}_dis-length-{args.aux_dis_lambda}_{f'nesterov_' if args.nesterov else ''}{f'seed_{args.seed}' if args.seed > -1 else ''}_run-{args.runs}"
cmd = f"CUDA_VISIBLE_DEVICES={args.gpu} " \
      "nohup python train.py " \
      f"-net {args.net} " \
      f"-lr {args.lr} " \
      f"-n_estimators {args.n_estimators} " \
      f"{f'-nesterov ' if args.nesterov else ''}" \
      f"-work_dir={os.path.join(args.work_dir, exp_name)} " \
      f"-aux_dis_lambda={args.aux_dis_lambda} " \
      f"-blob_dir={args.blob_dir} " \
      f"-seed={args.seed} " \
      f"> {exp_name}.out &"
print(cmd)
os.system(cmd)