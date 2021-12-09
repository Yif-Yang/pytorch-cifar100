import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('-net', type=str, required=True, help='net type')
parser.add_argument('-work_dir', type=str, default='./work_dir', help='dir name')
parser.add_argument('-exp_name', type=str, default='diversity_mean_loss_encoder_step', help='exp_name ')
parser.add_argument('-runs', type=str, default='1', help='exp_name ')
parser.add_argument('-blob_dir', type=str, default='/blob_aml_k8s_skt_australiav100data/output/ensemble/cifar100',
                    help='dir name')
parser.add_argument('-gpu',  type=int, default=0, help='batch size for dataloader')
parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('-aux_dis_lambda', type=float, default=1, help='aux_dis_lambda loss rate')
parser.add_argument('-main_dis_lambda', type=float, default=1, help='main_dis_lambda loss rate')
parser.add_argument('-resume', type=str, default=None, help='dir name')
parser.add_argument('-loss_aux_ensemble', action='store_true', default=False, help='loss_aux_ensemble')
parser.add_argument('-close_fc_grad', action='store_true', default=False, help='close_fc_grad')
parser.add_argument('-loss_aux_single', action='store_true', default=False, help='loss_aux_ensemble')
parser.add_argument('-seed', type=int, default=-1, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()
if not os.path.exists('./work_dir'):
      os.mkdir('./work_dir')
exp_name = f"{args.exp_name}_{args.net}_lr{args.lr}_{'close-fc_' if args.close_fc_grad else ''}{'ens_' if args.loss_aux_ensemble else ''}{'sing_' if args.loss_aux_single else ''}dis-length-{args.aux_dis_lambda}{f'_seed_{args.seed}' if args.seed > 0 else ''}_run-{args.runs}"
cmd = f"CUDA_VISIBLE_DEVICES={args.gpu} " \
      "nohup python train_encoder_step.py " \
      f"-net {args.net} " \
      f"-lr {args.lr} " \
      f"{'-loss_aux_single ' if args.loss_aux_single else ''}" \
      f"{'-close_fc_grad ' if args.close_fc_grad else ''}" \
      f"{'-loss_aux_ensemble ' if args.loss_aux_ensemble else ''}" \
      f"-aux_dis_lambda={args.aux_dis_lambda} " \
      f"-work_dir={os.path.join(args.work_dir, exp_name)} " \
      f"-blob_dir={args.blob_dir} " \
      f"-seed={args.seed} " \
      f"> {exp_name}.out &"
print(cmd)
os.system(cmd)