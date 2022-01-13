import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('-net', type=str, default='resnet18_new', help='net type')
parser.add_argument('-work_dir', type=str, default='./work_dir', help='dir name')
parser.add_argument('-exp_name', type=str, default='hm', help='exp_name ')
parser.add_argument('-runs', type=str, default='fix_mask', help='exp_name ')
parser.add_argument('-blob_dir', type=str, default='/blob_aml_k8s_skt_yif_resrchvc4data/output/ensemble/cifar100',
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
parser.add_argument('-hm_value', type=float, default=1, help='hm_value loss rate')
parser.add_argument('-add_cls_w', action='store_true', default=False, help='add_cls_w training')
parser.add_argument('-add_dis_w', action='store_true', default=False, help='add_dis_w training')
parser.add_argument('-distillation-type', default='soft', choices=['none', 'soft', 'hard'], type=str, help="")
parser.add_argument('-distillation-alpha', default=0.5, type=float, help="")
parser.add_argument('-distillation-tau', default=1.0, type=float, help="")
parser.add_argument('-div-tau', default=1.0, type=float, help="")
parser.add_argument('-hm_add_dis', action='store_true', default=False, help='add_dis_w training')

args = parser.parse_args()
if not os.path.exists('./work_dir'):
      os.mkdir('./work_dir')
exp_name = f"{args.exp_name}_{args.net}_nEns-{args.n_estimators}_lr{args.lr}_dis-length-{args.aux_dis_lambda}_hm-value-{args.hm_value}_" \
           f"{f'nesterov_' if args.nesterov else ''}{f'add_cls_w_' if args.add_cls_w else ''}{f'add_dis_w_' if args.add_dis_w else ''}{f'hm_add_dis_' if args.hm_add_dis else ''}" \
           f"distillation-type-{args.distillation_type}_distillation-alpha-{args.distillation_alpha}_distillation-tau-{args.distillation_tau}_" \
           f"_div-tau-{args.div_tau}_{f'seed_{args.seed}' if args.seed > -1 else ''}_run-{args.runs}"
cmd = f"CUDA_VISIBLE_DEVICES={args.gpu} " \
      "nohup python train.py " \
      f"-net {args.net} " \
      f"-lr {args.lr} " \
      f"-n_estimators {args.n_estimators} " \
      f"{f'-nesterov ' if args.nesterov else ''}" \
      f"-work_dir={os.path.join(args.work_dir, exp_name)} " \
      f"-aux_dis_lambda={args.aux_dis_lambda} " \
      f"-hm_value={args.hm_value} " \
      f"-distillation-type={args.distillation_type} " \
      f"-distillation-alpha={args.distillation_alpha} " \
      f"-distillation-tau={args.distillation_tau} " \
      f"-div-tau={args.div_tau} " \
      f"{f'-add_cls_w ' if args.add_cls_w else ''}" \
      f"{f'-add_dis_w ' if args.add_dis_w else ''}" \
      f"{f'-hm_add_dis ' if args.hm_add_dis else ''}" \
      f"-blob_dir={args.blob_dir} " \
      f"-seed={args.seed} " \
      f"> {exp_name}.out &"
print(cmd)
os.system(cmd)