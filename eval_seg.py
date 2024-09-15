import os
import sys
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

## input / output dir
parser.add_argument('--ckpt_dir', type=str, help='checkpoint folder')
parser.add_argument('--epoch', type=str,  default="1", help='epoch to evaluate')
parser.add_argument('--arch', type=str,  default='vit_small', help='vit architecture') # 'moco_vit_small', 'mae_vit_base', 'vit_small', 'dino_resnet50'
parser.add_argument('--cpu', type=str,  default='1-12', help='CPU')
parser.add_argument('--gpu', type=str,  default='0', help='GPU')
parser.add_argument('--patch_size', type=int,  default=16, help='vit patch size')
parser.add_argument('--which_matrix', type=str,  default='ours', help='Laplacian matrix')
parser.add_argument('--tau', type=float,  default=0.0, help='Tau Ncut')
parser.add_argument('--eigen_train', type=int, default=0)

args = parser.parse_args()
print (args)

FOLDER          = args.ckpt_dir
NAME            = FOLDER
epoch           = args.epoch
GPU             = args.gpu
CPU             = args.cpu
arch            = args.arch
patch_size      = args.patch_size
which_matrix    = args.which_matrix
tau             = args.tau
eigen_train=args.eigen_train

CKPT    = "checkpoints/{}/eigen-16-{}.pth".format(FOLDER, epoch)
if not os.path.exists(CKPT):
    CKPT = "1"
log_dir = "\"checkpoints/{}/log_segment.log\"".format(FOLDER)

dataset_list = ['ECSSD', 'DUTS', 'DUT-OMRON']
post_list = ['None']

for dataset in dataset_list:
        for post in post_list:
                cmd = "CUDA_VISIBLE_DEVICES={} taskset --cpu-list {} python ddt_dino_eigen.py --save_name {} --pretr_path {}  --test_dataset {} --postproc {} --patch_size {} --arch {} --which_matrix {} --tau {} ".format(GPU, CPU, NAME, CKPT, dataset, post, patch_size, arch, which_matrix, tau)    
                print(cmd)
                cmd += " 2>&1 | tee -a {}".format(log_dir)
                os.system(cmd)
