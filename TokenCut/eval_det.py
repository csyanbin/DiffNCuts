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
parser.add_argument('--tau', type=float,  default=0.2, help='Tau Ncut')
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
eigen_train     = args.eigen_train
arch            = args.arch


CKPT    = "\"../checkpoints/{}/eigen-16-{}.pth\"".format(FOLDER, epoch)
log_dir = "\"../checkpoints/{}/log_det.log\"".format(FOLDER)


script_list     = ['main_tokencut.py']
dataset_list    = ['VOC07', 'VOC12', 'COCO20k']
set_list        = {'VOC07': 'trainval', 'VOC12': 'trainval', 'COCO20k': 'train'}

for script in script_list:
        for dataset in dataset_list:
                cmd = "CUDA_VISIBLE_DEVICES={} taskset --cpu-list {} python {} --dataset {} --set {} --pretr_path {}  --patch_size {} --tau {} --eigen_train {} --arch {}".format(GPU, CPU, script, dataset, set_list[dataset], CKPT, patch_size, tau, eigen_train, arch)    
                cmd += " 2>&1 | tee -a {}".format(log_dir)
                print(cmd)
                os.system(cmd)
