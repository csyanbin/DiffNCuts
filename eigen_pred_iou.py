import torch
from util.torch_dist_sum import *
from data.dataloader import *
from data.transform_ovlp import CustomDataAugmentation
import torch.nn as nn
from util.meter import *
from network.eigen_pred_iou import EIGEN
from datetime import datetime
import time
import math
from util.LARS import LARS
from torch.nn.parallel import DistributedDataParallel
import argparse
import wandb
from util import utils

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size-pergpu', type=int, default=256)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument("--base_lr", default=0.0005, type=float)
parser.add_argument("--image_size", default=224, type=int)
parser.add_argument("--patch_size", default=16, type=int)
parser.add_argument("--save_path", default="./checkpoints/ckp_eigen/", type=str)
parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
parser.add_argument("--save_epoch", default=1, type=int)
parser.add_argument("--arch", default="vit_small", type=str, choices=['resnet50', 'moco_vit_small', 'mae_vit_base', 'vit_small'])
parser.add_argument("--resume_path", default=None, type=str)
parser.add_argument("--loss_eigtype", default="lossall", type=str)
parser.add_argument("--loss_crit", default="BCE", type=str)
parser.add_argument("--min_scale", default=0.2, type=float)
parser.add_argument("--max_scale", default=1.0, type=float)
parser.add_argument("--warm_up", default=1, type=int)

parser.add_argument("--max_thres", default=0.5, type=float)
parser.add_argument("--min_thres", default=0.5, type=float)
parser.add_argument("--epoch_thres", default=10, type=float)
parser.add_argument("--max_weight", default=0.05, type=float)
parser.add_argument("--min_weight", default=0.05, type=float)
parser.add_argument("--epoch_weight", default=4, type=float)

parser.add_argument("--weight_decay", default=1e-4, type=float)

parser.add_argument("--trainset", default="imagenet", type=str) 
parser.add_argument("--train_layers", default=-1, type=int)

parser.add_argument("--laplacian", default="ours", type=str) 
parser.add_argument("--tau", default=0.0, type=float) # 
parser.add_argument("--min_lr", default=1e-6, type=float) # 

args = parser.parse_args()
min_lr = args.min_lr
if args.patch_size == 8:
    args.pretrained_path = "dino_deitsmall8_pretrain.pth"

if args.local_rank != -1:
    torch.cuda.set_device(args.local_rank)
    device=torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method='env://')
    rank = torch.distributed.get_rank()
else:
    device=torch.device("cuda", 0)
    rank = 0

name = "LHVIoU{}{}_U{}-{}-{}_{}-{}-{}".format(args.loss_crit, args.loss_eigtype, args.max_thres, args.min_thres, args.epoch_thres, args.max_weight, args.min_weight, args.epoch_weight)
if args.patch_size!=16:
    name += 'p8'
if args.image_size!=224:
    name += 's{}'.format(args.image_size)
if args.trainset!="imagenet":
    name = args.trainset+name
if args.arch!='vit_small':
    if 'moco' in args.arch: name = 'moco'+name
    if 'mae' in args.arch: name = 'mae'+name
    if 'resnet50' in args.arch: name = 'res50'+name
if args.train_layers > 0:
    name += "_L{}".format(args.train_layers)
if args.laplacian!="ours": 
    name += args.laplacian
if args.tau>0:
    name += "tau"+str(args.tau)
if args.weight_decay!=1e-4:
    name += "wd"+str(args.weight_decay)
if args.min_scale!=0.2 or args.max_scale!=1.0:
    name += "ms{}_{}".format(args.min_scale, args.max_scale)
if args.warm_up<1:
    name += "nowarm"

args.save_path = "./checkpoints/"+name
if rank==0:
    wandb.init(project="Eigen-ImageNet", name=name, save_code=True, config=args)
    wandb.run.log_code(".")
    print(args)

epochs = args.epochs
warm_up = args.warm_up

def adjust_learning_rate(optimizer, epoch, base_lr, i, iteration_per_epoch):
    T = epoch * iteration_per_epoch + i
    warmup_iters = warm_up * iteration_per_epoch
    total_iters = (epochs - warm_up) * iteration_per_epoch

    if epoch < warm_up:
        lr = base_lr * 1.0 * T / warmup_iters
    else:
        T = T - warmup_iters
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(1.0 * T / total_iters * math.pi))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if rank==0:
        wandb.log({"lr": lr})

def train(train_loader, model, device, criterion, optimizer, epoch, iteration_per_epoch, base_lr, threshold_scheduler, weight_scheduler, checkpoint_path, args=args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    # switch to train mode
    model.train()

    end = time.time()
    for i, pack in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, base_lr, i, iteration_per_epoch)
        if epoch<args.epoch_thres and epoch*len(train_loader)+i<len(threshold_scheduler):
            threshold = threshold_scheduler[epoch*len(train_loader) + i]
        else:
            threshold = args.max_thres
        if epoch<args.epoch_weight and epoch*len(train_loader)+i<len(weight_scheduler):
            weight = weight_scheduler[epoch*len(train_loader) + i]
        else: 
            weight = args.max_weight
        data_time.update(time.time() - end)
        try:
            crops, coords, flags = pack
        except:
            print("ERROR and WHY????")
        img1 = crops[0]
        img2 = crops[1]

        img1 = img1.cuda()
        img2 = img2.cuda()

        # compute output
        loss_ovlp = model(img1, img2, coords, flags, args, i, threshold, weight)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_ovlp.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0 and rank == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch: {epoch} | Iter: {i} | loss_ovlp: {loss_ovlp} | lr: {lr} ")
        if rank==0:
            wandb.log({"loss_ovlp":loss_ovlp})
            wandb.log({"threshold":threshold, "weight_collapse": weight})
        if rank==0 and (i==int(0.5*len(train_loader))) and (epoch==0 or (epoch+1) % args.save_epoch == 0 or epoch==epochs):
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 0.5
            }, checkpoint_path.replace(".pth", "half.pth"))


def main():
    
    batch_size = args.batch_size_pergpu
    num_workers = 8
    base_lr = args.base_lr

    model = EIGEN(args.arch, device, args.patch_size, args.local_rank, train_layers=args.train_layers).cuda()
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print('use {} gpus!'.format(num_gpus))
        model = DistributedDataParallel(model, device_ids=[args.local_rank],
                                                    output_device=args.local_rank, find_unused_parameters=True)
    
    param_dict = {}
    for k, v in model.named_parameters():
        param_dict[k] = v

    bn_params = [v for n, v in param_dict.items() if ('bn' in n or 'bias' in n)]
    rest_params = [v for n, v in param_dict.items() if not ('bn' in n or 'bias' in n)]

    optimizer = torch.optim.SGD([{'params': bn_params, 'weight_decay': 0, 'ignore': True },
                                {'params': rest_params, 'weight_decay': args.weight_decay, 'ignore': False}], 
                                lr=base_lr, momentum=0.9, weight_decay=args.weight_decay)

    optimizer = LARS(optimizer, eps=0.0)
    
    torch.backends.cudnn.benchmark = True

    transform = CustomDataAugmentation(args.image_size, args.min_scale)
    if args.trainset == "imagenet": 
        train_dataset = Imagenet(aug=transform)
    else:
        train_dataset = ImagenetContrastive(aug=transform, max_class=1000)
    if num_gpus>1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=False, sampler=train_sampler, drop_last=True)
    
    iteration_per_epoch = train_loader.__len__()
    criterion = nn.CrossEntropyLoss()
    threshold_scheduler = utils.cosine_scheduler(args.min_thres, args.max_thres, args.epoch_thres, len(train_loader))
    weight_scheduler = utils.cosine_scheduler(args.min_weight, args.max_weight, args.epoch_weight, len(train_loader))

    start_epoch = 0
    
    model.train()

    if rank==0:
        ## Resume from checkpoint
        if args.resume_path:
            to_restore = {"epoch": 0}
            utils.restart_from_checkpoint(args.resume_path, run_variables=to_restore, model=model, optimizer=optimizer)
            start_epoch = to_restore["epoch"]
            print("resume from: {}, epoch: {:d}".format(args.resume_path,start_epoch))
        if os.path.exists(args.save_path): 
            args.save_path += datetime.now().strftime("_%Y-%m-%d-%H-%M-%S")
            print("Save_path now is:", args.save_path)
        os.makedirs(args.save_path, exist_ok=False)

    for epoch in range(start_epoch, epochs):
        
        checkpoint_path = args.save_path + '/eigen-16-{}.pth'.format(epoch)
        if num_gpus>1:
            train_sampler.set_epoch(epoch)
        train(train_loader, model, device, criterion, optimizer, epoch, iteration_per_epoch, base_lr, threshold_scheduler, weight_scheduler, checkpoint_path)
        checkpoint_path = args.save_path + '/eigen-16-{}.pth'.format(epoch+1)
        
        if rank==0 and (epoch==0 or (epoch+1) % args.save_epoch == 0 or epoch==epochs):
            torch.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1
            }, checkpoint_path)

if __name__ == "__main__":
    main()
