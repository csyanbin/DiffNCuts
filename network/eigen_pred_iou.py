# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from network.head import *
from network.resnet import *
import torch.nn.functional as F
import network.vision_transformer as vits
import numpy as np
import os
import torchvision

from util.eigen_decomp import EigenDecompositionFcnFast
import wandb
from PIL import Image

from network.utils import get_model

def convert_image(image_array):
    N, C, H, W = image_array.size()
    image_array = image_array.detach().permute(0,2,3,1).cpu().numpy()
    image_show = []
    for image in image_array:
        image_pil = Image.fromarray(image.squeeze().astype(np.uint8))
        image_show.append(wandb.Image(image_pil))
        
    return image_show
            
class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self):
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        mean = torch.as_tensor(img_mean)
        std = torch.as_tensor(img_std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
    

def get_eigenvectors(feats, imgs=None, which_matrix: str = 'laplacian', K=2, threshold_tau=0.0, train=True):
    # Eigenvectors of laplacian matrix
    A = torch.bmm(feats, feats.transpose(1,2)) 
    if threshold_tau>-1:
        _W_semantic = (A * (A>threshold_tau))
        _W_semantic += 1e-6 * (A<=threshold_tau)     # Avoid ill-conditioned and multiple same eigenvalue
    #_W_semantic = _W_semantic / _W_semantic.max()    # NOTE: If features are normalized, this naturally does nothing
    
    if which_matrix == 'ours':
        diag = _W_semantic.sum(2)         # row sum
        diag[diag < 1e-8] = 1.0           # prevent from dividing by zero
        D = torch.diag_embed(diag)        # diagonal
        Laplacian = D - _W_semantic
    
        eigenvalues, eigenvectors = EigenDecompositionFcnFast.apply(Laplacian.double(), K)

    if eigenvectors.isnan().any():
        print("Eigenvector contains Nan, the eigenvalues are ", eigenvalues)
        
    # Sign ambiguity (assume foreground is smaller than background areas at test time)
    batch_mask = torch.ones(eigenvectors.size(0), 1, eigenvectors.size(-1), device=feats.device)
    if not train:
        for b in range(eigenvectors.size(0)):
            for k in range(K):
                if 0.5 < torch.mean((eigenvectors[b, :, k] > 0).float()).item() < 1.0:  # reverse segment
                    batch_mask[b, 0, k] = -1.0
    eigenvectors =  batch_mask * eigenvectors

    return eigenvalues.float(), eigenvectors.float()



class EIGEN(nn.Module):
    def __init__(self, arch, device, patch_size=16, local_rank=-1, dim_hidden=384, dim=256, train_layers=-1):
        super(EIGEN, self).__init__()
        
        if 'mae' in arch:
            dim_input = 768
        elif 'resnet' in arch:
            dim_input = 2048
        else:
            dim_input = 384
        # pretrained_path is arch: resnet50, moco_vit_small, mae_vit_base, vit_small (patch=16|8)
        self.net = get_model(arch, patch_size)
        # ViT ---- Freeze layers, if desired
        if 'vit' in arch and train_layers > 0:
            num_unfrozen = -train_layers
            for module in list(self.net.children())[0:2]: # 0 PatchEmbed, 1 Dropout
                for p in module.parameters():
                    p.requires_grad_(False)
            for module in list(self.net.children())[2][:num_unfrozen]:  ## ViT DINO
                for p in module.parameters():
                    p.requires_grad_(False)
        # ResNet50 ---- Freeze layers, if desired
        if 'resnet' in arch and train_layers > 0:
            num_unfrozen = -train_layers
            for module in self.net.features[:num_unfrozen]:
                for p in module.parameters():
                    p.requires_grad_(False)
            for p in self.net.features[7][0].parameters():
                    p.requires_grad_(False)
        print(f'Parameters (total): {sum(p.numel() for p in self.net.parameters()):_d}')
        print(f'Parameters (train): {sum(p.numel() for p in self.net.parameters() if p.requires_grad):_d}')
        
        print(f"Dim Hidden:{dim_hidden}")
        self.head1 = ProjectionHead(dim_in=dim_input, dim_out=dim, dim_hidden=dim_hidden)
        self.head2 = ProjectionHead(dim_in=dim_input, dim_out=dim, dim_hidden=dim_hidden)
        self.head2d_proj = ProjectionHead2d(in_dim=dim_input, bottleneck_dim=dim, hidden_dim=dim_hidden, nlayers=2) ## 384-384, 384-256
        self.head2d_pred = ProjectionHead2d(in_dim=dim, bottleneck_dim=dim, hidden_dim=dim_hidden, nlayers=2) ## 256-384, 384-256
        self.device = device
        self.patch_size = patch_size
        self.arch = arch

        ## Syncronize Batch Normalization
        if local_rank!=-1:
            #self.net = nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
            self.head1 = nn.SyncBatchNorm.convert_sync_batchnorm(self.head1)
            self.head2 = nn.SyncBatchNorm.convert_sync_batchnorm(self.head2)
            self.head2d_proj = nn.SyncBatchNorm.convert_sync_batchnorm(self.head2d_proj)
            self.head2d_pred = nn.SyncBatchNorm.convert_sync_batchnorm(self.head2d_pred)

        
    def invaug(self, x, coords, flags):
        N, C, H, W = x.shape

        batch_idxs = torch.arange(N, device=coords.device).view(N, 1)
        coords_rescaled = coords.clone()
        coords_rescaled[:, 0] = coords_rescaled[:, 0] * W  # x1
        coords_rescaled[:, 2] = coords_rescaled[:, 2] * W  # x2
        coords_rescaled[:, 1] = coords_rescaled[:, 1] * H  # y1
        coords_rescaled[:, 3] = coords_rescaled[:, 3] * H  # y2
        coords_with_idxs = torch.cat([batch_idxs, coords_rescaled], dim=1)
        
        x_aligned = torchvision.ops.roi_align(x, coords_with_idxs.to(x.device), (H, W), aligned=True)
        x_flipped = torch.stack([feat.flip(-1) if flag else feat for feat, flag in zip(x_aligned, flags)])
        return x_flipped

    def entropy(self, inp, dim=0): 
        inp = inp
        return F.cross_entropy(inp, inp.softmax(dim=dim))
    # Binary Cross Entropy
    def binary_cross_entropy(self, inp1, inp2, temp1=0.05, temp2=0.05, reduce=False):
        # inp1 and inp2 are eigenvectors without sigmoid
        inp2 = inp2.detach()
        #inp1 = torch.sigmoid(inp1/temp1)  # p1
        #inp2 = torch.sigmoid(inp2/temp2)  # p2
        BCE  = nn.BCELoss(reduce=reduce)
        return BCE(inp1, inp2)
    
    def IOU(self, mask1, mask2, loss_crit="BCE", cuda=True):
        iou, img_num = 0.0, 0.0
        if loss_crit in ["BCE"]:
            thres = 0.5
        
        mask1, mask2 = (mask1>thres).to(torch.bool), (mask2>thres).to(torch.bool)
        intersection = torch.sum(mask1 * (mask1 == mask2), dim=[-1]).squeeze()
        union = torch.sum(mask1 + mask2, dim=[-1]).squeeze()
        iou = intersection.to(torch.float) / union
        
        return iou

    def eigen_loss(self, eigen1, eigen2, args, threshold=0.5):
        if args is not None and args.local_rank != -1:
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        if rank==0:
            wandb.log({"threshold": threshold})
            
        n, hw = eigen1.size()
        loss_etype = args.loss_eigtype
        if args.loss_crit in ["BCE"]:
            temp1, temp2 = 0.05, 0.05
            if args.loss_crit=="BCE":
                criterion = self.binary_cross_entropy 
            
            map1, map2 = torch.sigmoid(eigen1 / temp1), torch.sigmoid(eigen2 / temp2)
            map22      = 1-map2      # 0.5 to determine FG/BG
            ##
            with torch.no_grad():
                #fg_avg, bg_avg = 1, 0
                fg_avg, bg_avg = ((map1>0.5)*map1).sum(dim=1) / (map1>0.5).sum(dim=1), ((map1<=0.5)*map1).sum(dim=1) / (map1<=0.5).sum(dim=1)
                pseudo_mask1, pseudo_mask2 = torch.zeros_like(map1, device=map1.device), torch.zeros_like(map1, device=map1.device)
                pseudo_mask1[:, 0:hw//2], pseudo_mask1[:, hw//2:] = fg_avg.view(n,1), bg_avg.view(n,1)  
                pseudo_mask2[:, 0:hw//2], pseudo_mask2[:, hw//2:] = bg_avg.view(n,1), fg_avg.view(n,1)  
                pseudo_mask1V, pseudo_mask2V = torch.zeros_like(map1, device=map1.device), torch.zeros_like(map1, device=map1.device)
                pseudo_mask1V = pseudo_mask1V.resize(n, int(np.sqrt(hw)), int(np.sqrt(hw)))
                pseudo_mask1V[:, :, 0:int(np.sqrt(hw)/2)], pseudo_mask1V[:, :, int(np.sqrt(hw)/2):] = fg_avg.view(n,1,1), bg_avg.view(n,1,1)
                pseudo_mask1V = pseudo_mask1V.view(n,-1)
                pseudo_mask2V = pseudo_mask2V.resize(n, int(np.sqrt(hw)), int(np.sqrt(hw)))
                pseudo_mask2V[:, :, 0:int(np.sqrt(hw)/2)], pseudo_mask2V[:, :, int(np.sqrt(hw)/2):] = bg_avg.view(n,1,1), fg_avg.view(n,1,1)
                pseudo_mask2V = pseudo_mask2V.view(n,-1)
            all_avg = map1.mean(dim=1) 
            pseudo_mask3 = all_avg.view(n,1) * torch.ones_like(map1, device=map1.device)
        ## prevent mask from collapse into half/half masks
        loss_collapse1 = criterion(map1, pseudo_mask1.detach()).mean(1)
        loss_collapse2 = criterion(map1, pseudo_mask2.detach()).mean(1)
        loss_collapse = torch.minimum(loss_collapse1, loss_collapse2).mean() 
        loss_collapse1V = criterion(map1, pseudo_mask1V.detach()).mean(1)
        loss_collapse2V = criterion(map1, pseudo_mask2V.detach()).mean(1)
        loss_collapseV = torch.minimum(loss_collapse1V, loss_collapse2V).mean() 
        loss_uniform = criterion(map1, pseudo_mask3.detach()).mean(1).mean()
        with torch.no_grad():
            entropyB = self.entropy(map1, dim=0)
            entropyHW = self.entropy(map1, dim=1)
        if rank==0:
            wandb.log({"entropyB": entropyB, "entropyHW": entropyHW})
        lossfg = criterion(map1, map2.detach())        # BxHW, Original Loss
        lossbg = criterion(map1, map22.detach())       # BxHW, Flipped Sign Loss
        IoU_fg = self.IOU(map1, map2, args.loss_crit)
        IoU_bg = self.IOU(map1, map22, args.loss_crit)

        lossfg = lossfg.mean(1)   # BxHW --> B
        lossbg = lossbg.mean(1)   # BxHW --> B
        
        if loss_etype in ["lossall"]:            
            batch_num = 0
            threshold_tmp = threshold
            while batch_num<20:
                batch_mask1, batch_mask2 = torch.zeros_like(lossfg).cuda(), torch.zeros_like(lossbg).cuda()
                for b in range(n): 
                    batch_mask1[b] = 1 if IoU_fg[b]>IoU_bg[b] and IoU_fg[b]>threshold_tmp else 0
                    batch_mask2[b] = 1 if IoU_bg[b]>=IoU_fg[b] and IoU_bg[b]>threshold_tmp else 0
                
                batch_mask = batch_mask1+batch_mask2    
                loss = batch_mask1*lossfg + batch_mask2*lossbg
                IoU  = batch_mask1*IoU_fg + batch_mask2*IoU_bg
                batch_num   = batch_mask.sum()
                threshold_tmp -= 0.05
            
            fg_bool = 1.0*(batch_mask1>batch_mask2)    # B
            map2 = map2*fg_bool.view(n, 1) + map22*(1-fg_bool).view(n,1)
            map1 = map1[batch_mask.bool(),:]
            map2 = map2[batch_mask.bool(),:]
            loss_ovlp = (loss*batch_mask).sum() / batch_num
            
            if rank==0:
                wandb.log({"threshold_tmp": threshold_tmp+0.05})
                wandb.log({"IoU": IoU.sum()/batch_num})
            
        if loss_etype in ['lossall']:
            if rank==0:
                wandb.log({"batch_num": batch_num})
        
        return loss_ovlp, map1, map2, batch_mask, loss_collapse, loss_collapseV, loss_uniform, pseudo_mask1, pseudo_mask1V
    
    
    def forward(self, x1, x2, coords, flags, args=None, itr=0, threshold=0.5, weight=0.05, t=0.1):

        if 'resnet' not in self.arch:
            feat_out = {}
            def hook_fn_forward_qkv(module, input, output):
                feat_out["qkv"] = output
            self.net._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)

            attentions = self.net.get_last_selfattention(x1)
            attentions = attentions.detach()
            # Dimensions
            nb_im = attentions.shape[0]  # Batch size
            nh = attentions.shape[1]  # Number of heads
            nb_tokens = attentions.shape[2]  # Number of tokens

            # <feat1> 
            bakcbone_feat1 = self.net(x1)

            qkv1 = (
                feat_out["qkv"]
                .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
                .permute(2, 0, 3, 1, 4).contiguous()
            )
            q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]
            k1 = k1.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
            if self.patch_size == 16:
                patch_nb = 14
            else:
                patch_nb = 28
            if args.image_size!=224:
                patch_nb = int(args.image_size/self.patch_size)

            feat1_2d = k1.permute(0,2,1).contiguous()[:,:,1:].view(nb_im, -1, patch_nb, patch_nb)
            feat1_2d_proj_nonorm = self.head2d_proj(feat1_2d)                    ## Projection MLP
            feat1_2d_pred = F.normalize(self.head2d_pred(feat1_2d_proj_nonorm))  ## Prediction MLP
            # <feat1> 

            # <feat2>
            bakcbone_feat2 = self.net(x2)

            qkv2 = (
                feat_out["qkv"]
                .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
                .permute(2, 0, 3, 1, 4).contiguous()
            )
            q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]
            k2 = k2.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
            q2 = q2.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
            v2 = v2.transpose(1, 2).reshape(nb_im, nb_tokens, -1)

            feat2_2d = k2.permute(0,2,1).contiguous()[:,:,1:].view(nb_im, -1, patch_nb, patch_nb)
            feat2_2d_proj_nonorm = self.head2d_proj(feat2_2d)                    ## Projection MLP
            feat2_2d_pred = F.normalize(self.head2d_pred(feat2_2d_proj_nonorm))  ## Prediction MLP
        
        else:
            feat1_2d = self.net.forward(x1)
            nb_im, d, patch_nb, _ = feat1_2d.shape
            feat1_2d_proj_nonorm = self.head2d_proj(feat1_2d)                    ## Projection MLP
            feat1_2d_pred = F.normalize(self.head2d_pred(feat1_2d_proj_nonorm))  ## Prediction MLP

            feat2_2d = self.net.forward(x2)
            feat2_2d_proj_nonorm = self.head2d_proj(feat2_2d)                    ## Projection MLP
            feat2_2d_pred = F.normalize(self.head2d_pred(feat2_2d_proj_nonorm))  ## Prediction MLP

        # <feat2>
        
        # <Feat1_proj_pred, Feat2_proj> Ovlp Loss1
        feat2_2d_proj = F.normalize(feat2_2d_proj_nonorm)
        f1_aligned_1, f2_aligned_1 = self.invaug(feat1_2d_pred, coords[0], flags[0]), self.invaug(feat2_2d_proj, coords[1], flags[1])
        f1_aligned_1 = f1_aligned_1.view(nb_im, -1, patch_nb*patch_nb).transpose(1,2)  # bxHWxd
        f2_aligned_1 = f2_aligned_1.view(nb_im, -1, patch_nb*patch_nb).transpose(1,2)  # bxHWxd
        eigenval1_1, eigenvec1_1 = get_eigenvectors(f1_aligned_1, imgs=None, which_matrix=args.laplacian, threshold_tau=args.tau, train=True)
        eigenval2_1, eigenvec2_1 = get_eigenvectors(f2_aligned_1, imgs=None, which_matrix=args.laplacian, threshold_tau=args.tau, train=True)
        eigenvec1_1 = eigenvec1_1[:, :, 1] 
        eigenvec2_1 = eigenvec2_1[:, :, 1] 
        loss_ovlp1, map1_1, map2_1, batch_mask_1, loss_collapse1, loss_collapseV1, loss_uniform1, pseudo_mask1_1, pseudo_mask2_1 = self.eigen_loss(eigenvec1_1, eigenvec2_1, args, threshold)

        # <Feat1_proj, Feat2_proj_pred> Ovlp Loss2
        feat1_2d_proj = F.normalize(feat1_2d_proj_nonorm)
        f1_aligned_2, f2_aligned_2 = self.invaug(feat1_2d_proj, coords[0], flags[0]), self.invaug(feat2_2d_pred, coords[1], flags[1])
        f1_aligned_2 = f1_aligned_2.view(nb_im, -1, patch_nb*patch_nb).transpose(1,2)  # bxHWxd
        f2_aligned_2 = f2_aligned_2.view(nb_im, -1, patch_nb*patch_nb).transpose(1,2)  # bxHWxd
        eigenval1_2, eigenvec1_2 = get_eigenvectors(f1_aligned_2, imgs=None, which_matrix=args.laplacian, threshold_tau=args.tau, train=True)
        eigenval2_2, eigenvec2_2 = get_eigenvectors(f2_aligned_2, imgs=None, which_matrix=args.laplacian, threshold_tau=args.tau, train=True)
        eigenvec1_2 = eigenvec1_2[:, :, 1] 
        eigenvec2_2 = eigenvec2_2[:, :, 1] 
        loss_ovlp2, map1_2, map2_2, batch_mask_2, loss_collapse2, loss_collapseV2, loss_uniform2, pseudo_mask1_2, pseudo_mask2_2 = self.eigen_loss(eigenvec2_2, eigenvec1_2, args, threshold)
       
        if args is not None and args.local_rank != -1:
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        if itr%100==0 and rank==0:
            Inv = NormalizeInverse()
            x1_aligned, x2_aligned = self.invaug(x1, coords[0], flags[0]), self.invaug(x2, coords[1], flags[1])
            x1_1_aligned, x2_1_aligned = x1_aligned[batch_mask_1.bool()][0:4], x2_aligned[batch_mask_1.bool()][0:4]
            x1_2_aligned, x2_2_aligned = x1_aligned[batch_mask_2.bool()][0:4], x2_aligned[batch_mask_2.bool()][0:4]
            x1_1_crop_ori, x2_1_crop_ori = Inv(x1_1_aligned).clamp(0, 1)*255, Inv(x2_1_aligned).clamp(0, 1)*255 
            x1_2_crop_ori, x2_2_crop_ori = Inv(x1_2_aligned).clamp(0, 1)*255, Inv(x2_2_aligned).clamp(0, 1)*255 
            img_show1 = torch.zeros(8, 3, args.image_size, args.image_size).cuda()
            img_show2 = torch.zeros(8, 3, args.image_size, args.image_size).cuda()
            img_show1[0::2], img_show1[1::2] = x1_1_crop_ori, x2_1_crop_ori
            img_show2[0::2], img_show2[1::2] = x2_2_crop_ori, x1_2_crop_ori
            map_show1 = torch.zeros(8, 1, patch_nb, patch_nb).cuda()
            map_show2 = torch.zeros(8, 1, patch_nb, patch_nb).cuda()
            map_pseudo1 = torch.zeros(8, 1, patch_nb, patch_nb).cuda()
            map_pseudo2 = torch.zeros(8, 1, patch_nb, patch_nb).cuda()
            nb_im12 = map2_1.size(0)
            nb_im22 = map2_2.size(0)
            map_show1[0::2], map_show1[1::2] = map1_1.view(nb_im12,1,patch_nb,patch_nb)[0:4]*255, map2_1.view(nb_im12, 1, patch_nb, patch_nb)[0:4]*255
            map_show2[0::2], map_show2[1::2] = map1_2.view(nb_im22,1,patch_nb,patch_nb)[0:4]*255, map2_2.view(nb_im22, 1, patch_nb, patch_nb)[0:4]*255
            map_pseudo1[0::2], map_pseudo1[1::2] = pseudo_mask1_1.view(nb_im,1,patch_nb,patch_nb)[0:4]*255, pseudo_mask2_1.view(nb_im, 1, patch_nb, patch_nb)[0:4]*255
            map_pseudo2[0::2], map_pseudo2[1::2] = pseudo_mask1_2.view(nb_im,1,patch_nb,patch_nb)[0:4]*255, pseudo_mask2_2.view(nb_im, 1, patch_nb, patch_nb)[0:4]*255
            img_show = torch.cat([img_show1, img_show2], axis=-2) 
            map_show = torch.cat([map_show1, map_show2], axis=-2) 
            pseudo_show = torch.cat([map_pseudo1, map_pseudo2], axis=-2) 
            wandb.log({"img": convert_image(img_show), "map": convert_image(map_show), "pseudo": convert_image(pseudo_show)})

        if rank==0:
            wandb.log({"loss_ovlp1": loss_ovlp1, "loss_ovlp2": loss_ovlp2})
            wandb.log({"loss_collapse1": loss_collapse1, "loss_collapse2": loss_collapse2, "loss_collapseV1": loss_collapseV1, "loss_collapseV2": loss_collapseV2, "loss_uniform1": loss_uniform1, "loss_uniform2": loss_uniform2})
        ## Only consider the Loss Mask over Horizontal and Vertal Stripes. 
        w1, w2, w3 = weight, weight, weight
        loss_ovlp = loss_ovlp1+loss_ovlp2 - w1*(loss_collapse1+loss_collapse2) \
                                          - w2*(loss_collapseV1+loss_collapseV2) \
                                          - w3*(loss_uniform1 + loss_uniform2) \
    
        return loss_ovlp


