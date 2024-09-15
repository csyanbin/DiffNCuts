import os
from PIL import Image
import torchvision.transforms as tvt
import torch
import torch.nn.functional as F
import cv2
import util.vision_transformer as vits
import argparse
from util.eigen_decomp import EigenDecompositionFcnFast
from datetime import datetime
import torchvision
import bilateral_solver
import numpy as np
import logging
from measure import evaluate_function
import measure

from torchvision.models.resnet import resnet50
import network.vision_transformer as vits
from torch import nn

import os
num_numpy_threads = '8'
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = num_numpy_threads
os.environ['GOTO_NUM_THREADS'] = num_numpy_threads
os.environ['MKL_NUM_THREADS'] = num_numpy_threads
os.environ['NUMEXPR_NUM_THREADS'] = num_numpy_threads
os.environ['OMP_NUM_THREADS'] = num_numpy_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_numpy_threads 
os.environ["KMP_BLOCKTIME"] = "0"

image_trans = tvt.Compose([
    tvt.ToTensor(),
    tvt.Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
])

class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())

class ResNet50Bottom(nn.Module):
    # https://forums.fast.ai/t/pytorch-best-way-to-get-at-intermediate-layers-in-vgg-and-resnet/5707/2
    def __init__(self, original_model):
        super(ResNet50Bottom, self).__init__()
        # Remove avgpool and fc layers
        self.features = nn.Sequential(*list(original_model.children())[:-2])

    def forward(self, x):
        x = self.features(x)
        return x

def get_model(arch, patch_size, resnet_dilate=2):

    # Initialize model with pretraining
    url = None
    if "resnet" in arch:
        if resnet_dilate == 1:
            replace_stride_with_dilation = [False, False, False]
        elif resnet_dilate == 2:
            replace_stride_with_dilation = [False, False, True]
        elif resnet_dilate == 4:
            replace_stride_with_dilation = [False, True, True]

        if "imagenet" in arch:
            model = resnet50(
                pretrained=True,
                replace_stride_with_dilation=replace_stride_with_dilation,
            )
        else:
            model = resnet50(
                pretrained=False,
                replace_stride_with_dilation=replace_stride_with_dilation,
            )
        if "resnet50" in arch:
            url = "dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
            
    elif "moco" in arch:
        if arch == "moco_vit_small" and patch_size == 16:
            url = "moco-v3/vit-s-300ep/vit-s-300ep.pth.tar"
        elif arch == "moco_vit_base" and patch_size == 16:
            url = "moco-v3/vit-b-300ep/vit-b-300ep.pth.tar"
        model = vits.__dict__[arch](num_classes=0)
        
    elif "mae" in arch:
        if arch == "mae_vit_base" and patch_size == 16:
            url = "mae/visualize/mae_visualize_vit_base.pth"
        model = vits.__dict__[arch](num_classes=0)
        
    elif "vit" in arch:
        if arch == "vit_small" and patch_size == 16:
            url = "dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif arch == "vit_small" and patch_size == 8:
            url = "dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth" 
        elif arch == "vit_base" and patch_size == 16:
            url = "dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif arch == "vit_base" and patch_size == 8:
            url = "dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        elif arch == "resnet50":
            url = "dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
        model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
        
    else:
        raise NotImplementedError 

    # for p in model.parameters():
    #     p.requires_grad = False

    if url is not None:
        print(
            "Since no pretrained weights have been provided, we load the reference pretrained DINO weights."
        )
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/" + url
        )
        if "moco" in arch:
            state_dict = state_dict['state_dict']
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.head'):
                    # remove prefix
                    state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
        elif "mae" in arch:
            state_dict = state_dict['model']
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('decoder') or k.startswith('mask_token'):
                    # remove prefix
                    #state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]
                    
        strict_loading = False if "resnet" in arch else True
        msg = model.load_state_dict(state_dict, strict=strict_loading)
        print(
            "Pretrained weights found at {} and loaded with msg: {}".format(
                url, msg
            )
        )
    else:
        print(
            "There is no reference weights available for this model => We use random weights."
        )

    if "resnet" in arch:
        model = ResNet50Bottom(model)
        
    return model

  
def get_eigenvectors(feats, imgs=None, which_matrix: str = 'laplacian', K=2, threshold_tau=0.0, train=True):
         
    # Eigenvectors of laplacian matrix
    A = torch.bmm(feats, feats.transpose(1,2)) 
    if threshold_tau<0.0:                  # Binary Graph Construction 
        tau = -threshold_tau
        _W_semantic = (A>tau).float()
        _W_semantic += 1e-6 * (A<=tau)     # Avoid ill-conditioned and multiple same eigenvalue
    elif threshold_tau>=0.0:
        tau = threshold_tau
        _W_semantic = (A * (A>tau) )
        _W_semantic += 1e-6 * (A<=tau)     # Avoid ill-conditioned and multiple same eigenvalue
    #_W_semantic = _W_semantic / _W_semantic.max()    # NOTE: If features are normalized, this naturally does nothing
    
    if which_matrix == 'ours':
        diag = _W_semantic.sum(2)         # row sum
        diag[diag < 1e-8] = 1.0          # prevent from dividing by zero
        D = torch.diag_embed(diag)        # diagonal
        Laplacian = D - _W_semantic
    
        eigenvalues, eigenvectors = EigenDecompositionFcnFast.apply(Laplacian.double(), K)

    # Sign ambiguity (assume foreground is smaller than background areas at test time)
    batch_mask = torch.ones(eigenvectors.size(0), 1, eigenvectors.size(-1), device=feats.device)
    if not train:
        for b in range(eigenvectors.size(0)):
            for k in range(K):
                if 0.5 < torch.mean((eigenvectors[b, :, k] > 0).float()).item() < 1.0:  # reverse segment
                    batch_mask[b, 0, k] = -1.0
    eigenvectors =  batch_mask * eigenvectors

    return eigenvalues.float(), eigenvectors.float()

    
parser = argparse.ArgumentParser(description='Eigenvector Feature Maps')
parser.add_argument('--patch_size', type=int, default=16, choices=[8, 16])
parser.add_argument('--arch', type=str, default="vit_small")
parser.add_argument('--pretr_path', type=str, default="")
parser.add_argument('--test_dataset', type=str, default="DUTS", choices=["ECSSD", "DUTS", "DUT-OMRON", "CUB"])
parser.add_argument('--test_root', type=str, default="./datasets")
parser.add_argument('--save_name', type=str, default="Eigen")
parser.add_argument('--postproc', type=str, default="None", choices=["None", "CRF", "BS"])
parser.add_argument('--which_matrix', type=str, default="ours")
parser.add_argument('--tau', type=float, default=0.0)

args = parser.parse_args()
patch_size = args.patch_size

#save_path = os.path.join(args.test_root, args.test_dataset, "IMG/")
if args.test_dataset == "DUTS":
    save_path = os.path.join(args.test_root, "DUTS_Test/img/")
elif args.test_dataset == "CUB":
    save_path = os.path.join(args.test_root, "CUB/test_images/")
elif args.test_dataset == "ECSSD":
    save_path = os.path.join(args.test_root, "ECSSD/img/")
elif args.test_dataset == "DUT-OMRON":
    save_path = os.path.join(args.test_root, "DUT_OMRON/img/")

save_mask = os.path.join('save_masks/', args.test_dataset, args.save_name, args.postproc)
save_mask += datetime.now().strftime("_%Y-%m-%d-%H-%M-%S")
os.makedirs(save_mask, exist_ok=True)

## Load model and pre-trained weights
teacher = get_model(args.arch, args.patch_size)
teacher = teacher.cuda()

if len(args.pretr_path)>2:
    state_dict = torch.load(args.pretr_path, map_location="cpu")
    logging.info(args.pretr_path)
    print("Use eigen trained ckpt:", args.pretr_path)

    if args.pretr_path.startswith("dino"): 
        weights_dict = state_dict
    else:
        weights_dict = {k.replace('module.', '').replace('net.', ''): v for k, v in state_dict['model'].items()}
    teacher.load_state_dict(weights_dict, strict=False)
else:
    print('evaluate pre-loaded ckpt')

teacher.eval()

data_list_all = [save_path+f for f in sorted(os.listdir(os.path.join(save_path)))]
print(f"the image number of dataset is {len(data_list_all)}")
logging.info(f"the image number of dataset is {len(data_list_all)}")
id = 0
while id < len(data_list_all):
    #if id%1000==0:
    #    print(f"The {id}-th image in the Dataset")
    data_list = [data_list_all[id].split('/')[-1]]

    imgs = []
    for name in data_list:
        #img = image_trans(Image.open(os.path.join(save_path, str(id), name)).convert('RGB'))
        img = image_trans(Image.open(os.path.join(save_path, name)).convert('RGB'))
        imgs.append(img.unsqueeze(0))

    imgs = torch.cat(imgs).cuda()

    # <resize the image>
    w, h = imgs.shape[2], imgs.shape[3]
    new_w, new_h = int(round(w / patch_size)) * patch_size, int(round(h / patch_size)) * patch_size
    w_featmap, h_featmap = new_w // patch_size, new_h // patch_size
    imgs1 = F.interpolate(imgs, size=(new_w, new_h), mode='bilinear', align_corners=False)
    # <resize the image>

    ## <get feature>
    if 'resnet' not in args.arch:
        feat_out = {}
        def hook_fn_forward_qkv(module, input, output):
            feat_out["qkv"] = output
        teacher._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)

        attentions = teacher.get_last_selfattention(imgs1)
        attentions = attentions.detach()

        # Dimensions
        nb_im = attentions.shape[0]  # Batch size
        nh = attentions.shape[1]  # Number of heads
        nb_tokens = attentions.shape[2]  # Number of tokens

        qkv = (
            feat_out["qkv"]
            .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
            .permute(2, 0, 3, 1, 4)
        ).detach()
        q, k, v = qkv[0], qkv[1], qkv[2]
        k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
        q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
        v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)

        features = k[:, 1:, :]
    else:
        feat1_2d = teacher.forward(imgs1)
        nb_im, d, patch_nb, _ = feat1_2d.shape
        features = feat1_2d.view(nb_im, d, -1).transpose(2,1)
        ## Apply layernorm
        layernorm = torch.nn.LayerNorm(features.size()[1:]).to(imgs1.device)
        features = layernorm(features)
    
    ## <get feature>
    normalize = True
    if normalize:
        feats = F.normalize(features, p=2, dim=-1)
            
    # Get eigenvectors, BxHWxK
    which_matrix = args.which_matrix
    train = False
    eigenvalues, eigenvectors = get_eigenvectors(feats, img, which_matrix, train=train, threshold_tau=args.tau)
    eig_index = 1 # second-smallest eigenvector as heatmap
    eigenvectors = eigenvectors[:, :, eig_index]
    pred_mask = (eigenvectors>0.0).to(torch.float)
    pred_mask = pred_mask.reshape(nb_im, 1, w_featmap, h_featmap)
    
    if args.postproc == "None": 
        pred_mask = F.interpolate(pred_mask, size=(w, h), mode='bilinear', align_corners=False) * 255.
        project_map = pred_mask.detach().cpu()

    elif args.postproc == "BS": 
        Inv = NormalizeInverse(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        imgOri = Inv(imgs).clamp(0, 1)*255
        imgNP = imgOri.squeeze().permute(1,2,0).cpu().numpy()
            
        pred_mask = F.interpolate(pred_mask, size=(w, h), mode='bilinear', align_corners=False)
        pred_mask = (pred_mask>0.5)
        pred_mask = pred_mask.squeeze().cpu().numpy()
        sigma_spatial = 16
        sigma_luma = 16
        sigma_chroma = 8
        output_solver, binary_solver = bilateral_solver.bilateral_solver_output(imgNP, pred_mask, sigma_spatial, sigma_luma, sigma_chroma)
        pred_mask = torch.tensor(output_solver)
        pred_mask = pred_mask.unsqueeze(0)
        project_map = pred_mask * 255. 
                    

    save_imgs = []

    for i, name in enumerate(data_list):
        mask = project_map[i].repeat(3, 1, 1).permute(1, 2, 0).detach().numpy() # [:img.shape[0], :img.shape[1],:]
        bi_mask = mask

        name = name.split('/')[-1]
        cv2.imwrite(os.path.join(save_mask, name.replace('jpg', 'png')), bi_mask)

    id += 1



gt_dict = {'DUTS': './datasets/DUTS_Test/gt', 
           'CUB': 'dataset/CUB/test_segmentations', 
           'ECSSD': './datasets/ECSSD/gt', 
           'DUT-OMRON': './datasets/DUT_OMRON/gt'} 
gt_dir = gt_dict[args.test_dataset]
pred_dir = save_mask
evaluate_function(args.test_dataset, pred_dir, gt_dir)




