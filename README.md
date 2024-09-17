# Unsupervised Dense Prediction using Differentiable Normalized Cuts
This is the implementation of our ECCV 2024 paper "Unsupervised Dense Prediction using Differentiable Normalized Cuts" by Yanbin Liu and Stephen Gould. 

## 1. Dependencies
This code was implemented with Python 3.10, PyTorch 1.13 and CUDA 11.6.   
```
conda create -n DiffNCuts python=3.10
conda activate DiffNCuts
pip install -r requirements.txt
```

## 2. Data
Please follow [DOWNLOAD_DATA.md](datasets/DOWNLOAD_DATA.md)

## 3. Training

### DINO pretrained checkpoints

We initialize the model weights by using dino pretrained weights. To download the dino model, please launch the following commands:
```
wget https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth # vit-s/16
wget https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth # vit-s/8
wget https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth # vit-b/16
```
```
CUDA_VISIBLE_DEVICES=0 python eigen_pred_iou.py --batch-size-pergpu 256 --base_lr 0.0005 --loss_eigtype lossall --loss_crit BCE --max_thres 0.5 --min_thres 0.5 --max_weight 0.05 --min_weight 0.05 --epoch_weight 4 --trainset imagenet --epochs 10 --save_epoch 1 --warm_up 1 --weight_decay 5e-4
```
Checkpoint can be downloaded from [Here](https://drive.google.com/file/d/1hAPR0gxBHgIOSATtJNSStaf5aBDnNzYd/view?usp=sharing)

## 4. Evaluating Unsupervised Saliency Detection
```
python eval_seg.py --ckpt_dir CKPT_DIR_PATH  --arch vit_small --cpu 1-8 --gpu 2 --tau 0.0 --epoch 2 --eigen_train 1 
```

## 5. Evaluating Unsupervised Object Discovery
Adapted from [TokenCut](https://github.com/YangtaoWANG95/TokenCut)
Go to our ./TokenCut Folder and run
```
cd TokenCut
python eval_det.py --ckpt_dir CKPT_DIR_PATH  --arch vit_small --cpu 1-8 --gpu 2 --tau 0.2 --epoch 1 --eigen_train 1
```


### Bibtex
If you use this code or results for your research, please consider citing:
````
@inproceedings{Liu:ECCV2024,
  author    = {Yanbin Liu and
               Stephen Gould},
  title     = {Unsupervised Dense Prediction using Differentiable Normalized Cuts},
  booktitle = {ECCV},
  year      = {2024}
}
````
