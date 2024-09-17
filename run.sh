# Train
CUDA_VISIBLE_DEVICES=0 python eigen_pred_iou.py --batch-size-pergpu 256 --base_lr 0.0005 --loss_eigtype lossall --loss_crit BCE --max_thres 0.5 --min_thres 0.5 --max_weight 0.05 --min_weight 0.05 --epoch_weight 4 --trainset imagenet --epochs 10 --save_epoch 1 --warm_up 1 --weight_decay 5e-4
# Evaluate Unsupervised Saliency Detection
python eval_seg.py --ckpt_dir CKPT_DIR_PATH  --arch vit_small --cpu 1-16 --gpu 0 --tau 0.0 --epoch 2 --eigen_train 1 

# Ealuate Unsupervised Object Discovery
cd TokenCut
python eval_det.py --ckpt_dir CKPT_DIR_PATH  --arch vit_small --cpu 1-16 --gpu 0 --tau 0.2 --epoch 2 --eigen_train 1
