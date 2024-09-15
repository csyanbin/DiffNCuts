# Train
CUDA_VISIBLE_DEVICES=3 python eigen_pred_iou.py --batch-size-pergpu 256 --base_lr 0.0005 --loss_eigtype lossall --loss_crit BCE --epoch_weight 4 --epochs 10 --save_epoch 1 --warm_up 1 --weight_decay 1e-4

# Evaluate Unsupervised Saliency Detection
python eval_det.py --ckpt_dir CKPT_DIR_PATH  --arch vit_small --cpu 41-56 --gpu 2 --tau 0.2 --epoch 2 --eigen_train 1 

