#!/bin/bash
dataset=ICL2_clean_consistent_normal
ori_dataset=/home/yan/Dataset/ICL/living2/clean
output=ICL2_finetune_seq_from1500

for ((i=1;i<=8;i++));
do
    frame=$[100*$i]
    echo "frame-${frame}"
    python experiment_scripts/train_sdf.py --model_type=sine --point_cloud_path=/media/yan/Passport/${dataset}/${frame}.xyz --batch_size=170000 --checkpoint_path=logs/network/${output}/10/checkpoints/model_current.pth\
    --camera_pose_path=${ori_dataset}/traj0.gt.freiburg --camera_number=${frame} --camera_depth_path=${ori_dataset}/depth/${frame}.png --camera_intrinsic=${ori_dataset}/ICL_gt.yaml\
    --experiment_name=network/${output}/${frame} --epochs_til_ckpt=100 --num_epochs=1500 --train_mode=1
    python experiment_scripts/test_sdf.py --checkpoint_path=logs/network/${output}/${frame}/checkpoints/model_final.pth --experiment_name=model/${output}/${frame}/ --resolution=512
done
