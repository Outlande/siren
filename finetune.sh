#!/bin/bash
output=ICL2_finetune
dataset=ICL2_clean_consistent_normal
ori_dataset=/home/yan/Dataset/ICL/living2/clean

for ((i=2;i<=86;i++));
do
    frame=$[10*$i]
    next=$[$frame+10]
    echo "frame-${next}"
    python experiment_scripts/train_sdf.py --model_type=sine --point_cloud_path=/media/yan/Passport/${dataset}/${next}.xyz --batch_size=170000 \
    --experiment_name=network/${output}/${next} --epochs_til_ckpt=100 --checkpoint_path=logs/network/${output}/${frame}/checkpoints/model_current.pth --camera_pose_path=${ori_dataset}/traj0.gt.freiburg \
    --camera_number=${next} --camera_depth_path=${ori_dataset}/depth/${next}.png --camera_intrinsic=${ori_dataset}/ICL_gt.yaml --num_epochs=1500 --train_mode=1

    python experiment_scripts/test_sdf.py --checkpoint_path=logs/network/${output}/${next}/checkpoints/model_current.pth --experiment_name=model/${output}/${next}/ --resolution=512
done
