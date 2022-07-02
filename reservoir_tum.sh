#!/bin/bash
dataset=TUM_consistent_normal
subset=rgbd_dataset_freiburg1_desk
subset_id=1
ori_dataset=/home/yan/Dataset/TUM
output=TUM${subset_id}_reservoir

python experiment_scripts/train_TUM_sdf.py --model_type=sine --point_cloud_path=/media/yan/Passport/${dataset}/${subset}/10.xyz --batch_size=170000 \
    --experiment_name=network/${output}/10 --epochs_til_ckpt=100 --camera_pose_path=${ori_dataset}/${subset}/TUM_${subset_id}_sync.txt \
    --camera_number=10 --camera_depth_path=${ori_dataset}/${subset}/ --camera_intrinsic=${ori_dataset}/${subset}/TUM${subset_id}.yaml --num_epochs=10000
python experiment_scripts/test_sdf.py --checkpoint_path=logs/network/${output}/10/checkpoints/model_current.pth --experiment_name=model/${output}/10/ --resolution=512

for ((i=1;i<=56;i++));
do
    frame=$[10*$i]
    next=$[$frame+10]
    echo "frame-${next}"
    python experiment_scripts/train_TUM_sdf.py --model_type=sine --point_cloud_path=/media/yan/Passport/${dataset}/${subset}/${next}.xyz --batch_size=170000 --checkpoint_path=logs/network/${output}/${frame}/checkpoints/model_current.pth\
    --camera_pose_path=${ori_dataset}/${subset}/TUM_${subset_id}_sync.txt --camera_number=${next} --camera_depth_path=${ori_dataset}/${subset}/ --camera_intrinsic=${ori_dataset}/${subset}/TUM${subset_id}.yaml\
    --experiment_name=network/${output}/${next} --epochs_til_ckpt=100 --num_epochs=1500 --train_mode=2 --reservoir_path=/media/yan/Passport/${dataset}_sampled_buffer/${subset}/${frame}.xyz
    python experiment_scripts/test_sdf.py --checkpoint_path=logs/network/${output}/${next}/checkpoints/model_current.pth --experiment_name=model/${output}/${next}/ --resolution=512
done
