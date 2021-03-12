#/bin/bash

python experiment_scripts/train_sdf.py --model_type=sine --point_cloud_path=../datasets/ICL2_clean/10.xyz --batch_size=170000 \
    --experiment_name=network/test_camera/ --epochs_til_ckpt=100 --camera_pose_path=../datasets/ICL2_clean_RGBD/traj0.gt.freiburg \
    --camera_number=10 --camera_depth_path=../datasets/ICL2_clean_RGBD/depth/10.png --camera_intrinsic=../datasets/ICL2_clean_RGBD/ICL_gt.yaml


python experiment_scripts/test_sdf.py --checkpoint_path=logs/network/test_code_1/checkpoints/model_current.pth --experiment_name=test_result --resolution=512
