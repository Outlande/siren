'''Reproduces Sec. 4.2 in main paper and Sec. 4 in Supplement.
'''

# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, meta_modules, utils, training, loss_functions, modules
import torch
from torch.utils.data import DataLoader
import configargparse
import numpy as np
import yaml

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=1400)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=5e-5')
p.add_argument('--num_epochs', type=int, default=10000,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=1,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')
p.add_argument('--point_cloud_path', type=str, default='/home/sitzmann/data/point_cloud.xyz',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')

p.add_argument('--camera_pose_path', type=str, default='/home/sitzmann/data/trajctory.txt',
               help='camera pose in tum format')
p.add_argument('--camera_number', type=int, default=10,
               help='camera number to load')
p.add_argument('--camera_depth_path', type=str, default='/home/sitzmann/data/depth',
               help='camera depth image in')
p.add_argument('--camera_intrinsic', type=str, default='/home/sitzmann/data/intrinsic.txt',
               help='camera pose in tum format')
p.add_argument('--reservoir_path', type=str, default='/home/sitzmann/data/point_cloud.xyz',
               help='reservoir path')
p.add_argument('--train_mode', type=int, default=0, help='0:batch 1:finetune 2:reservoir.')

#p.add_argument('--last_checkpoint', default=None, help='Checkpoint to finetune model.')

opt = p.parse_args()

# load camera intrinsic
with open(opt.camera_intrinsic) as f:
    intrinsic = np.array(yaml.load(f.read(), yaml.SafeLoader)['K']).reshape(3,3)

# load camera pose
pose = np.genfromtxt(opt.camera_pose_path)[opt.camera_number//10 - 1,1:]
print(pose)

file=open(opt.camera_pose_path)
data=file.read()
lines = data.split("\n")
list = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
depth_path = opt.camera_depth_path + list[opt.camera_number//10 - 1][0]
#print(depth_path)

if opt.train_mode == 2:
    sdf_dataset = dataio.Reservoir(opt.point_cloud_path, opt.reservoir_path, on_surface_points=opt.batch_size, intrinsic=intrinsic,
                                    pose=pose, camera_depth_path=depth_path, last_checkpoint=opt.checkpoint_path)
else:
    sdf_dataset = dataio.PointCloud(opt.point_cloud_path, on_surface_points=opt.batch_size, intrinsic=intrinsic,
                                    pose=pose, camera_depth_path=depth_path)

dataloader = DataLoader(sdf_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)

# Define the model.
if opt.model_type == 'nerf':
    model = modules.SingleBVPNet(type='relu', mode='nerf', in_features=3)
else:
    model = modules.SingleBVPNet(type=opt.model_type, in_features=3)
if opt.train_mode != 0:
    model.load_state_dict(torch.load(opt.checkpoint_path))
    print('load checkpoint: ', opt.checkpoint_path)
model.cuda()

# Define the loss
loss_fn = loss_functions.sdf
summary_fn = utils.write_sdf_summary

root_path = os.path.join(opt.logging_root, opt.experiment_name)

training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, double_precision=False,
               clip_grad=True)