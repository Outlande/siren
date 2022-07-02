'''Test script for experiments in paper Sec. 4.2, Supplement Sec. 3, reconstruction from laplacian.
'''

# Enable import from parent package
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import modules, utils
import prob_sdf_meshing
import configargparse
import time

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=16384)
p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')
p.add_argument('--mode', type=str, default='mlp',
               help='Options are "mlp" or "nerf"')
p.add_argument('--resolution', type=int, default=1600)

opt = p.parse_args()
seq = ["", "_2nd", "_3rd", "_4th", "_5th"]
root_path = os.path.join(opt.logging_root, opt.experiment_name)
utils.cond_mkdir(root_path)

filename = os.path.join(root_path, 'test')
N=opt.resolution
num_samples = N ** 3
max_batch=64 ** 3
start = time.time()
ply_filename = filename

voxel_origin = [-1, -1, -1]
voxel_size = 2.0 / (N - 1)

overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
samples = torch.zeros(N ** 3, 4)

samples[:, 2] = overall_index % N
samples[:, 1] = (overall_index.long() / N) % N
samples[:, 0] = ((overall_index.long() / N) / N) % N

# transform first 3 columns
samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]
samples.requires_grad = False

for name in seq:
  checkpoint_path = opt.checkpoint_path + name + "/870/checkpoints/model_current.pth"

  class SDFDecoder(torch.nn.Module):
      def __init__(self):
          super().__init__()
          # Define the model.
          if opt.mode == 'mlp':
              self.model = modules.SingleBVPNet(type=opt.model_type, final_layer_factor=1, in_features=3)
          elif opt.mode == 'nerf':
              self.model = modules.SingleBVPNet(type='relu', mode='nerf', final_layer_factor=1, in_features=3)
          self.model.load_state_dict(torch.load(checkpoint_path))
          self.model.cuda()

      def forward(self, coords):
          model_in = {'coords': coords}
          return self.model(model_in)['model_out']

  decoder = SDFDecoder()
  decoder.eval()
  head = 0
  while head < num_samples:
      # print(head)
      sample_subset = samples[head: min(head + max_batch, num_samples), 0:3].cuda()

      samples[head: min(head + max_batch, num_samples), 3] = (
          decoder(sample_subset)
              .squeeze()  # .squeeze(1)
              .detach()
              .cpu()
      )
      head += max_batch

  sdf_values = samples[:, 3:4]
  print(seq.index(name))
  if seq.index(name) == 0:
      values = sdf_values
  else:
      print(values.size())
      values = torch.cat((values, sdf_values), 1)
      if (seq.index(name) == len(seq) - 1):
          print("meshing")
          print(values.size())
          mean = torch.mean(values, 1)
          print(mean.shape)
          var = torch.var(values, 1)
          mean = mean.reshape(N, N, N)
          var = var.reshape(N, N, N)
          end = time.time()
          print("sampling takes: %f" % (end - start))
          prob_sdf_meshing.convert_sdf_samples_to_ply(
              mean.data.cpu(),
              var.data.cpu(),
              voxel_origin,
              voxel_size,
              ply_filename + ".ply",
          )



