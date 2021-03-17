import torch
import dataio, modules
import utils
from utils import make_contour_plot
from torch.utils.tensorboard import SummaryWriter
import configargparse


def write_sdf_summary(model, writer, total_steps, prefix='train_'):
    slice_coords_2d = dataio.get_mgrid(512)
    with torch.no_grad():
        yz_slice_coords = torch.cat((-0.1*torch.ones_like(slice_coords_2d[:, :1]), slice_coords_2d), dim=-1)
        yz_slice_model_input = {'coords': yz_slice_coords.cuda()[None, ...]}

        yz_model_out = model(yz_slice_model_input)
        sdf_values = yz_model_out['model_out']
        sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
        fig = make_contour_plot(sdf_values)
        writer.add_figure(prefix + 'yz_sdf_slice', fig, global_step=total_steps)

        xz_slice_coords = torch.cat((slice_coords_2d[:,:1],
                                        0.1*torch.ones_like(slice_coords_2d[:, :1]),
                                        slice_coords_2d[:,-1:]), dim=-1)
        xz_slice_model_input = {'coords': xz_slice_coords.cuda()[None, ...]}

        xz_model_out = model(xz_slice_model_input)
        sdf_values = xz_model_out['model_out']
        sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
        fig = make_contour_plot(sdf_values)
        writer.add_figure(prefix + 'xz_sdf_slice', fig, global_step=total_steps)

        xy_slice_coords = torch.cat((slice_coords_2d[:,:2],
                                     0.1*torch.ones_like(slice_coords_2d[:, :1])), dim=-1)
        xy_slice_model_input = {'coords': xy_slice_coords.cuda()[None, ...]}

        xy_model_out = model(xy_slice_model_input)
        sdf_values = xy_model_out['model_out']
        sdf_values = dataio.lin2img(sdf_values).squeeze().cpu().numpy()
        fig = make_contour_plot(sdf_values)
        writer.add_figure(prefix + 'xy_sdf_slice', fig, global_step=total_steps)
        writer.add_figure(prefix + 'slice', fig, global_step=total_steps)


def main():
    p = configargparse.ArgumentParser()
    p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
    p.add_argument('--summary_dir', default=None, help='save summary dir.')
    opt = p.parse_args()

    model = modules.SingleBVPNet(type="sine", in_features=3)
    model.load_state_dict(torch.load(opt.checkpoint_path))
    print('load checkpoint: ', opt.checkpoint_path)
    model.cuda()

    utils.cond_mkdir(opt.summary_dir)
    writer = SummaryWriter(opt.summary_dir)
    write_sdf_summary(model, writer, 300)


if __name__ == "__main__":
    main()