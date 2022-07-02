'''From the DeepSDF repository https://github.com/facebookresearch/DeepSDF
'''
#!/usr/bin/env python3

import logging
import numpy as np
import plyfile
import skimage.measure
import time
import torch

def val2rgb(v):
  v = 1./np.log(v)
  v_min = -0.15#np.min(v)
  v_max = np.max(v)

  v[v < v_min] = v_min
  v[v > v_max] = v_max
  dv = v_max - v_min

  #print(v)
  print(np.min(v), np.max(v))
  r = np.ones(np.shape(v))
  g = np.ones(np.shape(v))
  b = np.zeros(np.shape(v))
  #print(np.shape(r), np.shape(g), np.shape(b), np.shape(v))
  r[v <= (v_min + 0.25 * dv)] = 0
  g[v <= (v_min + 0.25 * dv)] = 4 * (v[v <= (v_min + 0.25 * dv)] - v_min) / dv
  r[np.logical_and(v <= (v_min + 0.5*dv), v > (v_min + 0.25 * dv))] = 0
  b[np.logical_and(v <= (v_min + 0.5*dv), v > (v_min + 0.25 * dv))] = 1 + 4 * (v_min + 0.25 * dv - v[np.logical_and(v<=(v_min + 0.5*dv), v > (v_min + 0.25 * dv))]) / dv
  r[np.logical_and(v <= (v_min + 0.75 * dv), v>(v_min + 0.5*dv))] = 4 * (v[np.logical_and(v <= (v_min + 0.75 * dv), v>(v_min + 0.5*dv))] - v_min - 0.5 * dv) / dv
  b[np.logical_and(v <= (v_min + 0.75 * dv), v>(v_min + 0.5*dv))] = 0
  g[v > (v_min + 0.75 * dv)] = 1 + 4 * (v_min + 0.75 * dv - v[v > (v_min + 0.75 * dv)]) / dv
  return r,g,b


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    pytorch_sigma_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()
    numpy_sigma_tensor = pytorch_sigma_tensor.numpy()
    r,g,b = val2rgb(numpy_sigma_tensor)

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
            numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3#, mask=numpy_sigma_tensor**2 < 0.1
        )
    except:
        pass

    verts_ind = np.floor(verts / voxel_size).astype(int)
    color_r = r[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
    color_g = g[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
    color_b = b[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]


    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    #verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    for i in range(0, num_verts):
        #verts_tuple[i] = tuple(mesh_points[i, :])
        verts_tuple[i] = tuple([mesh_points[i, 0], mesh_points[i, 1], mesh_points[i, 2], 255*color_r[i], 255*color_g[i], 255*color_b[i]])
        if (np.logical_or(np.logical_or(np.logical_or(color_r[i]>1, color_r[i]<0),np.logical_or(color_b[i]>1, color_b[i]<0)), np.logical_or(color_g[i]>1, color_g[i]<0))):
           print(255*color_r[i], 255*color_g[i], 255*color_b[i])



    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )
