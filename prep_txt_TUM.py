import numpy as np
import open3d as o3d
import os
import os.path as osp
from transforms3d import quaternions
import math
from tqdm.autonotebook import tqdm

def tum_test_dict():
    """ the trajectorys held out for testing TUM dataset
    """
    return  {
        'fr1': {
            'calib': [517.3, 516.5, 318.6, 255.3],
            'seq': ['rgbd_dataset_freiburg1_desk']
        },

        'fr2': {
            'calib': [520.9, 521.0, 325.1, 249.7],
            'seq': ['rgbd_dataset_freiburg2_desk']
        },

        'fr3': {
            'calib': [535.4, 539.2, 320.1, 247.6],
            'seq': ['rgbd_dataset_freiburg3_long_office_household']
        }
    }


class TUM():
    def __init__(self, root='/home/yan/Dataset', category='test',
                 keyframes=[1], data_transform=None, select_traj=None):
        """
        :param the root directory of the data
        :param select the category (train, validation,test)
        :param select the number of keyframes
            Test data only support one keyfame at one time
            Train/validation data support mixing different keyframes
        :param select one particular trajectory at runtime
            Only support for testing
        """
        super(TUM, self).__init__()

        self.image_seq = []
        self.depth_seq = []
        self.invalid_seq = []
        self.cam_pose_seq = []
        self.calib = []
        self.seq_names = []

        self.ids = 0
        self.seq_acc_ids = [0]
        self.keyframes = keyframes

        self.transforms = data_transform

        if category == 'test':
            self.__load_test(root + '/TUM', select_traj)
        else:  # train and validation
            self.__load_train_val(root + '/TUM', category)

        # downscale the input image to a quarter
        self.fx_s = 0.25
        self.fy_s = 0.25


    def __load_test(self, root, select_traj=None):

        tum_data = tum_test_dict()

        assert (len(self.keyframes) == 1)
        kf = self.keyframes[0]
        self.keyframes = [1]

        for ks, scene in tum_data.items():
            for seq_name in scene['seq']:
                seq_path = osp.join(ks, seq_name)

                if select_traj is not None:
                    if seq_name != select_traj: continue

                self.calib.append(scene['calib'])

                def write_sync_trajectory(self, local_dir, dataset, subject_name):
                    """
                    :param the root of the directory
                    :param the dataset category 'fr1', 'fr2' or 'fr3'
                    """
                    rgb_file = osp.join(local_dir, subject_name, 'rgb.txt')
                    depth_file = osp.join(local_dir, subject_name, 'depth_ori.txt')
                    pose_file = osp.join(local_dir, subject_name, 'groundtruth.txt')
                    #print(rgb_file, depth_file, pose_file)

                    rgb_list = read_file_list(rgb_file)
                    depth_list = read_file_list(depth_file)
                    pose_list = read_file_list(pose_file)

                    matches = associate_three(rgb_list, depth_list, pose_list, offset=0.0, max_difference=0.02)

                    trajectory_info = []
                    frame_count = 0
                    for (a, b, c) in matches:
                        pose = [float(x) for x in pose_list[c]]
                        rgb_file = osp.join(local_dir, subject_name, rgb_list[a][0])
                        depth_file = depth_list[b][0]#osp.join(local_dir, subject_name, depth_list[b][0])
                        self.image_seq.append(rgb_file)
                        self.depth_seq.append(depth_file)
                        self.cam_pose_seq.append(pose)

                write_sync_trajectory(self, root, ks, seq_name)

        if len(self.image_seq) == 0:
            raise Exception("The specified trajectory is not in the test set.")

def associate_three(first_list, second_list, third_list, offset, max_difference):
    first_keys = list(first_list)
    second_keys = list(second_list)
    third_keys = list(third_list)
    # find the potential matches in (rgb, depth)
    potential_matches_ab = [(abs(a - (b + offset)), a, b)
                            for a in first_keys
                            for b in second_keys
                            if abs(a - (b + offset)) < max_difference]
    potential_matches_ab.sort()
    matches_ab = []
    for diff, a, b in potential_matches_ab:
        if a in first_keys and b in second_keys:
            matches_ab.append((a, b))

    matches_ab.sort()

    # find the potential matches in (rgb, depth, pose)
    potential_matches = [(abs(a - (c + offset)), abs(b - (c + offset)), a,b,c)
                        for (a,b) in matches_ab
                        for c in third_keys
                        if abs(a - (c + offset)) < max_difference and
                        abs(b - (c + offset)) < max_difference]

    potential_matches.sort()
    matches_abc = []
    for diff_rgb, diff_depth, a, b, c in potential_matches:
        if a in first_keys and b in second_keys and c in third_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            third_keys.remove(c)
            matches_abc.append((a,b,c))
    matches_abc.sort()
    return matches_abc


def read_file_list(filename):

    file = open(filename)
    data = file.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n")
    list = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
    list = [(float(l[0]),l[1:]) for l in list if len(l)>1]
    return dict(list)


if __name__ == "__main__":
    subsample = 30
    subset = 'rgbd_dataset_freiburg2_desk'
    loader = TUM(category='test', root='/home/yan/Dataset', select_traj=subset)
    f = open("TUM_2_sync.txt", "w")
    #print(len(loader.image_seq), len(loader.depth_seq), len(loader.cam_pose_seq))
    geometry = o3d.geometry.PointCloud()
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    view_control = visualizer.get_view_control()
    cam = view_control.convert_to_pinhole_camera_parameters()
    output_dir = '/media/yan/Passport/TUM2_consistent_normal_buffer/' + subset #+ '_buffer'
    coordinate = o3d.geometry.PointCloud()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for sample in range(1, len(loader.image_seq) // subsample):
        frame = sample * subsample
        color_name = loader.image_seq[frame]
        depth_name = loader.depth_seq[frame]
        pose = loader.cam_pose_seq[frame]

        np.savetxt(f, np.ravel(np.array(depth_name + " " + " ".join(str(x) for x in pose))), fmt='%s', newline=' ', delimiter=',')
        f.write("\n")
