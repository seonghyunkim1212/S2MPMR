import numpy as np
import torch
import torch.utils.data as data
import cv2
import math
import scipy.io as sio
import ref
from utils.utils import rnd, flip, shuffle_lr
from utils.img import transform, crop

#  0: R ankle
#  1: R knee
#  2: R hip
#  3: L hip
#  4: L knee
#  5: L ankle
#  6: R Wrist
#  7: R Elbow
#  8: R shoulder
#  9: L shoulder
# 10: L Elbow
# 11: L Wrist
# 12: Neck
# 13: Head-Top


# Set the image directory
img_dir = '../datasets/MPII_Pose/images'

num_joint_mpii = 16
num_joint_lsp = 14

# MPII(16) -> LSP(14)
inds = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 8, 9]


class MPII(data.Dataset):
    def __init__(self, split):
        print('==> Initializing MPII data')
        annot_file = "%s/mpii/annot.mat" % (ref.data_dir)
        mat_contents = sio.loadmat(annot_file)

        self.names = mat_contents['names']
        self.idxs = mat_contents['idxs']
        self.joints = mat_contents['joints']

        self.split = split
        self.num_image = self.names.shape[0]

        print('Load {} MPII samples'.format(self.num_image))

    def load_image(self, index):
        img_name = "%s/%s" % (img_dir, self.names[index])
        img = cv2.imread(img_name)
        return img

    def get_part_info(self, index):
        # Number of rects
        num_rect = int(self.idxs[index + 1] - self.idxs[index])
        idx_rect = int(self.idxs[index] - 1)

        # For each rect,
        rects = []
        for j in range(num_rect):
            # Get mpii joint
            joint_mpii = self.joints[idx_rect + j]

            # Set lsp joints
            joint_lsp = joint_mpii[inds, :]
            rects.append(joint_lsp)

        return rects

    def __getitem__(self, index):
        # Get global constants
        num_joints = ref.num_joints  # 14
        res_in = ref.res_in  # 448 , 224
        res_out = ref.res_out  # 14  , 7
        res_ratio = res_in / res_out

        # Get raw data
        img = self.load_image(index)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rects = self.get_part_info(index)

        # Size of image
        width = img.shape[1]
        height = img.shape[0]
        ori_shape = np.zeros((2), dtype=np.float32)
        ori_shape[0] = width
        ori_shape[1] = height

        # Set initial center, scaling and rotation
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(width, height)
        r = 0


        # Data augmentation (scaling and rotation)
        if self.split == 'train':
            s = s * (2 ** rnd(ref.scale))
            r = 0 if np.random.random() < 0.6 else rnd(ref.rotate)
        inp = crop(img, c, s, r, res_in) / 255.

        # Data augmentation (flipping and jittering)
        do_flip = False
        if self.split == 'train':
            if np.random.random() < 0.5:
                do_flip = True
                inp = flip(inp)
            inp[0] = np.clip(inp[0] * (np.random.random() * .4 + .6), 0, 1)
            inp[1] = np.clip(inp[1] * (np.random.random() * .4 + .6), 0, 1)
            inp[2] = np.clip(inp[2] * (np.random.random() * .4 + .6), 0, 1)

        # Initialize valid joints
        valid2d_ = []
        pose2d_ = []
        bbox_info_ = []
        offset_ = []
        ind_ = []
        for i in range(len(rects)):
            pose2d = rects[i][:, 0:2]
            valid2d = rects[i][:, 2]


            if not (valid2d[2] == 1 and valid2d[3] == 1):  # if both left and right hip are not visible, pass
                continue



            # Set 2D pose
            for j in range(num_joints):
                pt = transform(pose2d[j], c, s, r, res_in)
                pt = pt.astype(np.float32)
                pose2d[j] = pt
            if do_flip == True:
                pose2d = shuffle_lr(pose2d)
                valid2d = shuffle_lr(valid2d)
                for j in range(num_joints):
                    pose2d[j][0] = res_in - pose2d[j][0] - 1
            for j in range(num_joints):
                if valid2d[j] == 0:
                    pose2d[j] = 0




            # Make bounding box based on 2D pose
            bbox = np.zeros((4), dtype=np.float32)
            xmin = min(x for x in pose2d[:, 0] if x != 0)
            ymin = min(x for x in pose2d[:, 1] if x != 0)
            xmax = max(x for x in pose2d[:, 0] if x != 0)
            ymax = max(x for x in pose2d[:, 1] if x != 0)
            bbox_width = xmax - xmin - 1
            bbox_height = ymax - ymin - 1
            bbox[0] = (xmin + xmax) / 2. - bbox_width / 2 * 1.3
            bbox[1] = (ymin + ymax) / 2. - bbox_height / 2 * 1.3
            bbox[2] = bbox_width * 1.3 + bbox[0]
            bbox[3] = bbox_height * 1.3 + bbox[1]



            if bbox[0] < 0: bbox[0] = 0
            if bbox[1] < 0: bbox[1] = 0
            if bbox[2] >= res_in: bbox[2] = res_in - 1
            if bbox[3] >= res_in: bbox[3] = res_in - 1
            if not ((bbox[3] - bbox[1]) * (bbox[2] - bbox[0]) > 0 and bbox[2] >= bbox[0] and bbox[3] >= bbox[1]):
                continue




            # Set grid representation
            center_x = (bbox[0] + bbox[2]) * .5
            center_y = (bbox[1] + bbox[3]) * .5

            grid_x = center_x / res_ratio
            grid_y = center_y / res_ratio

            offset_x = int(grid_x) * res_ratio + res_ratio / 2
            offset_y = int(grid_y) * res_ratio + res_ratio / 2
            offset = np.array([offset_x, offset_y]).reshape(1, 2)

            ind = np.array([int(grid_x), int(grid_y)])

            x = grid_x - int(grid_x)
            y = grid_y - int(grid_y)
            w = (bbox[2] - bbox[0]) / res_in
            h = (bbox[3] - bbox[1]) / res_in

            bbox_info = np.zeros((4), dtype=np.float)
            bbox_info[0] = x
            bbox_info[1] = y
            bbox_info[2] = 1 / np.sqrt(w)
            bbox_info[3] = 1 / np.sqrt(h)

            bbox_info_.append(bbox_info)
            offset_.append(offset)
            ind_.append(ind)
            pose2d_.append(pose2d)
            valid2d_.append(valid2d)


        k_value = math.sqrt(ref.bbox_real[0] * ref.bbox_real[1] *  1500. * 1500. / (res_in * res_in))

        # Set target
        detect = np.zeros((res_out, res_out), dtype=np.float32)
        pose2d = np.zeros((res_out, res_out, num_joints * 2), dtype=np.float32)
        valid2d = np.zeros((res_out, res_out, num_joints), dtype=np.float32)
        pose3d = np.zeros((res_out, res_out, num_joints * 3), dtype=np.float32)
        valid3d = np.zeros((res_out, res_out, num_joints), dtype=np.float32)
        root_img_ = np.zeros((res_out, res_out, 3), dtype=np.float32)
        validroot_ = np.zeros((res_out, res_out, 3), dtype=np.float32)
        xywh_ = np.zeros((res_out, res_out, 4), dtype=np.float32)
        valid_xywh = np.zeros((res_out, res_out, 4), dtype=np.float32)
        k_value_ = np.zeros((res_out, res_out), dtype=np.float32)
        detect_num = np.zeros((2),dtype=np.float32)


        for i in range(len(pose2d_)):  # len(pose2d_) => # of people
            # Get 2D root joint
            root_img = np.zeros((3), dtype=np.float32)

            # Modify 2D pose
            pose2d_[i] = pose2d_[i] - offset_[i]

            for j in range(num_joints):
                if valid2d_[i][j] == 0:
                    pose2d_[i][j] = 0
            root_img[0] = (pose2d_[i][2, 0] + pose2d_[i][3, 0]) * .5
            root_img[1] = (pose2d_[i][2, 1] + pose2d_[i][3, 1]) * .5

            if ind_[i][0] >= 0 and ind_[i][0] < res_out and ind_[i][1] >= 0 and ind_[i][1] < res_out:
                detect[ind_[i][1], ind_[i][0]] = 1.0
                pose2d[ind_[i][1], ind_[i][0], :] = pose2d_[i].flatten()
                valid2d[ind_[i][1], ind_[i][0], :] = valid2d_[i].flatten()
                root_img_[ind_[i][1], ind_[i][0], :] = root_img.flatten()
                validroot_[ind_[i][1], ind_[i][0], :2] = 1.0
                xywh_[ind_[i][1], ind_[i][0], :] = bbox_info_[i].flatten()
                valid_xywh[ind_[i][1], ind_[i][0], :] = 1.0
                k_value_[ind_[i][1], ind_[i][0]] = k_value


        # dummy value
        cam_f = np.array([1500,1500],np.float32)
        cam_c = np.array([width/2,height/2],np.float32)

        # Return
        return {
            'img': torch.from_numpy(inp).float(),
            'detect': torch.from_numpy(detect).float(),
            'pose2d': torch.from_numpy(pose2d).float(),
            'valid2d': torch.from_numpy(valid2d).float(),
            'pose3d': torch.from_numpy(pose3d).float(),
            'valid3d': torch.from_numpy(valid3d).float(),
            'root_img': torch.from_numpy(root_img_).float(),
            'validroot': torch.from_numpy(validroot_).float(),
            'xywh': torch.from_numpy(xywh_).float(),
            'valid_xywh': torch.from_numpy(valid_xywh).float(),
            'k_value': torch.from_numpy(k_value_).float(),
            'gt_index': torch.from_numpy(detect_num).float(),
            'ori_shape': torch.from_numpy(ori_shape).float(),
            'cam_f': torch.from_numpy(cam_f).float(),
            'cam_c': torch.from_numpy(cam_c).float(),
        }

    def __len__(self):
        return self.num_image

