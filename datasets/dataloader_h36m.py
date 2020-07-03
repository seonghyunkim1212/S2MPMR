import numpy as np
import torch
import torch.utils.data as data
import cv2
from h5py import File
import ref
from utils.utils import shuffle_lr
from utils.img import transform, crop
from utils.utils import transform3d
from opts import Opts
import math

opt = Opts().parse()

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

img_dir = '/home/seonghyun/human_pose_estimation/datasets/H36M/images_original'

class H36M14(data.Dataset):
    def __init__(self, split):
        print('==> Initializing H36M %s data' % (split))
        annot = {}
        tags = ['idx', 'pose2d', 'pose3d', 'bbox', 'cam_f', 'cam_c', 'subject', 'action', 'subaction', 'camera', 'istrain']
        if opt.protocol == 'p1':
            f = File('%s/h36m14/h36m14s.h5' % (ref.data_dir), 'r')
        elif opt.protocol == 'p2':
            f = File('%s/h36m14/h36m14s_protocol2.h5' % (ref.data_dir), 'r')
        for tag in tags:
            annot[tag] = np.asarray(f[tag]).copy()
        f.close()

        idxs = np.arange(annot['idx'].shape[0])[annot['istrain'] == (1 if split == 'train' else 0)]
        for tag in tags:
            annot[tag] = annot[tag][idxs]

        self.split = split
        self.annot = annot
        self.num_samples = len(self.annot['idx'])

        print('Load %d H36M %s samples' % (self.num_samples, split))

    def load_image(self, index):
        dirname = 's_{:02d}_act_{:02d}_subact_{:02d}_ca_{:02d}'.format(self.annot['subject'][index], \
            self.annot['action'][index], self.annot['subaction'][index], self.annot['camera'][index])
        imgname = '{}/{}/{}_{:06d}.jpg'.format(img_dir, dirname, dirname, self.annot['idx'][index])
        img = cv2.imread(imgname)
        return img

    def get_part_info(self, index):
        pose2d = self.annot['pose2d'][index].copy()
        bbox = self.annot['bbox'][index].copy()
        pose3d = self.annot['pose3d'][index].copy()
        cam_f = self.annot['cam_f'][index].copy()
        cam_c = self.annot['cam_c'][index].copy()
        return pose2d, bbox, pose3d , cam_f, cam_c

    def __getitem__(self, index):
        if self.split == 'train':
            index = np.random.randint(self.num_samples)

        # Get global constants
        num_joints = ref.num_joints
        res_in = ref.res_in
        res_out = ref.res_out
        res_ratio = res_in / res_out

        # Get 2D/3D pose, bounding box, camera information
        pose2d, bbox, pose3d, cam_f, cam_c = self.get_part_info(index)



        # Get image
        img = self.load_image(index)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        width = img.shape[1]
        height = img.shape[0]

        ori_shape = np.zeros((2), dtype=np.float32)
        ori_shape[0] = width
        ori_shape[1] = height

        bbox[2] = bbox[2] + bbox[0]
        bbox[3] = bbox[3] + bbox[1]


        # Set initial center, scaling and rotation
        c = np.array([width/2, height/2], dtype=np.float32)
        s = width
        r = 0
        inp = crop(img, c, s, r, res_in) / 255.


        # Initialize valid nvidijoints
        valid2d = np.zeros((num_joints), dtype=np.float32)

        bbox[:2] = transform(bbox[:2],c,s,r,res_in)
        bbox[2:4] = transform(bbox[2:4],c,s,r,res_in)

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
        bbox_info = np.zeros((4),dtype=np.float)
        bbox_info[0] = x
        bbox_info[1] = y
        bbox_info[2] = 1 / np.sqrt(w)
        bbox_info[3] = 1 / np.sqrt(h)


        # Set 2D pose
        for i in range(num_joints):
            pt = transform(pose2d[i], c, s, r, res_in)
            pt = pt.astype(np.float32)
            pose2d[i] = pt
            valid2d[i] = 1.0

        # Data augmentation (jittering)
        do_flip = False
        if self.split == 'train':
            inp[0] = np.clip(inp[0] * (np.random.random() * .4 + .6), 0, 1)
            inp[1] = np.clip(inp[1] * (np.random.random() * .4 + .6), 0, 1)
            inp[2] = np.clip(inp[2] * (np.random.random() * .4 + .6), 0, 1)

        # Data augmentation for 3D pose
        if self.split == 'train':
            for i in range(num_joints):
                pt3d = transform3d(pose3d[i], r)
                pt3d = pt3d.astype(np.float32)
                pose3d[i] = pt3d

            if do_flip == True:
                pose3d = shuffle_lr(pose3d)
                for i in range(num_joints):
                    pose3d[i][0] = -pose3d[i][0]




        # Set 3D pose
        root3d = (pose3d[2,:]+pose3d[3,:])*.5
        pose3d = pose3d - root3d.reshape(1,3)
        valid3d = np.ones((num_joints), dtype=np.float32)


        # Get 2D root joint
        root2d = (pose2d[2,:]+pose2d[3,:])*.5

        root_img = np.zeros((3),dtype=np.float32)
        root_img[0] = root2d[0]
        root_img[1] = root2d[1]



        # Modify 2D pose
        pose2d -= offset
        root_img[0] -= offset[0][0]
        root_img[1] -= offset[0][1]
        root_img[2] = root3d[2]



        k_value= math.sqrt(ref.bbox_real[0] * ref.bbox_real[1] *  cam_f[0] * cam_f[1] / (res_in * res_in))

        # Set target
        detect = np.zeros((res_out, res_out), dtype=np.float32)
        pose2d_ = np.zeros((res_out, res_out, num_joints*2), dtype=np.float32)
        valid2d_ = np.zeros((res_out, res_out, num_joints), dtype=np.float32)
        pose3d_ = np.zeros((res_out, res_out, num_joints*3), dtype=np.float32)
        valid3d_ = np.zeros((res_out, res_out, num_joints), dtype=np.float32)
        root_img_ = np.zeros((res_out, res_out, 3), dtype=np.float32)
        validroot_ = np.zeros((res_out,res_out,3),dtype=np.float32)
        xywh_ = np.zeros((res_out,res_out,4),dtype=np.float32)
        valid_xywh = np.zeros((res_out,res_out,4),dtype=np.float32)
        k_value_ = np.zeros((res_out, res_out), dtype=np.float32)
        detect_num = np.array([ind[1], ind[0]], dtype=np.float32)

        if ind[0] >= 0 and ind[0] < res_out and ind[1] >= 0 and ind[1] < res_out:
            detect[ind[1], ind[0]] = 1.0
            pose2d_[ind[1], ind[0], :] = pose2d.flatten()
            pose3d_[ind[1], ind[0], :] = pose3d.flatten()
            valid2d_[ind[1], ind[0], :] = valid2d.flatten()
            valid3d_[ind[1], ind[0], :] = valid3d.flatten()
            validroot_[ind[1], ind[0], :] = 1.0
            root_img_[ind[1], ind[0], :] = root_img.flatten()
            xywh_[ind[1],ind[0],:] = bbox_info.flatten()
            valid_xywh[ind[1],ind[0],:] = 1.0
            k_value_[ind[1],ind[0]] = k_value


        return {
            'img': torch.from_numpy(inp).float(),
            'detect': torch.from_numpy(detect).float(),
            'pose2d': torch.from_numpy(pose2d_).float(),
            'valid2d': torch.from_numpy(valid2d_).float(),
            'pose3d': torch.from_numpy(pose3d_).float(),
            'valid3d': torch.from_numpy(valid3d_).float(),
            'root_img': torch.from_numpy(root_img_).float(),
            'validroot': torch.from_numpy(validroot_).float(),
            'xywh' : torch.from_numpy(xywh_).float(),
            'valid_xywh' : torch.from_numpy(valid_xywh).float(),
            'k_value' : torch.from_numpy(k_value_).float(),
            'gt_index': torch.from_numpy(detect_num).float(),
            'ori_shape': torch.from_numpy(ori_shape).float(),
            'cam_f' : torch.from_numpy(cam_f).float(),
            'cam_c' : torch.from_numpy(cam_c).float(),
        }

    def __len__(self):
        return self.num_samples




bone_r = [[0,1],[1,2],[6,7],[7,8]]
bone_l = [[3,4],[4,5],[9,10],[10,11]]
bone_c = [[12,13]]

def draw_skeleton(img, joints, valid):
    thickness = 3
    radius = 5

    for i in range(ref.num_joints):
        img = cv2.circle(img, (int(joints[i,0]+0.5), int(joints[i,1]+0.5)), 5, (0,255,255), -1)

    for i in range(len(bone_r)):
        a, b = bone_r[i][0], bone_r[i][1]
        if valid[a] == 1 and valid[b] == 1:
            img = cv2.line(img, (int(joints[a,0]+0.5), int(joints[a,1]+0.5)), (int(joints[b,0]+0.5), int(joints[b,1]+0.5)), (0,0,255), thickness)

    for i in range(len(bone_l)):
        a, b = bone_l[i][0], bone_l[i][1]
        if valid[a] == 1 and valid[b] == 1:
            img = cv2.line(img, (int(joints[a,0]+0.5), int(joints[a,1]+0.5)), (int(joints[b,0]+0.5), int(joints[b,1]+0.5)), (255,0,0), thickness)

    for i in range(len(bone_c)):
        a, b = bone_c[i][0], bone_c[i][1]
        if valid[a] == 1 and valid[b] == 1:
            img = cv2.line(img, (int(joints[a,0]+0.5), int(joints[a,1]+0.5)), (int(joints[b,0]+0.5), int(joints[b,1]+0.5)), (0,255,0), thickness)

    return img


def draw_bbox(img, bbox):
    thickness = 2
    cv2.line(img, (int(bbox[0,0]+0.5), int(bbox[0,1]+0.5)), (int(bbox[1,0]+0.5), int(bbox[1,1]+0.5)), (0,0,255), thickness)
    cv2.line(img, (int(bbox[1, 0] + 0.5), int(bbox[1, 1] + 0.5)), (int(bbox[3, 0] + 0.5), int(bbox[3, 1] + 0.5)),(0, 0, 255), thickness)
    cv2.line(img, (int(bbox[0, 0] + 0.5), int(bbox[0, 1] + 0.5)), (int(bbox[2, 0] + 0.5), int(bbox[2, 1] + 0.5)),(0, 0, 255), thickness)
    cv2.line(img, (int(bbox[2, 0] + 0.5), int(bbox[2, 1] + 0.5)), (int(bbox[3, 0] + 0.5), int(bbox[3, 1] + 0.5)),(0, 0, 255), thickness)

    return img

if __name__ == '__main__':
    dataset = H36M14(split='train')
    l = len(dataset)

    res_in = ref.res_in
    res_out = ref.res_out
    res_ratio = res_in / res_out

    for _ in range(l):
        ret = dataset.__getitem__(_)
        detect = ret['detect']
        pose2d = ret['pose2d']
        xywh = ret['xywh']
        valid2d = ret['valid2d']
        img = ret['img'].numpy().copy()
        img = (img.transpose((1, 2, 0)) * 255.0).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_joint = ret['img'].numpy().copy()
        img_joint = (img_joint.transpose((1, 2, 0)) * 255.0).astype(np.uint8)
        img_joint = cv2.cvtColor(img_joint, cv2.COLOR_RGB2BGR)

        for y in range(res_out):
            for x in range(res_out):
                if detect[y,x] == 1:
                    ind_x, ind_y = x, y
                    grid_x = int(x * res_ratio + res_ratio / 2)
                    grid_y = int(y * res_ratio + res_ratio / 2)
                    grid = np.array([grid_x, grid_y]).reshape(1, 2)
                    pose = pose2d[y, x].reshape(14, 2).numpy()
                    pose += grid

                    valid2d = valid2d[y,x].reshape(14,1).numpy()

                    for j in range(14):
                        if valid2d[j] == 0:
                            pose[j] = 0

                    b_info = xywh[y, x]
                    x_center = (b_info[0] + ind_x) * res_ratio
                    y_center = (b_info[1] + ind_y) * res_ratio
                    width = 1 / (b_info[2] * b_info[2]) * res_in
                    height = 1 / (b_info[3] * b_info[3]) * res_in

                    x1 = x_center - (width * .5)
                    y1 = y_center - (height * .5)
                    x2 = x_center + (width * .5)
                    y2 = y_center + (height * .5)

                    bbox = np.array([x1,y1,x2,y1,x1,y2,x2,y2]).reshape(4,2)

        draw_bbox(img_joint , bbox)
        img_joint = cv2.circle(img_joint, (int(x_center), int(y_center)), 5, (0, 255, 255), -1)
        #draw_skeleton(img_joint, pose, valid2d)
        cv2.imshow('img', img)
        cv2.imshow('img_joint',img_joint)

        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()
            break