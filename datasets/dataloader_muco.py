import os.path as osp
from pycocotools.coco import COCO
import numpy as np
import cv2
import torch
import copy
import math
import ref
from torch.utils.data.dataset import Dataset
from utils.img import transform, crop
from utils.utils import transform3d

inds = [10,9,8,11,12,13,4,3,2,5,6,7,16,0]

# Set the directory
img_dir = '../datasets/MuCo/data'
train_annot_path = '../datasets/MuCo/data/MuCo-3DHP.json'

class MuCo(Dataset):
    def __init__(self, data_split):
        self.data_split = data_split
        self.data = self.load_data()

        print('Load %d MuCo %s samples' % (len(self.data), self.data_split))

    def load_data(self):

        if self.data_split == 'train':
            db = COCO(train_annot_path)
        else:
            print('Unknown data subset')
            assert 0

        data = []
        for iid in db.imgs.keys():
            img = db.imgs[iid]
            img_id = img["id"]
            imgname = img['file_name']
            img_path = osp.join(img_dir, imgname)
            f = img["f"]
            c = img["c"]

            f = np.asarray(f,dtype=np.float32)
            c = np.asarray(c,dtype=np.float32)

            pose3d = []
            pose2d = []
            pose_vis = []
            bbox = []

            # crop the closesref.data_t person to the camera
            ann_ids = db.getAnnIds(img_id)
            anns = db.loadAnns(ann_ids)

            for ann in anns:
                pose3d.append(ann['keypoints_cam'])
                pose2d.append(ann['keypoints_img'])
                pose_vis.append(ann['keypoints_vis'])
                bbox.append(ann['bbox'])

            pose3d = np.asarray(pose3d,dtype=np.float32)
            pose2d = np.asarray(pose2d,dtype=np.float32)
            pose_vis = np.asarray(pose_vis,dtype=np.float32)
            bbox = np.asarray(bbox,dtype=np.float32)

            data.append({
                'img_path': img_path,
                'bbox': bbox,
                'pose3d': pose3d,  # [org_img_x, org_img_y, depth]
                'pose2d': pose2d,  # [X, Y, Z] in camera coordinate
                'pose_vis': pose_vis,
                'f': f,
                'c': c
                })
        return data

    def __getitem__(self, index):
        if self.data_split == 'train':
            index = np.random.randint(len(self.data))

        num_joints = 19
        num_joints_muco = 14

        res_in = ref.res_in
        res_out = ref.res_out
        res_ratio = res_in / res_out

        data = copy.deepcopy(self.data[index])

        bbox = data['bbox']
        pose3d = data['pose3d']
        pose2d = data['pose2d']
        pose_vis = data['pose_vis']
        cam_f = data['f']
        cam_c = data['c']


        for i in range(bbox.shape[0]):
            bbox[i][2] = bbox[i][2] + bbox[i][0]
            bbox[i][3] = bbox[i][3] + bbox[i][1]

        # 1. load image
        img = cv2.imread(data['img_path'])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        width = img.shape[1]
        height = img.shape[0]

        ori_shape = np.zeros((2), dtype=np.float32)
        ori_shape[0] = width
        ori_shape[1] = height


        # Set initial center, scaling and rotation
        c = np.array([width / 2, height / 2], dtype=np.float32)
        s = width
        r = 0

        inp = crop(img, c, s, r, res_in) / 255.

        if self.data_split == 'train':
            inp[0] = np.clip(inp[0] * (np.random.random() * .4 + .6), 0, 1)
            inp[1] = np.clip(inp[1] * (np.random.random() * .4 + .6), 0, 1)
            inp[2] = np.clip(inp[2] * (np.random.random() * .4 + .6), 0, 1)

        valid2d_ = []
        valid3d_ = []
        pose2d_ = []
        pose3d_ = []
        bbox_ = []
        bbox_info_ = []
        offset_ = []
        ind_ = []
        root_depth = []
        for i in range(len(pose3d)):
            pts = np.zeros((14,2),dtype=np.float32)
            valid = np.zeros((14),dtype=np.float32)
            pt3d = np.zeros((14,3),dtype=np.float32)
            for k in range(len(inds)):
                pts[k] = pose2d[i][inds[k]]
                valid[k] = pose_vis[i][inds[k]] * 1.0
                pt3d[k] = pose3d[i][inds[k]]



            if not (valid[2] == 1.0 and valid[3] == 1.0):
                continue

            for j in range(num_joints_muco):
                pt = transform(pts[j],c,s,r,res_in)
                pt = pt.astype(np.float32)
                pts[j] = pt

            for j in range(num_joints_muco):
                pt = transform3d(pt3d[j],r)
                pt = pt.astype(np.float32)
                pt3d[j] = pt


            root3d = (pt3d[2,:] + pt3d[3,:])*.5
            pt3d = pt3d - root3d.reshape(1,3)


            bbox[i][:2] = transform(bbox[i][:2],c,s,r,res_in)
            bbox[i][2:4] = transform(bbox[i][2:4],c,s,r,res_in)


            center_x = (bbox[i][0] + bbox[i][2]) * .5
            center_y = (bbox[i][1] + bbox[i][3]) * .5

            grid_x = center_x / res_ratio
            grid_y = center_y / res_ratio

            offset_x = int(grid_x) * res_ratio + res_ratio / 2
            offset_y = int(grid_y) * res_ratio + res_ratio / 2
            offset = np.array([offset_x,offset_y]).reshape(1,2)
            ind = np.array([int(grid_x),int(grid_y)])

            x = grid_x - int(grid_x)
            y = grid_y - int(grid_y)
            w = (bbox[i][2] - bbox[i][0]) / res_in
            h = (bbox[i][3] - bbox[i][1]) / res_in

            bbox_info = np.zeros((4), dtype=np.float)
            bbox_info[0] = x
            bbox_info[1] = y
            bbox_info[2] = 1 / np.sqrt(w)
            bbox_info[3] = 1 / np.sqrt(h)

            bbox_info_.append(bbox_info)
            root_depth.append(root3d[2])
            offset_.append(offset)
            ind_.append(ind)

            pose2d_.append(pts)
            valid2d_.append(valid)
            pose3d_.append(pt3d)
            valid3d_.append(valid)
            bbox_.append(bbox)

        k_value = math.sqrt(ref.bbox_real[0] * ref.bbox_real[1] * cam_f[0] * cam_f[1] / (s*s))


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
        detect_num = np.zeros((2), dtype=np.float32)

        for i in range(len(pose2d_)):

            root_img = np.zeros((3),dtype=np.float32)

            pose2d_[i] = pose2d_[i] - offset_[i]

            root_img[0] = (pose2d_[i][2, 0] + pose2d_[i][3, 0]) * .5
            root_img[1] = (pose2d_[i][2, 1] + pose2d_[i][3, 1]) * .5
            root_img[2] = root_depth[i]

            if ind_[i][0] >= 0 and ind_[i][0] < res_out and ind_[i][1] >= 0 and ind_[i][1] < res_out:
                detect[ind_[i][1],ind_[i][0]] = 1.0
                pose2d[ind_[i][1],ind_[i][0],:num_joints_muco*2] = pose2d_[i].flatten()
                valid2d[ind_[i][1],ind_[i][0],:num_joints_muco] = valid2d_[i].flatten()
                pose3d[ind_[i][1],ind_[i][0],:num_joints_muco*3] = pose3d_[i].flatten()
                valid3d[ind_[i][1],ind_[i][0],:num_joints_muco] = valid3d_[i].flatten()
                validroot_[ind_[i][1],ind_[i][0],:] = 1.0
                root_img_[ind_[i][1],ind_[i][0],:] = root_img.flatten()
                xywh_[ind_[i][1],ind_[i][0],:] = bbox_info_[i].flatten()
                valid_xywh[ind_[i][1],ind_[i][0],:] = 1.0
                k_value_[ind_[i][1],ind_[i][0]] = k_value




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
            'gt_index' : torch.from_numpy(detect_num).float(),
            'ori_shape' : torch.from_numpy(ori_shape).float(),
            'cam_f' : torch.from_numpy(cam_f).float(),
            'cam_c' : torch.from_numpy(cam_c).float()
        }




    def __len__(self):
        return len(self.data)

