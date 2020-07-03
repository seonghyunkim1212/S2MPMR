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

img_dir = '/home/seonghyun/human_pose_estimation/datasets/MuCo/data'
train_annot_path = '/home/seonghyun/human_pose_estimation/datasets/MuCo/data/MuCo-3DHP.json'

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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# For drawing 3D pose
b_c = np.array([0,1,2,14])
b_l_1 = np.array([5,4,3,14])
b_l_2 = np.array([14,12,13])
b_r_1 = np.array([11,10,9,12])
b_r_2 = np.array([12,8,7,6])

def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

def draw_pose_3d(pose3d, num_people, ax):
    ax.clear()

    for i in range(num_people):
        ax.plot(pose3d[i][b_c,0], pose3d[i][b_c,2], -pose3d[i][b_c,1], lw=5, c='g')
        ax.plot(pose3d[i][b_l_1,0], pose3d[i][b_l_1,2], -pose3d[i][b_l_1,1], lw=5, c='r')
        ax.plot(pose3d[i][b_l_2,0], pose3d[i][b_l_2,2], -pose3d[i][b_l_2,1], lw=5, c='r')
        ax.plot(pose3d[i][b_r_1,0], pose3d[i][b_r_1,2], -pose3d[i][b_r_1,1], lw=5, c='b')
        ax.plot(pose3d[i][b_r_2,0], pose3d[i][b_r_2,2], -pose3d[i][b_r_2,1], lw=5, c='b')

    #ax.set_aspect('equal')
    axisEqual3D(ax)

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("-Y")


bone = [[5,4],[4,3],[0,1],[1,2],[2,3],[9,3],[8,2],[8,9],[9,10],\
        [8,7],[10,11],[7,6],[15,16],[14,15],[14,16],[15,17],[16,18],[17,9],[18,8]]

def draw_skeleton(img, joints, valid):
    thickness = 3
    radius = 5

    for i in range(joints.shape[0]):
        cv2.circle(img, (int(joints[i,0]+0.5), int(joints[i,1]+0.5)), 5, (0,255,255), -1)

    for i in range(len(bone)):
        a, b = bone[i][0], bone[i][1]
        if valid[a] == 1 and valid[b] == 1:
            cv2.line(img, (int(joints[a,0]+0.5), int(joints[a,1]+0.5)), (int(joints[b,0]+0.5), int(joints[b,1]+0.5)), (0,255,0), thickness)

    return img


def draw_bbox(img, bbox):
    thickness = 2
    cv2.line(img, (int(bbox[0,0]+0.5), int(bbox[0,1]+0.5)), (int(bbox[1,0]+0.5), int(bbox[1,1]+0.5)), (0,0,255), thickness)
    cv2.line(img, (int(bbox[1, 0] + 0.5), int(bbox[1, 1] + 0.5)), (int(bbox[3, 0] + 0.5), int(bbox[3, 1] + 0.5)),(0, 0, 255), thickness)
    cv2.line(img, (int(bbox[0, 0] + 0.5), int(bbox[0, 1] + 0.5)), (int(bbox[2, 0] + 0.5), int(bbox[2, 1] + 0.5)),(0, 0, 255), thickness)
    cv2.line(img, (int(bbox[2, 0] + 0.5), int(bbox[2, 1] + 0.5)), (int(bbox[3, 0] + 0.5), int(bbox[3, 1] + 0.5)),(0, 0, 255), thickness)

    return img

if __name__ == '__main__':
    dataset = MuCo('train')

    res_in = ref.res_in
    res_out = ref.res_out
    res_ratio = res_in / res_out

    # 3D plot
    fig = plt.figure(figsize=(6.0, 6.0))
    ax = fig.gca(projection='3d')

    num_images = len(dataset)

    idx = 0
    while True:
        ret = dataset.__getitem__(idx)
        detect = ret['detect']
        pose2d = ret['pose2d']
        pose3d = ret['pose3d']
        xywh = ret['xywh']
        root_img = ret['root_img']
        valid2d = ret['valid2d']
        img = ret['img'].numpy().copy()
        img = (img.transpose((1, 2, 0)) * 255.0).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_joint = ret['img'].numpy().copy()
        img_joint = (img_joint.transpose((1, 2, 0)) * 255.0).astype(np.uint8)
        img_joint = cv2.cvtColor(img_joint, cv2.COLOR_RGB2BGR)

        pose2d_ = []
        pose3d_ = []
        for y in range(res_out):
            for x in range(res_out):
                if detect[y, x] == 1:
                    ind_x, ind_y = x, y
                    grid_x = int(x * res_ratio + res_ratio / 2)
                    grid_y = int(y * res_ratio + res_ratio / 2)
                    grid = np.array([grid_x, grid_y]).reshape(1, 2)
                    pose = pose2d[y, x].reshape(19, 2).numpy()
                    pose += grid

                    for j in range(19):
                        if valid2d[y, x][j] == 0:
                            pose[j] = 0


                    root = root_img[y,x].reshape(1,3).numpy()
                    root[0][:2] +=grid.reshape((2))



                    joint = draw_skeleton(img_joint, pose, valid2d[y, x])

                    b_info = xywh[y, x]
                    x_center = (b_info[0] + ind_x) * res_ratio
                    y_center = (b_info[1] + ind_y) * res_ratio
                    width = 1 / (b_info[2] * b_info[2]) * res_in
                    height = 1 / (b_info[3] * b_info[3]) * res_in

                    x1 = x_center - (width * .5)
                    y1 = y_center - (height * .5)
                    x2 = x_center + (width * .5)
                    y2 = y_center + (height * .5)

                    bbox = np.array([x1, y1, x2, y1, x1, y2, x2, y2]).reshape(4, 2)
                    draw_bbox(img_joint, bbox)

                    pose2d_.append(pose)

        num_people = len(pose2d_)


        cv2.imshow('img', img)
        cv2.imshow('img_joint', img_joint)

        #draw_pose_3d(pose3d_,num_people,ax)
        #plt.draw()
        #plt.pause(0.001)

        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()
            break
        elif k == ord('d'):
            if idx < num_images - 1:
                idx = idx + 1
        elif k == ord('a'):
            if idx > 0:
                idx = idx - 1