import os.path as osp
import numpy as np
import cv2
import torch
import copy
import math
import ref
from pycocotools.coco import COCO
from torch.utils.data.dataset import Dataset
from utils.img import transform, crop
from utils.utils import process_bbox


inds = [10,9,8,11,12,13,4,3,2,5,6,7,16,0]


img_dir = '/home/seonghyun/human_pose_estimation/datasets/MuPoTS/data/MultiPersonTestSet'
test_annot_path = '/home/seonghyun/human_pose_estimation/datasets/MuPoTS/data/MuPoTS-3D.json'

class MuPoTS(Dataset):
    def __init__(self, data_split):
        self.data_split = data_split
        self.data = self.load_data()

        print('Load %d MuCo %s samples' % (len(self.data), self.data_split))


    def load_data(self):

        if self.data_split != 'test':
            print('Unknown data subset')
            assert 0

        data = []
        db = COCO(test_annot_path)


        print("Get bounding box and root from groundtruth")

        for aid in db.anns.keys():
            ann = db.anns[aid]
            if ann['is_valid'] == 0:
                continue

            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(img_dir,img['file_name'])
            fx,fy,cx,cy = img['intrinsic']
            f = np.array([fx,fy])
            c = np.array([cx,cy])

            pose3d = np.array(ann['keypoints_cam'])
            pose2d = np.array(ann['keypoints_img'])

            spine3d = pose3d[15]
            spine2d = pose2d[15]

            pose_vis = np.array(ann['keypoints_vis'])
            if pose_vis[8] == 0 or pose_vis[11] == 0:
                continue
            gt_vis = np.array(ann['is_valid'])
            bbox = np.array(ann['bbox'])

            data.append({
                'img_path': img_path,
                'image_id' : image_id,
                'bbox' : bbox,
                'pose3d' : pose3d,
                'pose2d' : pose2d,
                'pose_vis': pose_vis,
                'gt_vis' : gt_vis,
                'f' : f,
                'c' : c,
                'spine3d': spine3d,
                'spine2d': spine2d
            })
        return data

    def __getitem__(self, index):

        num_joints = 19
        num_joints_mupots = 14

        res_in = ref.res_in
        res_out = ref.res_out
        res_ratio = res_in / res_out

        data = copy.deepcopy(self.data[index])


        img = cv2.imread(data['img_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


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


        pose3d = data['pose3d']
        pose2d = data['pose2d']
        pose_vis = data['pose_vis']
        bbox = data['bbox']
        cam_f = data['f']
        cam_c = data['c']

        spine3d = data['spine3d']
        spine2d = data['spine2d']

        img_name = data['img_path'].split('/')
        img_name = img_name[-2] + '_' + img_name[-1].split('.')[0]

        ori_bbox = bbox.copy()
        ori_bbox = process_bbox(ori_bbox,width,height)


        bbox[2] = bbox[2] + bbox[0]
        bbox[3] = bbox[3] + bbox[1]


        bbox[:2] = transform(bbox[:2], c, s, r, res_in)
        bbox[2:4] = transform(bbox[2:4], c, s, r, res_in)

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

        # Initialize valid joints
        valid2d = np.zeros((num_joints_mupots), dtype=np.float32)
        pts = np.zeros((num_joints_mupots,2),dtype=np.float32)
        # Set 2D pose
        for i in range(num_joints_mupots):
            pt = transform(pose2d[inds[i]], c, s, r, res_in)
            pt = pt.astype(np.float32)
            pts[i] = pt
            valid2d[i] = pose_vis[inds[i]]


        # Set 3D pose
        pt3d = pose3d[inds,:]
        root3d = (pt3d[2,:]+pt3d[3,:])*.5
        pt3d = pt3d - root3d.reshape(1,3)
        valid3d = pose_vis[inds]

        # Get 2D root joint
        root2d = (pts[2,:]+pts[3,:])*.5
        root_img = np.zeros((3), dtype=np.float32)
        root_img[0] = root2d[0]
        root_img[1] = root2d[1]

        # Modify 2D pose
        pts -= offset
        root_img[0] -= offset[0][0]
        root_img[1] -= offset[0][1]
        root_img[2] = root3d[2]


        k_value = math.sqrt(ref.bbox_real[0] * ref.bbox_real[1] * cam_f[0] * cam_f[1] / (s*s))


        # Set target
        detect = np.zeros((res_out, res_out), dtype=np.float32)
        pose2d_ = np.zeros((res_out, res_out, num_joints * 2), dtype=np.float32)
        valid2d_ = np.zeros((res_out, res_out, num_joints), dtype=np.float32)
        pose3d_ = np.zeros((res_out, res_out, num_joints * 3), dtype=np.float32)
        valid3d_ = np.zeros((res_out, res_out, num_joints), dtype=np.float32)
        root_img_ = np.zeros((res_out, res_out, 3), dtype=np.float32)
        validroot_ = np.zeros((res_out, res_out, 3), dtype=np.float32)
        xywh_ = np.zeros((res_out, res_out, 4), dtype=np.float32)
        valid_xywh = np.zeros((res_out, res_out, 4), dtype=np.float32)
        k_value_ = np.zeros((res_out, res_out), dtype=np.float32)
        detect_num = np.array([ind[1], ind[0]], dtype=np.float32)


        if ind[0] >= 0 and ind[0] < res_out and ind[1] >= 0 and ind[1] < res_out:
            detect[ind[1], ind[0]] = 1.0
            pose2d_[ind[1], ind[0], :num_joints_mupots*2] = pts.flatten()
            pose3d_[ind[1], ind[0], :num_joints_mupots*3] = pt3d.flatten()
            valid2d_[ind[1], ind[0], :num_joints_mupots] = valid2d.flatten()
            valid3d_[ind[1], ind[0], :num_joints_mupots] = valid3d.flatten()
            validroot_[ind[1], ind[0], :] = 1.0
            root_img_[ind[1], ind[0], :] = root_img.flatten()
            xywh_[ind[1],ind[0],:] = bbox_info.flatten()
            valid_xywh[ind[1],ind[0],:] = 1.0
            k_value_[:,:] = k_value



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
            'gt_index' : torch.from_numpy(detect_num).float(),
            'ori_shape': torch.from_numpy(ori_shape).float(),
            'cam_f' : torch.from_numpy(cam_f).float(),
            'cam_c' : torch.from_numpy(cam_c).float(),
            'img_name' : img_name,
            'spine3d' : spine3d,
            'spine2d' : spine2d
        }

    def __len__(self):
        return len(self.data)


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
    dataset = MuPoTS('test')

    l = len(dataset)

    res_in = ref.res_in
    res_out = ref.res_out
    res_ratio = res_in / res_out



    idx=0
    while True:
        ret = dataset.__getitem__(idx)
        detect_num = ret['detect_num']
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
                if detect[y, x] == 1:
                    ind_x, ind_y = x, y
                    grid_x = int(x * res_ratio + res_ratio / 2)
                    grid_y = int(y * res_ratio + res_ratio / 2)
                    grid = np.array([grid_x, grid_y]).reshape(1, 2)
                    pose = pose2d[y, x].reshape(19, 2).numpy()
                    pose += grid




                    valid2d = valid2d[y, x].reshape(19, 1).numpy()

                    for j in range(19):
                        if valid2d[j] == 0:
                            pose[j] = 0


                    b_info = xywh[y, x]
                    x_center = (b_info[0] + ind_x) * res_ratio
                    y_center = (b_info[1] + ind_y) * res_ratio
                    width = 1 / (b_info[2] * b_info[2]) * res_in
                    height = 1 / (b_info[3] * b_info[3]) * res_in
                    print(width,height)

                    x1 = x_center - (width * .5)
                    y1 = y_center - (height * .5)
                    x2 = x_center + (width * .5)
                    y2 = y_center + (height * .5)

                    bbox = np.array([x1, y1, x2, y1, x1, y2, x2, y2]).reshape(4, 2)
                    draw_bbox(img_joint, bbox)
                    draw_skeleton(img_joint, pose, valid2d)


        cv2.imshow('img', img)
        cv2.imshow('img_joint', img_joint)

        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()
            break
        elif k == ord('d'):
            if idx < l - 1:
                idx = idx + 1
        elif k == ord('a'):
            if idx > 0:
                idx = idx - 1

