import os
import torch
import ref
import math
import cv2
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from utils.img import transform, crop

inds = [16, 14, 12, 11, 13, 15, 10, 8, 6, 5, 7, 9, -1, -1, 0, 1, 2, 3, 4]


# Set the image directory
img_dir = '../datasets/coco'


class COCODataset(Dataset):
    '''
     0: R ankle
     1: R knee
     2: R hip
     3: L hip
     4: L knee
     5: L ankle
     6: R wrist
     7: R elbow
     8: R shoulder
     9: L shoulder
    10: L elbow
    11: L wrist
    12: neck
    13: head-top
    14: nose
    15: L eye
    16: R eye
    17: L ear
    18: R ear
    '''

    def __init__(self, image_set, is_train):
        # General options
        self.image_set = image_set
        self.is_train = is_train

        # Number of joints
        self.num_joints = 19

        # Bounding box
        self.aspect_ratio = 0.75
        self.pixel_std = 200

        # Initialize coco library (with specific annotation)
        self.coco = COCO(self._get_ann_file_keypoint())

        # Deal with class names
        cats = [cat['name']
                for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict([(self._class_to_coco_ind[cls],
                                             self._class_to_ind[cls])
                                            for cls in self.classes[1:]])

        # Get image set index
        self.image_set_index = self._load_image_set_index()

        # Number of images
        self.num_images = len(self.image_set_index)
        print('Number of images: %d' % (self.num_images))

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):

        # Get global constants
        res_in = ref.res_in
        res_out = ref.res_out
        res_ratio = res_in / res_out

        # Get index
        index = self.image_set_index[idx]

        # Load a single image
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        ori_shape = np.zeros((2), dtype=np.float32)
        ori_shape[0] = width
        ori_shape[1] = height


        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(annIds)

        # Get image
        image_file = self._image_path_from_index(index)
        img = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        # Size of image
        width = img.shape[1]
        height = img.shape[0]

        # Set initial center, scaling and rotation
        c = np.array([width/2., height/2.], dtype=np.float32)
        s = max(width, height)
        r = 0

        inp = crop(img, c, s, r, res_in) / 255.

        # Data augmentation (jittering)
        if self.is_train == True:
            inp[0] = np.clip(inp[0] * (np.random.random() * .4 + .6), 0, 1)
            inp[1] = np.clip(inp[1] * (np.random.random() * .4 + .6), 0, 1)
            inp[2] = np.clip(inp[2] * (np.random.random() * .4 + .6), 0, 1)

        # Sanitize bboxes
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))

            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2-x1, y2-y1]
                valid_objs.append(obj)
        objs = valid_objs


        # For each object,
        valid2d_ = []
        pose2d_ = []
        bbox_info_ = []
        offset_ = []
        ind_ = []
        for obj in objs:
            cls = self._coco_ind_to_class_ind[obj['category_id']]
            if cls != 1:
                continue

            # Ignore objs without keypoints annotation
            if max(obj['keypoints']) == 0:
                continue


            pose2d = np.zeros((self.num_joints,2),dtype=np.float32)
            valid2d = np.zeros((self.num_joints),dtype=np.float32)
            for i in range(self.num_joints):
                ipt = inds[i]
                if ipt != -1:
                    pt = np.array([obj['keypoints'][ipt * 3 + 0], obj['keypoints'][ipt * 3 + 1]], dtype=np.float32)
                    t_vis = obj['keypoints'][ipt * 3 + 2]
                    if t_vis >= 1:
                        pose2d[i] = transform(pt, c, s, r, res_in)

                    if t_vis >= 1:
                        t_vis = 1
                    valid2d[i] = t_vis

            if not (valid2d[2]==1 and valid2d[3]==1):
                continue



            obj['clean_bbox'][2] = obj['clean_bbox'][0] + obj['clean_bbox'][2]
            obj['clean_bbox'][3] = obj['clean_bbox'][1] + obj['clean_bbox'][3]


            obj['clean_bbox'][:2] = transform(obj['clean_bbox'][:2],c,s,r,res_in)    # res_in space
            obj['clean_bbox'][2:4] = transform(obj['clean_bbox'][2:4],c,s,r,res_in)  # res_in space


            center_x = (obj['clean_bbox'][0] + obj['clean_bbox'][2]) * .5
            center_y = (obj['clean_bbox'][1] + obj['clean_bbox'][3]) * .5


            grid_x = center_x / res_ratio
            grid_y = center_y / res_ratio


            offset_x = int(grid_x) * res_ratio + res_ratio/2
            offset_y = int(grid_y) * res_ratio + res_ratio/2
            offset = np.array([offset_x,offset_y]).reshape(1,2)
            ind = np.array([int(grid_x),int(grid_y)])

            x = grid_x - int(grid_x)
            y = grid_y - int(grid_y)
            w = (obj['clean_bbox'][2] - obj['clean_bbox'][0]) / res_in
            h = (obj['clean_bbox'][3] - obj['clean_bbox'][1]) / res_in

            bbox_info = np.zeros((4),dtype=np.float)
            bbox_info[0] = x
            bbox_info[1] = y
            bbox_info[2] = 1 / np.sqrt(w)
            bbox_info[3] = 1 / np.sqrt(h)
            bbox_info_.append(bbox_info)
            offset_.append(offset)
            ind_.append(ind)


            pose2d_.append(pose2d)
            valid2d_.append(valid2d)



        k_value= math.sqrt(ref.bbox_real[0] * ref.bbox_real[1] * 1500. * 1500. / (s*s))

        # Set target
        detect = np.zeros((res_out, res_out), dtype=np.float32)
        pose2d = np.zeros((res_out, res_out, self.num_joints * 2), dtype=np.float32)
        valid2d = np.zeros((res_out, res_out, self.num_joints), dtype=np.float32)
        pose3d = np.zeros((res_out, res_out, self.num_joints * 3), dtype=np.float32)
        valid3d = np.zeros((res_out, res_out, self.num_joints), dtype=np.float32)
        root_img_ = np.zeros((res_out, res_out, 3), dtype=np.float32)
        validroot_ = np.zeros((res_out, res_out, 3), dtype=np.float32)
        xywh_ = np.zeros((res_out, res_out, 4), dtype=np.float32)
        valid_xywh = np.zeros((res_out,res_out,4),dtype=np.float32)
        k_value_ = np.zeros((res_out,res_out),dtype=np.float32)
        detect_num = np.zeros((2), dtype=np.float32)



        for i in range(len(pose2d_)):
            # Get 2D root joint
            root_img = np.zeros((3), dtype=np.float32)

            # Modify 2D pose
            pose2d_[i] = pose2d_[i] - offset_[i]

            root_img[0] = (pose2d_[i][2, 0] + pose2d_[i][3, 0]) * .5
            root_img[1] = (pose2d_[i][2, 1] + pose2d_[i][3, 1]) * .5

            for j in range(self.num_joints):
                if valid2d_[i][j] == 0:
                    pose2d_[i][j] = 0


            if ind_[i][0] >= 0 and ind_[i][0] < res_out and ind_[i][1] >= 0 and ind_[i][1] < res_out:
                detect[ind_[i][1], ind_[i][0]] = 1.0
                pose2d[ind_[i][1], ind_[i][0], :] = pose2d_[i].flatten()
                valid2d[ind_[i][1], ind_[i][0], :] = valid2d_[i].flatten()
                root_img_[ind_[i][1], ind_[i][0], :] = root_img.flatten()
                validroot_[ind_[i][1], ind_[i][0], :2] = 1.0
                xywh_[ind_[i][1],ind_[i][0],:] = bbox_info_[i].flatten()
                valid_xywh[ind_[i][1],ind_[i][0],:] = 1.0
                k_value_[ind_[i][1],ind_[i][0]] = k_value




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
            'xywh' : torch.from_numpy(xywh_).float(),
            'valid_xywh' : torch.from_numpy(valid_xywh).float(),
            'k_value' : torch.from_numpy(k_value_).float(),
            'gt_index' : torch.from_numpy(detect_num).float(),
            'ori_shape' : torch.from_numpy(ori_shape).float(),
            'cam_f' : torch.ones((2)).float(),      #dummy value
            'cam_c' : torch.ones((2)).float()  #dummy value
        }


    def _get_ann_file_keypoint(self):
        """ get path to annotation file """
        return os.path.join(img_dir, 'annotations/person_keypoints_' + self.image_set + '.json')

    def _load_image_set_index(self):
        """ get image idxs using coco library """
        image_ids = self.coco.getImgIds()
        return image_ids

    def _box2cs(self, box):
        """ bbox to center/scale """
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        """ bbox to center/scale """
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def _image_path_from_index(self, index):
        file_name = '%012d.jpg' % index
        image_path = os.path.join(img_dir, 'images/' + self.image_set, file_name)
        return image_path


