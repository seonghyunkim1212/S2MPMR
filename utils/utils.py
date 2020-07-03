import numpy as np
from numpy.random import randn
import ref


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def rnd(x):
    return max(-2 * x, min(2 * x, randn() * x))


def flip(img):
    return img[:, :, ::-1].copy()


def shuffle_lr(x):
    for e in ref.flip_index:
        x[e[0]], x[e[1]] = x[e[1]].copy(), x[e[0]].copy()
    return x



def transform3d(pt, rot):
    rot = -rot
    t = np.eye(3)
    ang = rot * np.math.pi / 180
    s = np.math.sin(ang)
    c = np.math.cos(ang)
    t[0, 0] = c
    t[0, 1] = -s
    t[1, 0] = s
    t[1, 1] = c
    new_pt = np.dot(t, pt)
    return new_pt

def process_bbox(bbox, width, height):
    # sanitize bboxes
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
    if w*h > 0 and x2 >= x1 and y2 >= y1:
        bbox = np.array([x1, y1, x2-x1, y2-y1])
    else:
        return None

    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    aspect_ratio = ref.res_in / ref.res_in
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w*1.25
    bbox[3] = h*1.25
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.
    return bbox

