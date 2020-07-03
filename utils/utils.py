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

