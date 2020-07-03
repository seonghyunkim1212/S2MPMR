import numpy as np
import torch
import torch.utils.data as data
from h5py import File
import ref

class Mosh(data.Dataset):
    def __init__(self):
        print('==> Initializing Mosh data')

        f = File('%s/mosh/mosh.h5' % (ref.data_dir), 'r')
        self.trans = np.asarray(f['trans']).copy()
        self.poses = np.asarray(f['poses']).copy()
        self.betas = np.asarray(f['betas']).copy()
        f.close()

        self.num_samples = self.trans.shape[0]

        print('Load %d Mosh samples' % (self.num_samples))

    def __getitem__(self, index):
        pose, shape = self.poses[index], self.betas[index]

        return {
            'theta': torch.tensor(np.concatenate((shape, pose), axis=0)).float()
        }

    def __len__(self):
        return self.num_samples

