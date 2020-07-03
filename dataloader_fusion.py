import torch.utils.data as data
from datasets.dataloader_h36m import H36M14
from datasets.dataloader_mpii import MPII

class Fusion(data.Dataset):
    def __init__(self, split):
        self.split = split
        self.H36M = H36M14(split)
        self.MPII = MPII(split)
        self.num_h36m = len(self.H36M)
        self.num_mpii = len(self.MPII)
        print('Load {} H36M and {} MPII samples'.format(self.num_h36m, self.num_mpii))

    def __getitem__(self, index):
        if index < self.num_mpii:
            return self.H36M[index]
        else:
            return self.MPII[index - self.num_mpii]

    def __len__(self):
        return self.num_mpii*2

