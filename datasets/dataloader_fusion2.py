import torch.utils.data as data
from datasets.dataloader_muco import MuCo
from datasets.dataloader_coco import COCODataset

class Fusion2(data.Dataset):
    def __init__(self, split):
        self.split = split
        self.muco = MuCo(split)
        self.COCO = COCODataset(split + '2017',True)
        self.num_muco = len(self.muco)
        self.num_coco = len(self.COCO)
        print('Load {} MuCo and {} COCO samples'.format(self.num_muco, self.num_coco))

    def __getitem__(self, index):
        if index < self.num_coco:
            return self.muco[index]
        else:
            return self.COCO[index - self.num_coco]

    def __len__(self):
        return self.num_coco*2