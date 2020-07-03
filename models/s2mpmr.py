import torch
import torch.nn as nn
import torchvision.models as models
from models.smpl import SMPL
import ref

# Dimension of theta: detection, rotation, shape, scale+translation , root joint
dim_theta = 1 + 24 * 3 + 10 + 3 + 3

# Dimension of bbox
dim_bbox = 4


class S2MPMR(nn.Module):
    def __init__(self):
        super(S2MPMR, self).__init__()

        # Load pretrained resnet-50 model
        pretrained = models.resnet50(pretrained=True)

        # Remove last 2 layers
        modules = list(pretrained.children())[:-2]

        self.module = nn.ModuleList(modules)

        # Add regression layer
        self.reg = nn.Conv2d(2048, dim_theta, 1)


        # bbox regression layer
        self.bbox_reg = nn.Conv2d(2048,dim_bbox,1)
        self.relu = nn.ReLU(inplace=True)


        # SMPL layer
        self.smpl = SMPL(joint_type='lsp', obj_saveable=True)

    def forward(self, x, k_value):
        nb = x.shape[0]

        # Compute feature
        x = self.module[0](x)
        x = self.module[1](x)
        x = self.module[2](x)
        x = self.module[3](x)
        C2 = self.module[4](x)
        C3 = self.module[5](C2)
        C4 = self.module[6](C3)
        C5 = self.module[7](C4)

        # Regress outputs
        y = self.reg(C5)
        bbox = self.bbox_reg(C5)
        bbox = self.relu(bbox)
        ny = y.shape[2]
        nx = y.shape[3]
        y = y.permute(0, 2, 3, 1).reshape(nb * ny * nx, dim_theta)
        bbox = bbox.permute(0, 2, 3, 1).reshape(nb * ny * nx, dim_bbox)

        detect = y[:, 0].clone()
        thetas = y[:, 1:83].clone()
        betas = y[:, 1:11].clone()
        poses = y[:, 11:83].view(-1, 24, 3).clone()
        camera = y[:, 83:86].clone()
        root_y = y[:, 86:].clone()



        # Compute vertices and 3D joints
        verts, j3d, Rs = self.smpl(beta=betas, theta=poses, trans=torch.zeros(nb * ny * nx, 3).to(x.device),
                                   get_skin=True)
        verts = verts * 1000.
        j3d = j3d * 1000.

        # Normalize 3D joints (zero mean)
        j3d_root = ((j3d[:, 2, :] + j3d[:, 3, :]) * .5).reshape(nb * ny * nx, 1, 3)
        j3d = j3d - j3d_root
        verts = verts - j3d_root

        # Compute projected 2D joints
        j2d = j3d[:, :, :2].clone()
        j2d = j2d * camera[:, 0].view(-1, 1, 1) + camera[:, 1:3].view(-1, 1, 2)


        # Set output
        detect = detect.reshape(nb, ny, nx)
        thetas = thetas.reshape(nb, ny, nx, 82)
        camera = camera.reshape(nb, ny, nx, 3)
        verts = verts.reshape(nb, ny, nx, -1, 3)
        j3d = j3d.reshape(nb, ny, nx, -1)
        j2d = j2d.reshape(nb, ny, nx, -1)
        root_y = root_y.reshape(nb, ny, nx, -1)
        bbox = bbox.reshape(nb,ny,nx,-1)

        width = bbox[:,:,:,2].clone()
        height = bbox[:,:,:,3].clone()


        # Compute absolute depth of the root joint
        root_y[:,:,:,2] = root_y[:,:,:,2] * k_value * width * height


        return (detect, thetas, camera, verts, j3d, j2d, root_y, bbox)

