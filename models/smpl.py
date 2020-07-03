# Code from Xiong Zhang


import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import ref
from models.util import batch_global_rigid_transformation, batch_rodrigues

model_path = './models/neutral_smpl_with_cocoplus_reg.pkl'


class SMPL(nn.Module):
    def __init__(self, joint_type='cocoplus', obj_saveable=False):
        super(SMPL, self).__init__()

        if joint_type not in ['cocoplus', 'lsp']:
            msg = 'unknow joint type: {}, it must be either "cocoplus" or "lsp"'.format(joint_type)
            sys.exit(msg)

        # Load SMPL parameters
        self.joint_type = joint_type
        with open(model_path, 'rb') as f:
            model = pickle.load(f, encoding='latin1')

        # Maintain face information?
        if obj_saveable:
            self.faces = model['f']
        else:
            self.faces = None

        # T: Mean template vertices (6890 x 3)
        # N x 3
        np_v_template = np.array(model['v_template'], dtype=np.float)
        self.register_buffer('v_template', torch.from_numpy(np_v_template).float())
        self.size = [np_v_template.shape[0], 3]

        # S: Shape blend shape basis (10 x 20670)
        # |beta| x 3N
        np_shapedirs = np.array(model['shapedirs'], dtype=np.float)
        self.num_betas = np_shapedirs.shape[-1]
        np_shapedirs = np.reshape(np_shapedirs, [-1, self.num_betas]).T
        self.register_buffer('shapedirs', torch.from_numpy(np_shapedirs).float())

        # J: Regressor for joint locations given shape (6890 x 24)
        # N x |J|
        np_J_regressor = np.array(model['J_regressor'].T.todense(), dtype=np.float)
        self.register_buffer('J_regressor', torch.from_numpy(np_J_regressor).float())

        # P: Pose blend shape basis (207 x 20670)
        # 9K x 3N
        np_posedirs = np.array(model['posedirs'], dtype=np.float)
        num_pose_basis = np_posedirs.shape[-1]
        np_posedirs = np.reshape(np_posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', torch.from_numpy(np_posedirs).float())

        # Indices of parents for each joints
        self.parents = np.array(model['kintree_table'])[0].astype(np.int32)

        # Return 19 or 14 keypoints (6890 x 19 or 14)
        np_joint_regressor = np.array(model['cocoplus_regressor'].T.todense(), dtype=np.float)
        if joint_type == 'lsp':
            self.register_buffer('joint_regressor', torch.from_numpy(np_joint_regressor[:, :14]).float())
        else:
            self.register_buffer('joint_regressor', torch.from_numpy(np_joint_regressor).float())

        # LBS weights (6890 x 24)
        np_weights = np.array(model['weights'], dtype=np.float)

        vertex_count = np_weights.shape[0]
        vertex_component = np_weights.shape[1]

        batch_size = ref.max_batch_size
        np_weights = np.tile(np_weights, (batch_size, 1))
        self.register_buffer('weight', torch.from_numpy(np_weights).float().reshape(-1, vertex_count, vertex_component))

        # rest pose (identity matrix)
        self.register_buffer('e3', torch.eye(3).float())

        self.cur_device = None

    def save_obj(self, verts, obj_mesh_name):
        if self.faces is None:
            msg = 'obj not saveable!'
            sys.exit(msg)

        with open(obj_mesh_name, 'w') as fp:
            for v in verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

            for f in self.faces:  # Faces are 1-based, not 0-based in obj files
                fp.write('f %d %d %d\n' % (f[0] + 1, f[1] + 1, f[2] + 1))

    def forward(self, beta, theta, trans, get_skin=False):
        if not self.cur_device:
            device = beta.device
            self.cur_device = torch.device(device.type, device.index)

        num_batch = beta.shape[0]

        # 1. Add shape blend shapes
        # (N x 10) x (10 x 6890*3) = N x 6890 x 3
        v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template

        # 2. Infer shape-dependent joint locations
        # N x 24 x 3
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim=2)

        # 3. Add pose blend shapes
        Rs = batch_rodrigues(theta.view(-1, 3)).view(-1, 24, 3, 3)
        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)

        # (N x 207) x (207, 20670) -> N x 6890 x 3
        v_posed = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1]) + v_shaped

        # 4. Get the global joint location
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, rotate_base=True)

        # 5. Do skinning
        weight = self.weight[:num_batch]
        # W is N x 6890 x 24
        W = weight.view(num_batch, -1, 24)
        # (N x 6890 x 24) x (N x 24 x 16) -> (N x 6890 x 16) -> (N x 6890 x 4 x 4)
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        # homogeneous: N x 6890 x 4
        v_posed_homo = torch.cat([v_posed, torch.ones(num_batch, v_posed.shape[1], 1, device=self.cur_device)], dim=2)
        # (N x 6890 x 4 x 4) x (N x 6890 x 4 x 1) -> (N x 6890 x 4 x 1)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))
        # N x 6890 x 3
        verts = v_homo[:, :, :3, 0]

        # 6. Translation
        verts = verts + torch.reshape(trans, (num_batch, 1, 3))

        # 7. Get cocoplus or lsp joints
        joint_x = torch.matmul(verts[:, :, 0], self.joint_regressor)
        joint_y = torch.matmul(verts[:, :, 1], self.joint_regressor)
        joint_z = torch.matmul(verts[:, :, 2], self.joint_regressor)

        joints = torch.stack([joint_x, joint_y, joint_z], dim=2)

        if get_skin:
            return verts, joints, Rs
        else:
            return joints
