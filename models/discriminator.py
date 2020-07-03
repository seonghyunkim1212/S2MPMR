# Code from Xiong Zhang
import torch
import torch.nn as nn
from models.linear_model import LinearModel
import models.util as util


class ShapeDiscriminator(LinearModel):
    def __init__(self, fc_layers, use_dropout, drop_prob, use_ac_func):
        if fc_layers[-1] != 1:
            msg = 'the neuron count of the last layer must be 1, but got {}'.format(fc_layers[-1])
            sys.exit(msg)

        super(ShapeDiscriminator, self).__init__(fc_layers, use_dropout, drop_prob, use_ac_func)

    def forward(self, inputs):
        # inputs: N x 10
        return self.fc_blocks(inputs)


class PoseDiscriminator(nn.Module):
    def __init__(self, channels):
        super(PoseDiscriminator, self).__init__()

        if channels[-1] != 1:
            msg = 'the neuron count of the last layer must be 1, but got {}'.format(channels[-1])
            sys.exit(msg)

        self.conv_blocks = nn.Sequential()
        l = len(channels)
        for idx in range(l - 2):
            self.conv_blocks.add_module(
                name='conv_{}'.format(idx),
                module=nn.Conv2d(in_channels=channels[idx], out_channels=channels[idx + 1], kernel_size=1, stride=1)
            )

        self.fc_layer = nn.ModuleList()
        for idx in range(23):
            self.fc_layer.append(nn.Linear(in_features=channels[l - 2], out_features=1))

    def forward(self, inputs):
        # inputs: N x 23 x 9
        batch_size = inputs.shape[0]
        inputs = inputs.transpose(1, 2).unsqueeze(2)  # to N x 9 x 1 x 23
        internal_outputs = self.conv_blocks(inputs)  # to N x c x 1 x 23
        o = []
        for idx in range(23):
            o.append(self.fc_layer[idx](internal_outputs[:, :, 0, idx]))

        return torch.cat(o, 1), internal_outputs


class FullPoseDiscriminator(LinearModel):
    def __init__(self, fc_layers, use_dropout, drop_prob, use_ac_func):
        if fc_layers[-1] != 1:
            msg = 'the neuron count of the last layer must be 1, but got {}'.format(fc_layers[-1])
            sys.exit(msg)

        super(FullPoseDiscriminator, self).__init__(fc_layers, use_dropout, drop_prob, use_ac_func)

    def forward(self, inputs):
        return self.fc_blocks(inputs)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self._create_sub_modules()

    def _create_sub_modules(self):
        # Create pose discriminator for 23 joint
        self.pose_discriminator = PoseDiscriminator([9, 32, 32, 1])

        # Create full pose discriminator
        fc_layers = [23 * 32, 1024, 1024, 1]
        use_dropout = [False, False, False]
        drop_prob = [0.5, 0.5, 0.5]
        use_ac_func = [True, True, False]
        self.full_pose_discriminator = FullPoseDiscriminator(fc_layers, use_dropout, drop_prob, use_ac_func)

        # Shape discriminator
        fc_layers = [10, 5, 1]
        use_dropout = [False, False]
        drop_prob = [0.5, 0.5]
        use_ac_func = [True, False]
        self.shape_discriminator = ShapeDiscriminator(fc_layers, use_dropout, drop_prob, use_ac_func)

        print('finished create the discriminator modules...')

    def forward(self, thetas):
        # inputs: N x 82(=72+10)
        batch_size = thetas.shape[0]
        shapes, poses = thetas[:, :10], thetas[:, 10:]
        shape_disc_value = self.shape_discriminator(shapes)
        rotate_matrixs = util.batch_rodrigues(poses.contiguous().view(-1, 3)).view(-1, 24, 9)[:, 1:, :]
        pose_disc_value, pose_inter_disc_value = self.pose_discriminator(rotate_matrixs)
        full_pose_disc_value = self.full_pose_discriminator(pose_inter_disc_value.contiguous().view(batch_size, -1))
        return torch.cat((pose_disc_value, full_pose_disc_value, shape_disc_value), 1)

