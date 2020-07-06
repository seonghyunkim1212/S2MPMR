# Data directory
data_dir = './annot'

# Experiment directory
exp_dir = './S2MPMR'

# Number of threads
num_threads = 4

# Number of joints
num_joints = 19 # MuCo
#num_joints = 14 # H36M


# Input resolutions
res_in = 448

# Output resolutions
res_out = 14

# Max batch size : 14x14x10(num_batch) = 1960
max_batch_size = 1960

bbox_real = [2000,2000]

# Parameters for data augmentation
scale = 0.25
rotate = 30
flip_index = [[0, 5], [1, 4], [2, 3],
              [6, 11], [7, 10], [8, 9]]


