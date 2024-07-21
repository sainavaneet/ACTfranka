import os
import torch

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
os.environ['DEVICE'] = device

# Paths
CHECKPOINT_DIR = '/home/navaneet/Desktop/ACTfranka/ACTfranka/real_dir2/checkpoints'
DATASET_DIR = "/home/navaneet/Desktop/ACTfranka/ACTfranka/real_dir2/processed/"

# Initial configuration
INITIAL_JOINTS = [0, -0.7, 0, -2.35619449, 0, 1.57079632679, 0.785398163397]
OPEN_GRIPPER_POSE = 0.08
GRASP = 0.025
GRIPPER_FORCE = 1
INITIAL_GRIPPER_POSE = 0.06
TOTAL_EPISODES = 20
BOX_Z = 0.04
MAX_STEPS = 194

CAMERA_NAMES = ['top' , 'front']



# Task configuration
TASK_CONFIG = {
    'dataset_dir': DATASET_DIR,
    'episode_len': MAX_STEPS,
    'state_dim': 8,
    'action_dim': 8,
    'cam_width': 640,
    'cam_height': 480,
    'camera_names': CAMERA_NAMES,
    'camera_port': 50,
}

# Policy configuration
POLICY_CONFIG = {
    'lr': 1e-5,
    'device': device,
    'num_queries': MAX_STEPS,
    'kl_weight': 100,
    'hidden_dim': 512,
    'dim_feedforward': 3200,
    'lr_backbone': 1e-5,
    'backbone': 'resnet18',
    'enc_layers': 4,
    'dec_layers': 7,
    'nheads': 8,
    'camera_names': CAMERA_NAMES,
    'policy_class': 'ACT',
    'temporal_agg': False,
}

# Training configuration
TRAIN_CONFIG = {
    'seed': 42,
    'num_epochs': 20000,
    'batch_size_val': 8,
    'batch_size_train': 8,
    'eval_ckpt_name': 'policy_last.ckpt',
    'checkpoint_dir': CHECKPOINT_DIR,
}
