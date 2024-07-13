import os
# fallback to cpu if mps is not available for specific operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"
import torch

# data directory


# checkpoint directory
#CHECKPOINT_DIR = 'checkpoints2/'
CHECKPOINT_DIR = '/home/navaneet/Desktop/ACTfranka/ACTfranka/checkpoints2'
# CHECKPOINT_DIR = 'testing_files'

# device
device = 'cpu'
if torch.cuda.is_available(): device = 'cuda'
#if torch.backends.mps.is_available(): device = 'mps'
os.environ['DEVICE'] = device

DATASET_DIR = "/home/navaneet/Desktop/ACTfranka/ACTfranka/datasets/test"

DATA_DIR = DATASET_DIR

INITIAL_JOINTS = [0, -0.7, 0, -2.35619449, 0, 1.57079632679, 0.785398163397]

OPEN_GRIPPER_POSE = 0.06

GRASP = 0.025

GRIPPER_FORCE = 0.12

INITAL_GRIPPER_POSE = 0.06

TOTAL_EPISODES = 30

BOX_Z = 0.04

MAX_STEPS = 262
# task config (you can add new tasks)
TASK_CONFIG = {
    'dataset_dir': DATA_DIR,
    'episode_len': MAX_STEPS,
    'state_dim': 8,
    'action_dim': 8,
    'cam_width': 640,
    'cam_height': 480,
    'camera_names': ['top' , 'front'],
    'camera_port': 50
}


####change --------------------- shape 
# policy config
POLICY_CONFIG = {
    'lr': 1e-5,
    'device': device,
    'num_queries': MAX_STEPS,
    'kl_weight': 100,
    'hidden_dim': 256,
    'dim_feedforward': 3200,
    'lr_backbone': 1e-5,
    'backbone': 'resnet18',
    'enc_layers': 4,
    'dec_layers': 7,
    'nheads': 8,
    'camera_names': ['top' , 'front'],
    'policy_class': 'ACT',
    'temporal_agg': False
}

# training config
TRAIN_CONFIG = {
    'seed': 42,
    'num_epochs':20000,
    'batch_size_val': 8,
    'batch_size_train': 8,
    'eval_ckpt_name': 'policy_last.ckpt',
    'checkpoint_dir': CHECKPOINT_DIR
}
