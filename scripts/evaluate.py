import torch
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import pickle
import pandas as pd
from panda_robot import PandaArm
from training.utils import *
from settings.var import *
from tqdm import tqdm
import time
# Image processing and handling
bridge = CvBridge()

def image_callback(msg):
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")  # Convert ROS Image message to OpenCV image (BGR format)
    except CvBridgeError as e:
        rospy.logerr(e)
    return cv_image

def capture_image():
    msg = rospy.wait_for_message('/camera/color/image_raw', Image, timeout=10)  # Added timeout for reliability
    return image_callback(msg)

# Configuration
cfg = TASK_CONFIG
policy_config = POLICY_CONFIG
train_cfg = TRAIN_CONFIG
device = torch.device('cuda')

if __name__ == "__main__":
    rospy.init_node('panda_robot_controller', anonymous=True)

    # Initialize the PandaArm
    panda = PandaArm()
    panda.move_to_neutral()

    # Load the policy
    ckpt_path = os.path.join(train_cfg['checkpoint_dir'], train_cfg['eval_ckpt_name'])
    policy = make_policy(policy_config['policy_class'], policy_config)
    policy.load_state_dict(torch.load(ckpt_path, map_location=device))
    policy.to(device)
    policy.eval()
    rospy.loginfo(f'Policy loaded from {ckpt_path}')

    stats_path = os.path.join(train_cfg['checkpoint_dir'], 'dataset_stats.pkl')

    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda pos: (pos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda act: act * stats['action_std'] + stats['action_mean']



    query_frequency = policy_config['num_queries']


    if policy_config['temporal_agg']:
        query_frequency = 1
        num_queries = policy_config['num_queries']


    position = panda.angles()
    gripper_width = panda.gripper_state()['position'][0]

# Assign the modified position list to pos
    pos = np.append(position, gripper_width)


    obs = {
        'qpos': pos,
        'images': {cn: capture_image() for cn in cfg['camera_names']}
    }

    n_rollouts = 1  

    count_loop = 0

    for i in range(n_rollouts):
        ### evaluation loop
        if policy_config['temporal_agg']:
            all_time_actions = torch.zeros([cfg['episode_len'], cfg['episode_len']+num_queries, cfg['state_dim']]).to(device)
        qpos_history = torch.zeros((1, cfg['episode_len'], cfg['state_dim'])).to(device)
        with torch.inference_mode():
             # init buffers
            obs_replay = []
            action_replay = []
            
            for t in tqdm(range(cfg['episode_len'])):
                count_loop = t+1
                
                print(f"torch type {type(obs['qpos'])}")

                if isinstance(obs['qpos'], np.ndarray):
                    print("entered the ndarray function")
                    qpos_numpy = obs['qpos']
                elif isinstance(obs['qpos'], torch.Tensor):
                    print("entered the tensor function")
                    print(obs['qpos'].dim())
                    qpos_numpy = obs['qpos'].cpu().numpy()
                else:
                    print("Unhandled data type for qpos:", type(obs['qpos']))



                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().to(device).unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(obs['images'], cfg['camera_names'], device)

                if t % query_frequency == 0:
                    all_actions = policy(qpos, curr_image)
                if policy_config['temporal_agg']:
                    all_time_actions[[t], t:t+num_queries] = all_actions
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights.astype(np.float32)).to(device).unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = all_actions[:, t % query_frequency]
                    

            



                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                # action = pos2pwm(action).astype(int)
                ### take action
                print(action)
                pos = action[:7]

                gripper_state = action[-1]
                panda.move_to_joint_position(pos)
                panda.exec_gripper_cmd(gripper_state , 0.1004)
                
                pos = np.append(position, gripper_width)

                obs = {
                    'qpos': pos,
                    'images': {cn: capture_image() for cn in cfg['camera_names']}
                }

                # ### store data
                # obs_replay.append(obs)
                # action_replay.append(action)
                print(f'loop number = {count_loop}')


