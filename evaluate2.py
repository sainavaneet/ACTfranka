import torch
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import pickle
import pandas as pd
from training.utils import *
from settings.var import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG
from tqdm import tqdm
import time
from gazebo_msgs.srv import SetModelState, SetModelStateRequest 
from controler.robot_state import *
import random
from std_srvs.srv import Empty

from settings.var import *
bridge = CvBridge()


import subprocess

def set_box_position(x, y, z):
        rospy.wait_for_service('/gazebo/set_model_state')
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        state_msg = SetModelStateRequest()
        state_msg.model_state.model_name = 'stone'
        state_msg.model_state.pose.position.x = x
        state_msg.model_state.pose.position.y = y
        state_msg.model_state.pose.position.z = z
        set_state(state_msg)

def generate_cordinate():
        box_length = 0.18
        box_width = 0.11
        box_x_center = 0.45
        box_y_center = -0.21
        cube_x = 0.025
        cube_y = 0.032
        min_x = box_x_center - box_length / 2 + cube_x / 2
        max_x = box_x_center + box_length / 2 - cube_x / 2
        min_y = box_y_center - box_width / 2 + cube_y / 2
        max_y = box_y_center + box_width / 2 - cube_y / 2
        
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)

        return x , y


set_box_position(0.46 , -0.25 , BOX_Z)


def image_callback(msg):
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")  # Convert ROS Image message to OpenCV image (BGR format)
    except CvBridgeError as e:
        rospy.logerr(e)
    return cv_image

def capture_image(camera_name):
    camera_topics = {
        'top': '/fr3/camera/image_raw',
        'front': '/fr3/camera2/image_raw'
    }
    topic = camera_topics.get(camera_name, '/fr3/camera/color/image_raw')  # default to 'top' if not found
    msg = rospy.wait_for_message(topic, Image, timeout=10)
    return image_callback(msg)


# Configuration
cfg = TASK_CONFIG
policy_config = POLICY_CONFIG
train_cfg = TRAIN_CONFIG
device = torch.device('cuda')

if __name__ == "__main__":
    franka = RobotController()
    franka.inital_pose()


    # box_x , box_y = generate_cordinate()

    
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
        query_frequency = 20
        print("entered")
        num_queries = policy_config['num_queries']


    position = franka.angles()

    gripper_width = franka.gripper_state()

    if gripper_width > 0.04:
        gripper_width = 0
    else:
        gripper_width = 1


# Assign the modified position list to pos
    pos = np.append(position, gripper_width)


    obs = {
        'qpos': pos,
        'images': {cn: capture_image(cn) for cn in cfg['camera_names']}
    }

    n_rollouts = 1

    count_loop = 0

    for i in tqdm(range(n_rollouts) , desc=f"rollout : {n_rollouts}"):
        ### evaluation loop
        if policy_config['temporal_agg']:
            all_time_actions = torch.zeros([cfg['episode_len'], cfg['episode_len']+num_queries, cfg['state_dim']]).to(device)
        qpos_history = torch.zeros((1, cfg['episode_len'], cfg['state_dim'])).to(device)
        with torch.inference_mode():
             # init buffers
            obs_replay = []
            action_replay = []
            action_list = []
            
            for t in range(cfg['episode_len']):
                count_loop = t+1
                
                # print(f"torch type {type(obs['qpos'])}")

                if isinstance(obs['qpos'], np.ndarray):
                   
                    qpos_numpy = obs['qpos']
                elif isinstance(obs['qpos'], torch.Tensor):
                    
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
                action_list.append(action.tolist())  # Convert numpy array to list before appending
                # print(f'loop number = {count_loop}')


    # After loop ends, write actions to a file
            df = pd.DataFrame(action_list, columns=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "gripper"])
            df.to_csv('actions.csv', index=False)  # Replace 'path_to_save_file' with your desired path

            # rospy.loginfo("Actions have been saved to file.")



