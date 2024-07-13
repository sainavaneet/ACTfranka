import pandas as pd
from controler.robot_state import *
import rospy
from settings.var import *
from record_episodes import *

from panda_kinematics import PandaWithPumpKinematics
franka = RobotController()
kinematics = PandaWithPumpKinematics()
franka.inital_pose()

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


# set_box_position(0.4 , -0.25 , BOX_Z)


file_path = 'actions.csv'
data = pd.read_csv(file_path, header=None, skiprows=1)

# Flag to track if the gripper has entered the > 0.9 loop
gripper_entered_high_position = False

for index, row in data.iterrows():
    if index % 25 == 0:
        joint_positions = row[:7].tolist()
        gripper_position = row[7]
        franka.move_to_joint_position(joint_positions)

        # Check the flag before executing gripper commands
        if not gripper_entered_high_position:
            if gripper_position > 0.9:
                franka.grasp(0.025, 25)
                
                gripper_entered_high_position = True
            else:
                franka.exec_gripper_cmd(0.08, GRIPPER_FORCE)


franka.exec_gripper_cmd(0.08)

joint_positions = np.zeros(7)
pos = np.array([0.44, 0.21, 0.17])
quat = np.array([np.pi, np.pi/2, 0.0, 0.0])
solution = kinematics.ik(joint_positions, pos, quat)
franka.move_to_joint_position(solution)
franka.inital_pose()