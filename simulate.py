import pandas as pd
import numpy as np
import random
import rospy
from controler.robot_state import RobotController
from panda_kinematics import PandaWithPumpKinematics
from settings.var import GRIPPER_FORCE, BOX_Z
from std_msgs.msg import String
from gazebo_msgs.srv import SetModelState, SetModelStateRequest


class Simulator:
    def __init__(self, file_path='actions.csv'):
        
        if not rospy.core.is_initialized():
            rospy.init_node('simulator_node', anonymous=True)

        self.franka = RobotController()
        self.kinematics = PandaWithPumpKinematics()
        self.data = pd.read_csv(file_path, header=None, skiprows=1)
        self.gripper_entered_high_position = False
        self.franka.initial_pose()

    def set_box_position(self, x, y, z):
        rospy.wait_for_service('/gazebo/set_model_state')
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        state_msg = SetModelStateRequest()
        state_msg.model_state.model_name = 'stone'
        state_msg.model_state.pose.position.x = x
        state_msg.model_state.pose.position.y = y
        state_msg.model_state.pose.position.z = z
        set_state(state_msg)

    def generate_coordinate(self):
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
        return x, y

    def simulate(self):
        for index, row in self.data.iterrows():
            if index % 25 == 0:
                joint_positions = row[:7].tolist()
                gripper_position = row[7]
                self.franka.move_to_joint_position(joint_positions)

                if not self.gripper_entered_high_position:
                    if gripper_position > 0.9:
                        self.franka.grasp(0.025, 15)
                        self.gripper_entered_high_position = True
                    else:
                        self.franka.exec_gripper_cmd(0.08, GRIPPER_FORCE)

        self.franka.exec_gripper_cmd(0.08)
        joint_positions = np.zeros(7)
        pos = np.array([0.44, 0.21, 0.17])
        quat = np.array([np.pi, np.pi/2, 0.0, 0.0])
        solution = self.kinematics.ik(joint_positions, pos, quat)
        self.franka.move_to_joint_position(solution)
        self.franka.initial_pose()
