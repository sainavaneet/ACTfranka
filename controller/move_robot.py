#!/usr/bin/env python3
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np
from panda_kinematics import PandaWithPumpKinematics
from robot_state import *
import time
def perform_trajectory():
    
    
    robot = RobotController()
    robot.exec_gripper_cmd(0.08 ,0.5)

    robot.initial_pose()
    # robot.exec_gripper_cmd(0.08 ,0.5)
    kinematics = PandaWithPumpKinematics()
    current_position = robot.angles()
    # print(current_position)
    position = np.array([0.7232432291839545, -0.1696193286799512, 0.017792108035412868])
    orientation_quat = np.array([0.9980894386124141, -0.016944793950549396, -0.0018080098350337212, 0.059341709313117705])
    initial_joint_positions = np.array([ 0,0,0,0,0, 0.0, 0.0])
    # solution = kinematics.ik(initial_joint_positions, position, orientation_quat)
    # print(f"SOLUTION : {solution}")
    # robot.move_to_joint_position(solution)

    # robot.exec_gripper_cmd(0.04 ,0.5)



if __name__ == '__main__':
    perform_trajectory()
