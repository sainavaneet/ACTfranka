#!/usr/bin/env python3
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np
from panda_kinematics import PandaWithPumpKinematics
from robot_state import *
import time
def perform_trajectory():
    
    
    robot = RobotController()
    kinematics = PandaWithPumpKinematics()
    current_position = robot.angles()
    print(current_position)
    position = np.array([0.4,0.225, 0.2])
    orientation_quat = np.array([ np.pi, np.pi/2 ,0, 0])
    initial_joint_positions = np.array([ 0,0,0,0,0, 0.0, 0.0])
    solution = kinematics.ik(initial_joint_positions, position, orientation_quat)
    print(f"SOLUTION : {solution}")
    robot.move_to_joint_position(solution)

    robot.exec_gripper_cmd(0.08 ,0.5)

    # time.sleep(2)
    # robot.grasp(0.03 ,6)


if __name__ == '__main__':
    perform_trajectory()
