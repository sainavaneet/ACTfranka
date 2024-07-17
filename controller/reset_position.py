#!/usr/bin/env python3
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint 
import sys

def perform_trajectory():
    rospy.init_node('initial_position')
    controller_name = '/position_joint_trajectory_controller/command'
    trajectory_publisher = rospy.Publisher(controller_name, JointTrajectory, queue_size=10)
    argv = sys.argv[1:]
    fr3_joints = ['fr3_joint1', 
                    'fr3_joint2', 
                    'fr3_joint3', 
                    'fr3_joint4', 
                    'fr3_joint5',
                    'fr3_joint6', 
                    'fr3_joint7']
    default_positions = [0, 
                         -0.785398163, 
                         0, 
                         -2.35619449, 
                         0, 
                         1.57079632679, 
                         0.785398163397]
    
    goal_positions = [float(arg) if idx < len(argv) else default_positions[idx] for idx, arg in enumerate(default_positions)]
    
    rospy.loginfo("-----------------Resetting the position-------------- ")
    rospy.sleep(1)

    trajectory_msg = JointTrajectory()
    trajectory_msg.joint_names = fr3_joints
    trajectory_msg.points.append(JointTrajectoryPoint())
    trajectory_msg.points[0].positions = goal_positions
    trajectory_msg.points[0].velocities = [0.0 for _ in fr3_joints] 
    trajectory_msg.points[0].accelerations = [0.0 for _ in fr3_joints]
    trajectory_msg.points[0].time_from_start = rospy.Duration(5)
    rospy.sleep(1)

    trajectory_publisher.publish(trajectory_msg)

if __name__ == '__main__':
    perform_trajectory()