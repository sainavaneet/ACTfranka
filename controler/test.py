import rospy
from robot_state import RobotController
import time

def main():
   
    robot = RobotController()

    angles = [            1, 
                         -0.785398163, 
                         0, 
                         -2.35619449, 
                         0, 
                         1.57079632679, 
                         0.785398163397]  # Example angles
    # robot.move_to_joint_position(angles)

    time.sleep(1)  

    current_positions = robot.gripper_state()
    if current_positions:
        print("Current joint positions:", current_positions)
    else:
        print("Joint positions not yet received.")

if __name__ == '__main__':
    main()
