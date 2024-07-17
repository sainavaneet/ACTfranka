import pandas as pd
import numpy as np
import random
import rospy
from controller.robot_state import RobotController
from panda_kinematics import PandaWithPumpKinematics
from settings.var import GRIPPER_FORCE, BOX_Z
from std_msgs.msg import String
from gazebo_msgs.srv import SetModelState, SetModelStateRequest
from rich.progress import Progress
from place_traj import *
import progressbar
class Simulator:
    def __init__(self, file_path='actions.csv'):
        
        self.franka = RobotController()
        self.kinematics = PandaWithPumpKinematics()
        self.data = pd.read_csv(file_path, header=None, skiprows=1)
        self.gripper_grasp_position = False
        self.franka.initial_pose()
        self.place = PlaceTraj()



    def simulate(self):
        with Progress() as progress:
            task = progress.add_task("[green]Simulating...", total=len(self.data))
            
            for index, row in self.data.iterrows():
                
                progress.update(task, advance=1)
            
                if index % 30 == 0:
                    
                    joint_positions = row[:7].tolist()
                    gripper_position = row[7]
                    
                   
                    self.franka.move_to_joint_position(joint_positions)

                    if not self.gripper_grasp_position:
                        if gripper_position > 0.9:
                            self.franka.exec_gripper_cmd(0.027, 0.8)
                            self.gripper_grasp_position = True
                        else:
                            self.franka.exec_gripper_cmd(0.08, 1)
        self.place.place_simulate()
            


if __name__ == "__main__":
    try:
    
        simulator = Simulator(file_path='actions.csv')  
        simulator.simulate()
    
    except rospy.ROSInterruptException:
        pass