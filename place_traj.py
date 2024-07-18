import pandas as pd
import numpy as np
import random
import rospy
from controller.robot_state import *

from std_msgs.msg import String

from rich.progress import Progress

import progressbar

class PlaceTraj:
    def __init__(self, file_path='csv/place_traj.csv'):
        
        self.franka = RobotController()
        self.data = pd.read_csv(file_path, header=None)
        self.gripper_grasp_position = False



    def place_simulate(self):
        with Progress() as progress:
            task = progress.add_task("[green]Simulating...", total=len(self.data))
            
            for index, row in self.data.iterrows():
                
                progress.update(task, advance=1)
            
                if index % 1 == 0:
                    
                    joint_positions = row[:7].tolist()
                    gripper_position = row[7]
                    
                   
                    self.franka.move_to_joint_position(joint_positions)

                    if not self.gripper_grasp_position:
                        if gripper_position > 0.9:
                            self.franka.exec_gripper_cmd(0.027, 0.8)
                            self.gripper_grasp_position = True
                        else:
                            self.franka.exec_gripper_cmd(0.08, 1)
        self.franka.exec_gripper_cmd(0.08, 1)






if __name__ == "__main__":
    try:
    
        simulator = PlaceTraj(file_path='csv/place_traj.csv')  
        simulator.simulate()
    
    except rospy.ROSInterruptException:
        pass