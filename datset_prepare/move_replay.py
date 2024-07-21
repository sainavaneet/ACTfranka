import cv2
import h5py

from controller.robot_state import RobotController

franka = RobotController()
franka.initial_pose()

def play_images_from_hdf5_opencv(file_path, group_name, dataset_name, joint_positions_dataset, frame_rate=30):
    interval = int(1000 / frame_rate)
    
    # Open the HDF5 file
    with h5py.File(file_path, 'r') as file:
        images_dataset = file[group_name][dataset_name]
        joint_positions = file[group_name][joint_positions_dataset]

        cv2.namedWindow('HDF5 Image Sequence', cv2.WINDOW_NORMAL)
        
        for i in range(images_dataset.shape[0]):
            image = images_dataset[i, ...]
            if image.dtype != 'uint8':
                image = image.astype('uint8')

            cv2.imshow('HDF5 Image Sequence', image)
            
            if i < len(joint_positions):
                joint_pos = joint_positions[i]
                print(f"Frame {i+1}: Joint Positions: {joint_pos}")
                
                franka.move_to_joint_position(joint_pos[:7]) 
                if joint_pos[7] == 1:
                    franka.exec_gripper_cmd(0.055 , 1)
                elif joint_pos[7] == 0:
                    franka.exec_gripper_cmd(0.08 , 1)
                    
            else:
                print(f"Frame {i+1}: No joint position data available.")
            
            # Wait for the specified interval
            key = cv2.waitKey(interval)
            if key == 27:  # Escape key to exit
                break
        
        cv2.destroyAllWindows()

if __name__ == '__main__':
    hdf5_path = f'/home/navaneet/Desktop/ACTfranka/ACTfranka/real_dir2/processed2/episode_5.hdf5'
    group_name = 'observations'
    images_dataset_name = 'images/front'
    joint_positions_dataset_name = 'qpos'
    frame_rate = 30  
    play_images_from_hdf5_opencv(hdf5_path, group_name, images_dataset_name, joint_positions_dataset_name, frame_rate)
