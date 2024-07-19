import cv2
import h5py

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
                print(f"Frame {i+1}: Joint Positions: {joint_positions[i]}")
            else:
                print(f"Frame {i+1}: No joint position data available.")
            
            # Wait for the specified interval
            key = cv2.waitKey(interval)
            if key == 27:  
                break

        
        cv2.destroyAllWindows()


if __name__ == '__main__':

    episode_number = input("Episode Number to replay: ")
    # hdf5_path = f'/home/navaneet/Desktop/ACTfranka/ACTfranka/datasets/main/processed/episode_{episode_number}.hdf5'
    hdf5_path = f'/home/navaneet/Desktop/ACTfranka/ACTfranka/real_dir2/episode_0.hdf5'

    
    group_name = 'observations'
    images_dataset_name = 'images/top'
    joint_positions_dataset_name = 'qpos'
    frame_rate = 30  
    play_images_from_hdf5_opencv(hdf5_path, group_name, images_dataset_name, joint_positions_dataset_name, frame_rate)
