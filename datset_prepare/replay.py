import cv2
import h5py

def play_images_from_hdf5_opencv(file_path, group_name, dataset_name, joint_positions_dataset, frame_rate=30):
    # Calculate the pause interval between frames in milliseconds
    interval = int(1000 / frame_rate)
    
    # Open the HDF5 file
    with h5py.File(file_path, 'r') as file:
        # Navigate to the specific group and dataset for images
        images_dataset = file[group_name][dataset_name]
        # Access the joint positions dataset
        joint_positions = file[group_name][joint_positions_dataset]

        # Create a window to display the images
        cv2.namedWindow('HDF5 Image Sequence', cv2.WINDOW_NORMAL)
        
        # Iterate through the dataset
        for i in range(images_dataset.shape[0]):
            image = images_dataset[i, ...]
            if image.dtype != 'uint8':
                image = image.astype('uint8')

            # Display the image using OpenCV
            cv2.imshow('HDF5 Image Sequence', image)
            
            # Print the corresponding joint positions
            if i < len(joint_positions):
                print(f"Frame {i+1}: Joint Positions: {joint_positions[i]}")
            else:
                print(f"Frame {i+1}: No joint position data available.")
            
            # Wait for the specified interval
            key = cv2.waitKey(interval)
            if key == 27:  # Exit if the 'ESC' key is pressed
                break

        # Destroy the window after displaying all images
        cv2.destroyAllWindows()

# Usage example
if __name__ == '__main__':
    hdf5_path = '/home/navaneet/Desktop/ACTfranka/ACTfranka/datasets/main/processed/episode_25.hdf5'
    group_name = 'observations'
    images_dataset_name = 'images/top'
    joint_positions_dataset_name = 'qpos'
    frame_rate = 30  
    play_images_from_hdf5_opencv(hdf5_path, group_name, images_dataset_name, joint_positions_dataset_name, frame_rate)
