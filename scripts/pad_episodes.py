import h5py
import numpy as np
import os
from tqdm import tqdm

def find_maximum_dataset_length(source_directory, dataset_path):
    """ Find the maximum length of a dataset across all files. """
    max_length = 0
    files = [f for f in os.listdir(source_directory) if f.endswith('.hdf5')]
    for filename in tqdm(files, desc="Analyzing dataset lengths"):
        file_path = os.path.join(source_directory, filename)
        with h5py.File(file_path, 'r') as file:
            if dataset_path in file:
                data_length = len(file[dataset_path])
                if data_length > max_length:
                    max_length = data_length
    return max_length

def pad_or_truncate(data, target_length, pad_value=None):
    """ Pad or truncate the array to the target length. """
    if len(data) > target_length:
        return data[:target_length]
    elif len(data) < target_length:
        padding = [data[-1] if pad_value is None else pad_value] * (target_length - len(data))
        return np.concatenate([data, padding])
    return data

def process_and_save_datasets(source_directory, target_directory, target_length):
    """ Process all HDF5 files, truncate or pad datasets as needed, and save them. """
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)  # Create target directory if it doesn't exist
    
    files = [f for f in os.listdir(source_directory) if f.endswith('.hdf5')]
    for filename in tqdm(files, desc="Processing files"):
        source_file_path = os.path.join(source_directory, filename)
        new_file_path = os.path.join(target_directory, f"{filename}")
        with h5py.File(source_file_path, 'r') as file, h5py.File(new_file_path, 'w') as new_file:
            # Assuming data structure based on your description, adjust as needed
            root = new_file
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            images = obs.create_group('images')
            
            # Handle image data
            image_data = np.array(file['observations/images/top'])
            image_data = pad_or_truncate(image_data, target_length)
            images.create_dataset('top', data=image_data, dtype='uint8', chunks=(1, 480, 640, 3))
            
            # Handle joint positions
            qpos = np.array(file['observations/qpos'])
            qpos = pad_or_truncate(qpos, target_length)
            obs.create_dataset('qpos', data=qpos)
            
            # Handle box positions without padding
            box_positions = np.array(file['observations/box_positions'])
            obs.create_dataset('box_positions', data=box_positions)
            
            # Handle action data, same as joint positions
            action_data = qpos
            root.create_dataset('action', data=action_data)

            # tqdm.write(f"Processed and saved: {new_file_path}")

# Directories
source_directory = '/home/navaneet/Desktop/ACTPanda/datasets/main'
target_directory = '/home/navaneet/Desktop/ACTPanda/datasets/main2/processed/'

# Find the maximum dataset length (excluding 'box_positions')
target_length = find_maximum_dataset_length(source_directory, 'observations/qpos')

print(f"Target length determined: {target_length}")

# Process all files using the found target length
process_and_save_datasets(source_directory, target_directory, target_length)
