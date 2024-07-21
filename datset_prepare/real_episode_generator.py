import h5py
import numpy as np
import os
from tqdm import tqdm

def find_maximum_dataset_length(source_directory, dataset_path):
    """ Find the maximum length of a dataset across all files after removing every second element. """
    max_length = 0
    files = [f for f in os.listdir(source_directory) if f.endswith('.hdf5')]
    for filename in tqdm(files, desc="Analyzing dataset lengths"):
        file_path = os.path.join(source_directory, filename)
        with h5py.File(file_path, 'r') as file:
            if dataset_path in file:
                data = np.array(file[dataset_path])[::2]  # Skip every second timestep
                data_length = len(data)
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
    """ Process all HDF5 files, truncate or pad datasets as needed, and save them after reducing frame rate. """
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)  
    
    files = [f for f in os.listdir(source_directory) if f.endswith('.hdf5')]
    for filename in tqdm(files, desc="Processing files"):
        source_file_path = os.path.join(source_directory, filename)
        new_file_path = os.path.join(target_directory, f"{filename}")
        with h5py.File(source_file_path, 'r') as file, h5py.File(new_file_path, 'w') as new_file:
            root = new_file
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            images = obs.create_group('images')
            
           
            if 'observations/images/top' in file:
                top_image_data = np.array(file['observations/images/top'])[::2]
                top_image_data = pad_or_truncate(top_image_data, target_length)
                images.create_dataset('top', data=top_image_data, dtype='uint8', chunks=(1, 480, 640, 3))
            
           
            # if 'observations/images/front' in file:
            #     front_image_data = np.array(file['observations/images/front'])[::2]
            #     front_image_data = pad_or_truncate(front_image_data, target_length)
            #     images.create_dataset('front', data=front_image_data, dtype='uint8', chunks=(1, 480, 640, 3))
            
            qpos = np.array(file['observations/qpos'])[::2]
            qpos = pad_or_truncate(qpos, target_length)
            obs.create_dataset('qpos', data=qpos)

            action_data = qpos
            root.create_dataset('action', data=action_data)


source_directory = '/home/navaneet/Desktop/ACTfranka/ACTfranka/real_dir2/'
target_directory = '/home/navaneet/Desktop/ACTfranka/ACTfranka/real_dir2/processed/'
target_length = find_maximum_dataset_length(source_directory, 'observations/qpos')
print(f"Target length determined: {target_length}")
process_and_save_datasets(source_directory, target_directory, target_length)
