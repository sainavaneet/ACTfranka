import h5py

def explore_hdf5_group(group, indent=0):
    """ Recursively explore groups and datasets in the HDF5 file """
    indent_str = '    ' * indent  
    try:
        for key in group:
            item = group[key]
            print(f"{indent_str}{key}: {item}")
            if isinstance(item, h5py.Dataset):
                
                print(f"{indent_str}  - Shape: {item.shape}, Type: {item.dtype}")
                
                #print(f"{indent_str}  Data: {item[...]}") to see the data
            elif isinstance(item, h5py.Group):
                explore_hdf5_group(item, indent + 1)
    except AttributeError:
        print(f"{indent_str}Error: Object {group} is not a group or dataset")

# Open the HDF5 file
file_path = '/home/navaneet/Desktop/ACTfranka/ACTfranka/real_dir2/processed/episode_0.hdf5'
with h5py.File(file_path, 'r') as file:
    print("Contents of the HDF5 file:")
    explore_hdf5_group(file)
