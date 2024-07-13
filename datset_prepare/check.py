import h5py

def explore_hdf5_group(group, indent=0):
    """ Recursively explore groups and datasets in the HDF5 file """
    indent_str = '    ' * indent  # Indentation for pretty printing
    try:
        for key in group:
            item = group[key]
            print(f"{indent_str}{key}: {item}")
            if isinstance(item, h5py.Dataset):
                # Print dataset shape and datatype; comment the next line to print dataset contents
                print(f"{indent_str}  - Shape: {item.shape}, Type: {item.dtype}")
                # Uncomment the following line to print actual data (might be memory intensive)
                # print(f"{indent_str}  Data: {item[...]}")
            elif isinstance(item, h5py.Group):
                explore_hdf5_group(item, indent + 1)
    except AttributeError:
        print(f"{indent_str}Error: Object {group} is not a group or dataset")

# Open the HDF5 file
file_path = '/home/navaneet/Desktop/ACTfranka/ACTfranka/datasets/main/processed/episode_10.hdf5'
with h5py.File(file_path, 'r') as file:
    print("Contents of the HDF5 file:")
    explore_hdf5_group(file)
