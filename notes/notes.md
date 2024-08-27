To control the robot we created a control code which is present in here'ACTfranka/controller/robot_state.py'

* created some important functions like move , gripper movement , get states etc



To perform the task using ACT imitation learning follow these steps

1. Create an Environment to record our actions for imiation learning 
2. We do training using our created dataset with using act original code ()
3. we do inference.


1st step 

create an environment in Gazebo using Franka robot with "libfranka" package

after that you can record episodes using ACTfranka/simulation/record_episodes.py this file **note the dataset path should be set in this file  "ACTfranka/settings/var.py" **


* The dataset should be exactly in this format and in hdf5 format

`Contents of the HDF5 file:
action: <HDF5 dataset "action": shape (149, 8), type "<f8">
  - Shape: (149, 8), Type: float64
observations: <HDF5 group "/observations" (2 members)>
    images: <HDF5 group "/observations/images" (1 members)>
        top: <HDF5 dataset "top": shape (149, 480, 640, 3), type "|u1">
          - Shape: (149, 480, 640, 3), Type: uint8
    qpos: <HDF5 dataset "qpos": shape (149, 8), type "<f8">
      - Shape: (149, 8), Type: float64`



* You can reply the episode by using 'ACTfranka/datset_prepare/replay.ipynb' this file by passing the episode path

2nd step 

* For training the act code is important because we modified that code that can used for franka

* Primarly we need to assing the hyper parameters in "ACTfranka/settings/var.py" and then by using the datsets 

we can train and generate the policy using "ACTfranka/train.py"

3rd Step 

* Once the training is done we do the inference for franka robot using the policy that we trained 

using the policy with this file "ACTfranka/simulation/evaluate.py"




To perform this on the real robot we created a seperate folder which is "ACTfranka/real_robot"

where you will find all the required files



this is aREdme file for github can you properly write it