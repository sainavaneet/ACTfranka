
# ACT Imitation Learning Framework for Franka Robot

This repository provides a comprehensive guide to using the ACT imitation learning framework with the Franka robot in both simulated and real-world environments. The codebase includes utilities for environment setup, training, and inference.

## Prerequisites
- Ubuntu 20.04
- ROS Noetic
- `libfranka` package for Franka robot control

## Installation
Clone the repository to your local machine:
```bash
git clone https://github.com/sainavaneet/ACTfranka.git
cd ACTfranka
```

## Project Structure
- `controller/`: Robot control code modules including movement and state management.
- `settings/`: Configuration files including dataset paths and hyperparameters.
- `simulation/`: Simulation scripts for recording episodes and evaluating models.
- `train.py`: Script to train the policy using ACT imitation learning.
- `real_robot/`: Scripts tailored for deploying the model on a real Franka robot.

## Step 1: Create the Environment
1. Set up a simulated environment in Gazebo using the Franka robot and the `libfranka` package.
2. Record episodes with the script located at `simulation/record_episodes.py`.
   - Ensure the dataset path is correctly set in `settings/var.py`.

### Dataset Format
The dataset should be in HDF5 format with the following structure:
```
HDF5 file contents:
- action: <HDF5 dataset "action": shape (149, 8), type "<f8">
- observations: <HDF5 group "/observations" (2 members)>
  - images:
    - top: <HDF5 dataset "top": shape (149, 480, 640, 3), type "|u1">
  - qpos: <HDF5 dataset "qpos": shape (149, 8), type "<f8">
```

### Replay Episodes
Use the Jupyter notebook `dataset_prepare/replay.ipynb` to replay recorded episodes by specifying the episode path.

## Step 2: Train the Model
1. Configure the necessary hyperparameters in `settings/var.py`.
2. Train the model using the `train.py` script with the prepared dataset to generate the policy.

## Step 3: Model Inference
1. Once training is complete, load and evaluate the trained policy using the script `simulation/evaluate.py`.
2. This step simulates how the Franka robot will perform the learned tasks in a controlled environment.

## Real Robot Deployment
To deploy on a real Franka robot, navigate to the `real_robot` directory. This directory contains scripts specifically adapted for real-world operations of the Franka robot.

## Support
For any issues or further questions, please open an issue on the GitHub repository.
