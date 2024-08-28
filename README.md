
# ACT Imitation Learning Framework for Franka Robot

**Welcome to the ACT imitation learning framework repository, designed for the Franka robot. This guide covers both simulated and real-world environment setups, and includes utilities for environment setup, training, and inference.**

We have modified the original ACT code from [this repo](https://github.com/tonyzhaozh/act.git) to complete this project, enhancing its capabilities to better suit our specific application needs.


> **website** - https://sainavaneet.github.io/ACTfranka.github.io/

## 📋 Prerequisites
Ensure you have the following installed:
- **Ubuntu 20.04**
- **ROS Noetic**
- **`libfranka`** package for Franka robot control

## 🚀 Installation
To get started with the ACT imitation learning framework, follow these steps:

```bash
git clone https://github.com/sainavaneet/ACTfranka.git
cd ACTfranka
```

## 🗂 Project Structure
- `controller/`: Contains robot control code modules including movement and state management.
- `settings/`: Houses configuration files for dataset paths and hyperparameters.
- `simulation/`: Scripts for recording episodes and evaluating models are here.
- `train.py`: The main script to train the policy using ACT imitation learning.
- `real_robot/`: Specialized scripts for deploying the model on an actual Franka robot.

## 🏗 Step 1: Create the Environment
1. **Setup a simulated environment** in Gazebo using the Franka robot and the `libfranka` package.
2. **Record episodes** using the script located at `simulation/record_episodes.py`.
   - Make sure the dataset path is correctly set in `settings/var.py`.

### Dataset Format
The dataset should be structured in HDF5 format as follows:
```
HDF5 file contents:
- action: <HDF5 dataset "action": shape (149, 8), type "<f8">
- observations:
  - images:
    - top: <HDF5 dataset "top": shape (149, 480, 640, 3), type "|u1">
  - qpos: <HDF5 dataset "qpos": shape (149, 8), type "<f8">
```

### Replay Episodes
Utilize the Jupyter notebook `dataset_prepare/replay.ipynb` to replay recorded episodes by specifying the episode path.

## 🏋️ Step 2: Train the Model
1. Configure the necessary hyperparameters in `settings/var.py`.
2. Execute the `train.py` script with the prepared dataset to generate the policy.

## 🤖 Step 3: Model Inference
1. **Load and evaluate the trained policy** using the script `simulation/evaluate.py`.
2. This process simulates how the Franka robot will perform the learned tasks in a controlled environment.

## 🌍 Real Robot Deployment
To deploy on a real Franka robot, navigate to the `real_robot` directory. Scripts here are specifically adapted for real-world operations of the Franka robot.

## 🆘 Support
For any issues or further questions, please open an issue on the [GitHub repository](https://github.com/sainavaneet/ACTfranka).
```
