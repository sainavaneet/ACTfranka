import cv2
import os
import h5py
import numpy as np
from controller.robot_state import *

class CameraController:
    def __init__(self, camera_index=0):
        self.capture = cv2.VideoCapture(camera_index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.data = []
        self.robot_state_data = []
        self.recording = False
        self.franka = RobotController()

    def capture_frames(self):
        print("Press 's' to start/stop recording. Press 'q' to quit.")
        while True:
            ret, frame = self.capture.read()
            if ret:
                cv2.imshow('Camera Feed', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    self.recording = not self.recording
                    if self.recording:
                        print("Recording started.")
                    else:
                        print("Recording stopped. Saving data...")
                        self.record_extra_frames(10) 
                        self.save_data()
                        self.data = []
                        self.robot_state_data = []
                elif key == ord('q'):
                    break
                if self.recording:
                    self.data.append(frame)
                    robot_state = self.get_robot_state(0)  # Initial state with '0'
                    self.robot_state_data.append(robot_state)
        cv2.destroyAllWindows()

    def get_robot_state(self, end_marker=0):
        angles = self.franka.angles()
        return np.concatenate((angles, [end_marker]))

    def record_extra_frames(self, count):
        last_frame = self.data[-1]
        last_state = self.get_robot_state(1)  # Final state with '1'
        for _ in range(count):
            self.data.append(last_frame)
            self.robot_state_data.append(last_state)

    def save_data(self):
        if not self.data:
            print("No data to save.")
            return

        episode_idx = 0
        directory = "real_dir2"
        if not os.path.exists(directory):
            os.makedirs(directory)
        while os.path.exists(os.path.join(directory, f'episode_{episode_idx}.hdf5')):
            episode_idx += 1
        file_path = os.path.join(directory, f'episode_{episode_idx}.hdf5')
        with h5py.File(file_path, 'w') as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            images = obs.create_group('images')
            camera_names = ['top']
            for cam_name, data in zip(camera_names, [self.data]):
                image_data = np.array(data, dtype='uint8')
                images.create_dataset(cam_name, data=image_data, dtype='uint8', chunks=(1, 480, 640, 3))
            robot_data = np.array(self.robot_state_data, dtype='float64')
            obs.create_dataset('qpos', data=robot_data)
            root.create_dataset('action', data=robot_data)

def main():
    camera_index = 0
    camera_controller = CameraController(camera_index)
    camera_controller.capture_frames()

if __name__ == '__main__':
    main()
