import cv2
import os
import h5py
import numpy as np
from controller.robot_state import *

class CameraController:
    def __init__(self, camera_indices=[0, 2]):
        self.captures = [cv2.VideoCapture(index) for index in camera_indices]
        for capture in self.captures:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.data = [[] for _ in camera_indices]  # Create a list of lists for each camera
        self.robot_state_data = []
        self.recording = False
        self.franka = RobotController()

    def capture_frames(self):
        print("Press 's' to start/stop recording. Press 'q' to quit.")
        cv2.namedWindow("Camera Feed")  # Ensure a named window is created
        while True:
            frames = []
            for capture in self.captures:
                ret, frame = capture.read()
                if ret:
                    cv2.imshow(f'Camera Feed {self.captures.index(capture)}', frame)
                    frames.append(frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                self.recording = not self.recording
                if self.recording:
                    print("Recording started.")
                else:
                    print("Recording stopped. Saving data...")
                    self.save_data()
                    self.data = [[] for _ in self.captures]
                    self.robot_state_data = []
            elif key == ord('q'):
                break
            if self.recording and frames:
                for i, frame in enumerate(frames):
                    self.data[i].append(frame)
                robot_state = self.get_robot_state(0)
                self.robot_state_data.append(robot_state)
        cv2.destroyAllWindows()


    def get_robot_state(self, end_marker=0):
        angles = self.franka.angles()
        return np.concatenate((angles, [end_marker]))


    def save_data(self):
        if not any(self.data):
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
            camera_names = ['top', 'front']
            for cam_name, data in zip(camera_names, self.data):
                image_data = np.array(data, dtype='uint8')
                images.create_dataset(cam_name, data=image_data, dtype='uint8', chunks=(1, 480, 640, 3))
            robot_data = np.array(self.robot_state_data, dtype='float64')
            obs.create_dataset('qpos', data=robot_data)
            root.create_dataset('action', data=robot_data)

def main():
    camera_indices = [0, 2]
    camera_controller = CameraController(camera_indices)
    camera_controller.capture_frames()

if __name__ == '__main__':
    main()

