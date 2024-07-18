import csv
import time
from pynput import keyboard
from controller.robot_state import *

class Placetraj:
    def __init__(self):
        self.recording = False
        self.data = []
        self.franka = RobotController()
        self.listener = keyboard.Listener(on_press=self.on_press)

    def toggle_recording(self):
        self.recording = not self.recording
        if not self.recording:
            self.save_data()
            print("Stopped recording. Data saved.")
        else:
            self.data = []  
            print("Started recording.")

    def save_data(self):
        if self.data:  # Ensure there is data to save.
            with open('csv/place_actions.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Angle1', 'Angle2', 'Angle3', 'Angle4', 'Angle5', 'Angle6', 'Angle7', 'Gripper'])
                writer.writerows(self.data)
            print("Data successfully saved to 'place_actions.csv'.")

    def on_press(self, key):
        try:
            if key.char == 's':
                self.toggle_recording()
        except AttributeError:
            pass

    def run(self):
        with self.listener:
            print("Press 's' to start or stop recording. Press 'Ctrl+C' to exit.")
            try:
                while True:
                    if self.recording:
                        angles = self.franka.angles()
                        angles.append(1)  # Simulating the gripper status.
                        print(angles)
                        self.data.append(angles)
                    time.sleep(1)
            except KeyboardInterrupt:
                print("Exiting program due to Ctrl+C.")
            finally:
                if self.recording:
                    self.save_data()
                self.listener.stop()

if __name__ == '__main__':
    controller = Placetraj()
    controller.run()
