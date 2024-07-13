
import rospy
from gazebo_msgs.srv import SetModelState, SetModelStateRequest



def set_box_position(x, y, z):
        rospy.wait_for_service('/gazebo/set_model_state')
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        state_msg = SetModelStateRequest()
        state_msg.model_state.model_name = 'block_red_2'
        state_msg.model_state.pose.position.x = x
        state_msg.model_state.pose.position.y = y
        state_msg.model_state.pose.position.z = z
        set_state(state_msg)


def set_box_position_from_input():
        print("Enter new box coordinates as x, y ")
        coords = input()
        try:
            x, y = map(float, coords.split(','))
            new_x, new_y = x, y
            set_box_position(x, y, 0.025)
        except ValueError:
            print("Invalid input. Using default position (-0.5, 0.5).")


set_box_position_from_input()
