import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from franka_gripper.msg import MoveActionGoal, GraspActionGoal, GraspGoal

class RobotController:
    def __init__(self, controller_name='/position_joint_trajectory_controller/command', joint_state_topic='/joint_states'):
        rospy.init_node('robot_controller', anonymous=True)
        self.trajectory_publisher = rospy.Publisher(controller_name, JointTrajectory, queue_size=10)
        self.gripper_publisher = rospy.Publisher('/franka_gripper/move/goal', MoveActionGoal, queue_size=10)
        self.grasp_publisher = rospy.Publisher('/franka_gripper/grasp/goal', GraspActionGoal, queue_size=10)
        
        self.joint_state_subscriber = rospy.Subscriber(joint_state_topic, JointState, self.joint_state_callback)
        self.gripper_state_subscriber = rospy.Subscriber('/franka_gripper/joint_states', JointState, self.gripper_joint_state_callback)
        
        self.joints = ['fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4', 'fr3_joint5', 'fr3_joint6', 'fr3_joint7']
        self.gripper_positions = []
        self.current_positions = [None] * 7
        self.position_tolerance = 0.001  # Tolerance for position errors in radians

    def gripper_joint_state_callback(self, msg):
        positions = msg.position
        if positions is not None:
            self.gripper_positions = positions

    def gripper_state(self):
        if not self.gripper_positions:
            rospy.logwarn("No gripper positions received yet.")
            return None
        
        position = sum(self.gripper_positions)

        rospy.loginfo(f"gripper distance: {position}")
        return position


    def move_to_joint_position(self, angles, timeout=5):
        if len(angles) != 7:
            raise ValueError("Exactly 7 angles must be provided.")
        # rospy.loginfo("Moving robot to specified angles...")
        
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = self.joints
        point = JointTrajectoryPoint()
        point.positions = angles
        point.velocities = [0.0] * 7
        point.accelerations = [0.0] * 7
        point.time_from_start = rospy.Duration(5)
        rospy.sleep(0.5)
        trajectory_msg.points.append(point)

        self.trajectory_publisher.publish(trajectory_msg)
        # rospy.loginfo("Command published to move the robot.")

        start_time = rospy.Time.now()
        while (rospy.Time.now() - start_time).to_sec() < timeout:
            if self.current_positions is None:
                rospy.logwarn("Current positions have not been received yet.")
                rospy.sleep(0.1)
                continue

            if self.positions_close_enough(self.current_positions, angles):
                # rospy.loginfo("Target positions reached.")
                break
            rospy.sleep(0.1)
        # else:
        #     rospy.logwarn("Timeout reached before reaching the target positions.")

    def positions_close_enough(self, current, target):
        if None in current:
            return False
        errors = [abs(c - t) for c, t in zip(current, target)]
        return all(error < self.position_tolerance for error in errors)

    def joint_state_callback(self, msg):
        positions = [None] * len(self.joints)
        for idx, name in enumerate(self.joints):
            if name in msg.name:
                pos_idx = msg.name.index(name)
                positions[idx] = msg.position[pos_idx]
        self.current_positions = positions

    def angles(self):
        if self.current_positions is None:
            rospy.logwarn("Current positions have not been received yet.")
            return None
        return self.current_positions
    
    def exec_gripper_cmd(self ,width, speed=0.5):
        while self.gripper_publisher.get_num_connections() == 0:
            rospy.sleep(0.1)
        move_goal = MoveActionGoal()
        move_goal.goal.width = width
        move_goal.goal.speed = speed

        self.gripper_publisher.publish(move_goal)

    def grasp(self,width,force=5,speed=0.1,inner_epsilon=0.005,outer_epsilon=0.005):
               
        if not rospy.core.is_initialized():
            rospy.init_node('gripper_command_publisher', anonymous=True)

     
        grasp_goal = GraspGoal()
        grasp_goal.width = width
        grasp_goal.speed = speed
        grasp_goal.force = force
        grasp_goal.epsilon.inner = inner_epsilon
        grasp_goal.epsilon.outer = outer_epsilon

      
        action_goal = GraspActionGoal()
        action_goal.goal = grasp_goal

        # Publish the message
        self.grasp_publisher.publish(action_goal)
        # rospy.loginfo("Published grasp goal to /franka_gripper/grasp/goal")


    def initial_pose(self):

        angles = [0, 
                         -0.785398163, 
                         0, 
                         -2.35619449, 
                         0, 
                         1.57079632679, 
                         0.785398163397]
        
        self.move_to_joint_position(angles)




