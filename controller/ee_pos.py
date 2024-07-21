import rospy
import tf

def listen_to_tf():
    rospy.init_node('tf_listener')
    listener = tf.TransformListener()

    rate = rospy.Rate(1.0)  # Set the rate to 1Hz
    while not rospy.is_shutdown():
        try:
            # Wait for the frame to be available
            listener.waitForTransform("/fr3_link0", "fr3_EE", rospy.Time(), rospy.Duration(4.0))
            (trans, rot) = listener.lookupTransform('/fr3_link0', 'fr3_EE', rospy.Time(0))
            print("Translation: ", trans)
            print("Rotation: ", rot)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            print("Error: ", e)

        rate.sleep()

if __name__ == '__main__':
    listen_to_tf()
