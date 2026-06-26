import rospy
import rospy
from sensor_msgs.msg import JointState

left_joint_states = None
right_joint_states = None
# a ros node that subscribe /puppet_left/joint_states and /puppet_right/joint_states, output /joint_states
def callback_left(msg:JointState):
    global left_joint_states
    # Process left joint states
    left_joint_states = msg

def callback_right(msg:JointState):
    global right_joint_states
    # Process right joint states
    right_joint_states = msg

def combine_joint_states(msg1, msg2):
    combined_msg = JointState()

    # Update header of combined message
    combined_msg.header = msg1.header
    combined_msg.header.stamp = rospy.Time.now()

    # Update names of joints
    combined_msg.name = [f"puppet_left/{joint}" for joint in msg1.name] + [f"puppet_right/{joint}" for joint in msg2.name]

    # Concatenate positions
    combined_msg.position = msg1.position + msg2.position

    # Leave velocity and effort as empty lists
    combined_msg.velocity = msg1.velocity + msg2.velocity
    combined_msg.effort = msg1.effort + msg2.effort

    return combined_msg


def remap_joint_states():
    rospy.init_node('joint_state_remapper', anonymous=True)
    rospy.Subscriber('/puppet_left/joint_states', JointState, callback_left)
    rospy.Subscriber('/puppet_right/joint_states', JointState, callback_right)
    pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
    rate = rospy.Rate(100)  # Adjust the publishing rate as needed

    while not rospy.is_shutdown():
        if left_joint_states is None or right_joint_states is None:
            rate.sleep()
            continue
        
        # Combine and publish joint states
        joint_states = combine_joint_states(left_joint_states, right_joint_states)
        # Combine left and right joint states into joint_states variable
        pub.publish(joint_states)
        rate.sleep()

if __name__ == '__main__':
    try:
        remap_joint_states()
    except rospy.ROSInterruptException:
        pass

