from y1_msg.msg import ArmJointState
from y1_msg.msg import ArmJointPositionControl
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import rospy
import numpy as np
import torch
from typing import Union

class RealRobotEnv:
  def __init__(self, single_arm: bool, cam_names: list):
    self.single_arm = single_arm
    self.cam_names = cam_names
    
    self.bridge = CvBridge()
    self.right_puppet_arm_state = None
    self.left_puppet_arm_state = None
    self.img_dict = {}
    self.left_arm_joint_position_control_pub_ = None
    self.right_arm_joint_position_control_pub_ = None
    self.init_topic()
    
  def init_topic(self):
    rospy.init_node("eval_real_robot")
    
    # robotic arm joint data
    if self.single_arm:
      # single arm, default right arm
      rospy.Subscriber("/puppet_arm_right/joint_states",
          ArmJointState, self.puppet_arm_right_callback, queue_size=1, tcp_nodelay=True)
      
      # control right arm
      self.right_arm_joint_position_control_pub_ = rospy.Publisher('/master_arm_right/joint_states', 
                                                                   ArmJointPositionControl, queue_size=1)
    else:
      # dual arm
      rospy.Subscriber("/puppet_arm_left/joint_states",
            ArmJointState, self.puppet_arm_left_callback, queue_size=1, tcp_nodelay=True)
      rospy.Subscriber("/puppet_arm_right/joint_states",
            ArmJointState, self.puppet_arm_right_callback, queue_size=1, tcp_nodelay=True)
      
      # control left and right arm
      self.left_arm_joint_position_control_pub_ = rospy.Publisher('/master_arm_left/joint_states', 
                                                                  ArmJointPositionControl, queue_size=1)
      self.right_arm_joint_position_control_pub_ = rospy.Publisher('/master_arm_right/joint_states', 
                                                                   ArmJointPositionControl, queue_size=1)
  
    # camera rgb image data
    for cam_name in self.cam_names:
      if cam_name == "cam_right_wrist":
        # right arm wrist camera rgb image
        rospy.Subscriber("/camera_right/color/image_raw", 
          Image, self.img_right_callback, queue_size=1, tcp_nodelay=True)
      elif cam_name == "cam_left_wrist":
        # left arm wrist camera rgb image
        rospy.Subscriber("/camera_left/color/image_raw", 
          Image, self.img_left_callback, queue_size=1, tcp_nodelay=True)
      elif cam_name == "cam_high":
        # front camera rgb image
        rospy.Subscriber("/camera_high/color/image_raw", 
          Image, self.img_high_callback, queue_size=1, tcp_nodelay=True)
      elif cam_name == "cam_low":
        # top camera rgb image
        rospy.Subscriber("/camera_low/color/image_raw", 
          Image, self.img_low_callback, queue_size=1, tcp_nodelay=True)
      else:
        raise Exception(f"camera name [{cam_name}] not defined in CameraSubscriber, only support [{cam_name}]")

  def puppet_arm_right_callback(self, msg: ArmJointState):
    """right arm"""
    self.right_puppet_arm_state = msg 
    
  def puppet_arm_left_callback(self, msg: ArmJointState):
    """left arm"""
    self.left_puppet_arm_state = msg 
    
  def img_right_callback(self, msg: Image):
    """right arm wrist camera rgb image"""
    self.img_dict["cam_right_wrist"] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
    
  def img_left_callback(self, msg: Image):
    """left arm wrist camera rgb image"""
    self.img_dict["cam_left_wrist"] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
    
  def img_high_callback(self, msg: Image):
    """high camera rgb image"""
    self.img_dict["cam_high"] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
    
  def img_low_callback(self, msg: Image):
    """low camera rgb image"""
    self.img_dict["cam_low"] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
    
  def get_observation(self):
    observation = {}

    # robot joint position
    if self.single_arm:
      # single arm
      if self.right_puppet_arm_state is None:
        print("not receive right arm data")
        return None
      else:
        joint_state = np.array(self.right_puppet_arm_state.joint_position)
        observation["observation.state"] = torch.from_numpy(joint_state).float()
    else:
      # dual arm
      if self.left_puppet_arm_state is None:
        print("not receive left arm data")
        return None
      
      if self.right_puppet_arm_state is None:
        print("not receive right arm data")
        return None
      
      observation["observation.state"] = torch.from_numpy(np.concatenate([self.left_puppet_arm_state.joint_position,
                                 self.right_puppet_arm_state.joint_position])).float()
    
    # camera image
    for cam_name in self.cam_names:
      if cam_name not in self.img_dict:
        print(f"not receive {cam_name} image data")
        return None
      
      observation[f"observation.images.{cam_name}"] = torch.from_numpy(self.img_dict[cam_name])
      
    return observation
    
  def step(self, action: Union[list, np.ndarray, torch.Tensor]):
    if self.single_arm:
      assert len(action) >= 7
      
      # single arm, default right arm
      joint_control_msg = ArmJointPositionControl()
      joint_control_msg.header.stamp = rospy.Time.now()
      joint_control_msg.joint_position = action[0:6]
      joint_control_msg.joint_velocity = 3
      joint_control_msg.gripper_stroke = action[6]
      joint_control_msg.gripper_velocity = 3
      self.right_arm_joint_position_control_pub_.publish(joint_control_msg)

    else:
      assert len(action) >= 14

      # action[0:6]  -> left arm control
      left_arm_control_msg = ArmJointPositionControl()
      left_arm_control_msg.header.stamp = rospy.Time.now()
      left_arm_control_msg.joint_position = action[0:6]
      left_arm_control_msg.joint_velocity = 5
      left_arm_control_msg.gripper_stroke = action[6]
      left_arm_control_msg.gripper_velocity = 5
      self.left_arm_joint_position_control_pub_.publish(left_arm_control_msg)

      # action[7:13] -> right arm control
      right_arm_control_msg = ArmJointPositionControl()
      right_arm_control_msg.header.stamp = rospy.Time.now()
      right_arm_control_msg.joint_position = action[7:13]
      right_arm_control_msg.joint_velocity = 5
      right_arm_control_msg.gripper_stroke = action[13]
      right_arm_control_msg.gripper_velocity = 5
      self.right_arm_joint_position_control_pub_.publish(right_arm_control_msg)

    # TODO: add mobile_base control

