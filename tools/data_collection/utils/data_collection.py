from utils.data_collection_cfg import DataCollectionCfg
from dataset.hdf5_dataset import H5Dataset
from y1_msg.msg import ArmJointState
from sensor_msgs.msg import Image
import rospy
import collections
import dm_env
import numpy as np
import cv2
import time
from cv_bridge import CvBridge
from utils.key_handle import init_keyboard_listener

class DataCollection:
  def __init__(self, cfg: DataCollectionCfg):
    self.cfg = cfg
    self.bridge = CvBridge()
    self.master_arm_right_state = None
    self.master_arm_left_state = None
    self.puppet_arm_right_state = None
    self.puppet_arm_left_state = None
    self.last_state = None  # for check static or move
    self.img_dict = {}
    self.depth_dict = {}
    self.init_topic()
    time.sleep(1)
    
  def init_topic(self):
    rospy.init_node("data_collection")
    # subscribe robotic arm data
    robotic_arm_cfg = self.cfg.robotic_arm_cfg
    if robotic_arm_cfg.master_arm_right_topic is not None:
      rospy.Subscriber(robotic_arm_cfg.master_arm_right_topic,
            ArmJointState, self.master_arm_right_callback, queue_size=1, tcp_nodelay=True)
    
    if robotic_arm_cfg.master_arm_left_topic is not None:
      rospy.Subscriber(robotic_arm_cfg.master_arm_left_topic,
          ArmJointState, self.master_arm_left_callback, queue_size=1, tcp_nodelay=True)
    
    if robotic_arm_cfg.puppet_arm_right_topic is not None:
      rospy.Subscriber(robotic_arm_cfg.puppet_arm_right_topic,
        ArmJointState, self.puppet_arm_right_callback, queue_size=1, tcp_nodelay=True)
        
    if robotic_arm_cfg.puppet_arm_left_topic is not None:
      rospy.Subscriber(robotic_arm_cfg.puppet_arm_left_topic,
        ArmJointState, self.puppet_arm_left_callback, queue_size=1, tcp_nodelay=True)
      
    # subscribe camera rgb data
    camera_cfg = self.cfg.camera_cfg
    if camera_cfg.img_right_topic:
      # right arm wrist camera rgb image
      rospy.Subscriber(camera_cfg.img_right_topic, 
        Image, self.img_right_callback, queue_size=1, tcp_nodelay=True)
    
    if camera_cfg.img_left_topic:
      # left arm wrist camera rgb image
      rospy.Subscriber(camera_cfg.img_left_topic, 
        Image, self.img_left_callback, queue_size=1, tcp_nodelay=True)
      
    if camera_cfg.img_high_topic:
      # front external camera rgb image
      rospy.Subscriber(camera_cfg.img_high_topic, 
        Image, self.img_high_callback, queue_size=1, tcp_nodelay=True)
      
    if camera_cfg.img_low_topic:
      # top external camera rgb image
      rospy.Subscriber(camera_cfg.img_low_topic, 
        Image, self.img_low_callback, queue_size=1, tcp_nodelay=True)
      
    # subscribe camera depth data
    if camera_cfg.depth_right_topic:
      # right arm wrist camera depth image
      rospy.Subscriber(camera_cfg.depth_right_topic, 
        Image, self.depth_right_callback, queue_size=1, tcp_nodelay=True)
    
    if camera_cfg.depth_left_topic:
      # left arm wrist camera depth image
      rospy.Subscriber(camera_cfg.depth_left_topic, 
        Image, self.depth_left_callback, queue_size=1, tcp_nodelay=True)
      
    if camera_cfg.depth_high_topic:
      # front external camera depth image
      rospy.Subscriber(camera_cfg.depth_high_topic, 
        Image, self.depth_high_callback, queue_size=1, tcp_nodelay=True)
      
    if camera_cfg.depth_low_topic:
      # top external camera depth image
      rospy.Subscriber(camera_cfg.depth_low_topic, 
        Image, self.depth_low_callback, queue_size=1, tcp_nodelay=True)
      
  def master_arm_right_callback(self, msg: ArmJointState):
    self.master_arm_right_state = msg 
    
  def puppet_arm_right_callback(self, msg: ArmJointState):
    self.puppet_arm_right_state = msg
    
  def master_arm_left_callback(self, msg: ArmJointState):
    self.master_arm_left_state = msg
    
  def puppet_arm_left_callback(self, msg: ArmJointState):
    self.puppet_arm_left_state = msg
    
  def img_right_callback(self, msg: Image):
    cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
    # 编码为 JPEG 格式的二进制数据
    ret, buf = cv2.imencode('.jpg', cv_image)
    if ret:
      self.img_dict["cam_right_wrist"] = buf.tobytes()
    
  def img_left_callback(self, msg: Image):
    cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
    # 编码为 JPEG 格式的二进制数据
    ret, buf = cv2.imencode('.jpg', cv_image)
    if ret:
      self.img_dict["cam_left_wrist"] = buf.tobytes()
    
  def img_high_callback(self, msg: Image):
    cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
    # 编码为 JPEG 格式的二进制数据
    ret, buf = cv2.imencode('.jpg', cv_image)
    if ret:
      self.img_dict["cam_high"] = buf.tobytes()
    
  def img_low_callback(self, msg: Image):
    cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
    # 编码为 JPEG 格式的二进制数据
    ret, buf = cv2.imencode('.jpg', cv_image)
    if ret:
      self.img_dict["cam_low"] = buf.tobytes()
    
  def depth_right_callback(self, msg: Image):
    self.depth_right = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    
  def depth_left_callback(self, msg: Image):
    self.depth_left = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    
  def depth_high_callback(self, msg: Image):
    self.depth_high = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    
  def depth_low_callback(self, msg: Image):
    self.depth_low = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    
  def get_frame(self):
    """Get all the data of a frame"""
    frame = {}
    
    # TODO: 要对数据做时间同步吗？
    if self.cfg.robotic_arm_cfg.collection_type == "one_master":
      # 单臂末端2合1数采，默认为右臂
      if self.master_arm_right_state is None:
        print("Not receive master_arm_right_state data!")
        return False
      
      frame["state"] = np.array(self.master_arm_right_state.joint_position)
      frame["action"] = np.array(self.master_arm_right_state.joint_position)
      if self.cfg.robotic_arm_cfg.use_joint_velocity:
        frame["velocity"] = np.array(self.master_arm_right_state.joint_velocity)
      if self.cfg.robotic_arm_cfg.use_joint_effort:
        frame["effort"] = np.array(self.master_arm_right_state.joint_effort)
        
    elif self.cfg.robotic_arm_cfg.collection_type == "two_master":
      # 双臂末端2合1数采
      if self.master_arm_right_state is None:
        print("Not receive master_arm_right_state data!")
        return False
      if self.master_arm_left_state is None:
        print("Not receive master_arm_left_state data!")
        return False
      
      frame["state"] = np.concatenate([self.master_arm_right_state.joint_position,
                                 self.master_arm_left_state.joint_position])
      frame["action"] = np.concatenate([self.master_arm_right_state.joint_position,
                                 self.master_arm_left_state.joint_position])
      if self.cfg.robotic_arm_cfg.use_joint_velocity:
        frame["velocity"] = np.concatenate([self.master_arm_right_state.joint_velocity,
                                 self.master_arm_left_state.joint_velocity])
      if self.cfg.robotic_arm_cfg.use_joint_effort:
        frame["effort"] = np.concatenate([self.master_arm_right_state.joint_effort,
                                 self.master_arm_left_state.joint_effort])
    elif self.cfg.robotic_arm_cfg.collection_type == "one_master_slave":
      # 单臂主从数采, 默认为右臂
      if self.master_arm_right_state is None:
        print("Not receive master_arm_right_state data!")
        return False
      if self.puppet_arm_right_state is None:
        print("Not receive puppet_arm_right_state data!")
        return False
      
      frame["state"] = np.array(self.puppet_arm_right_state.joint_position)
      frame["action"] = self.master_arm_right_state.joint_position
      if self.cfg.robotic_arm_cfg.use_joint_velocity:
        frame["velocity"] = np.array(self.puppet_arm_right_state.joint_velocity)
      if self.cfg.robotic_arm_cfg.use_joint_effort:
        frame["effort"] = np.array(self.puppet_arm_right_state.joint_effort)
    
    elif self.cfg.robotic_arm_cfg.collection_type == "two_master_slave":
      # 双臂主从数采
      if self.master_arm_right_state is None:
        print("Not receive master_arm_right_state data!")
        return False
      
      if self.master_arm_left_state is None:
        print("Not receive master_arm_left_state data!")
        return False
      
      if self.puppet_arm_right_state is None:
        print("Not receive puppet_arm_right_state data!")
        return False
      
      if self.puppet_arm_left_state is None:
        print("Not receive puppet_arm_left_state data!")
        return False
      
      frame["state"] = np.concatenate([self.puppet_arm_right_state.joint_position,
                                      self.puppet_arm_left_state.joint_position])
      frame["action"] = np.concatenate([self.master_arm_right_state.joint_position,
                                 self.master_arm_left_state.joint_position])
      if self.cfg.robotic_arm_cfg.use_joint_velocity:
        frame["velocity"] = np.concatenate([self.puppet_arm_right_state.joint_velocity,
                                              self.puppet_arm_left_state.joint_velocity])
      if self.cfg.robotic_arm_cfg.use_joint_effort:
        frame["effort"] = np.concatenate([self.puppet_arm_right_state.joint_effort,
                                          self.puppet_arm_left_state.joint_effort])
      
    else:
      print(f"Unsupport collection type: {self.cfg.robotic_arm_cfg.collection_type}")
      exit(1)
    
    cameras = {}
    for cam_name in self.cfg.camera_cfg.camera_names:
      if cam_name == "cam_right_wrist" and self.img_dict["cam_right_wrist"] is None:
        print("Not receive right image data!")
        return False
      if cam_name == "cam_left_wrist" and self.img_dict["cam_left_wrist"] is None:
        print("Not receive left image data!")
        return False
      if cam_name == "cam_high" and self.img_dict["cam_high"] is None:
        print("Not receive high image data!")
        return False
      if cam_name == "cam_low" and self.img_dict["cam_low"] is None:
        print("Not receive low image data!")
        return False
      
      cameras[cam_name] = self.img_dict[cam_name]
      
    frame["cameras"] = cameras
     
    return frame
    
  def collect(self, events: dict):
    rate = rospy.Rate(self.cfg.save_rate)
    timesteps = []
    actions = []
    first_step = True
    
    while not rospy.is_shutdown():
      if events["finish_current_recording"]:
        events["finish_current_recording"] = False
        break
      
      if events["stop_recording"]:
        print("Exit program!")
        exit(1)
      
      frame = self.get_frame() 
      if not frame:
        print("get frame failed!")
        rate.sleep()
        continue

      # not save static data
      if self.cfg.skip_static_data:
        if first_step:
          self.last_state = frame["state"]
        else:
          if np.all(np.abs(self.last_state - frame["state"]) < 0.0001):
            # print("Skip static data!")
            rate.sleep()
            continue
          else:
            # print("Collecting data...")
            self.last_state = frame["state"]
          
      # observation
      obs = collections.OrderedDict()
      
      # state
      obs["state"] = frame["state"]

      if self.cfg.robotic_arm_cfg.use_joint_velocity:
        obs["velocity"] = frame["velocity"]
      if self.cfg.robotic_arm_cfg.use_joint_effort:
        obs["effort"] = frame["effort"]
      
      # camera image data
      obs['cameras'] = frame["cameras"]
      
      if first_step:
        # not append action in first step
        first_step = False
        ts = dm_env.TimeStep(
          step_type=dm_env.StepType.FIRST,
          reward=None,
          discount=None,
          observation=obs
        )
        timesteps.append(ts)
        continue
      
      ts = dm_env.TimeStep(
        step_type=dm_env.StepType.MID,
        reward=None,
        discount=None,
        observation=obs
      )
      
      timesteps.append(ts)
      actions.append(frame["action"])
      rate.sleep()
      
    # ctrl-c exit the program  
    if rospy.is_shutdown():
        print("Exit program!")
        exit(-1)
      
    return timesteps, actions
    
  def run(self):
    if self.cfg.num_episodes < 1:
      raise ValueError("num_episodes must be greater than 0")
    
    h5_dataset = H5Dataset(self.cfg)
    
    # start_e_listener()
    listener, events = init_keyboard_listener()
    if listener is None:
      exit(1)
      
    # 一共可以采集num_episodes条数据
    count = 0
    while count < self.cfg.num_episodes:
    # for i in range(self.cfg.num_episodes):
      # TODO: 每次采集前让机械臂回初始零位？
      print(f"Press key [S] to start record {count + 1}th episode!")
      while events["start_recording"] is False:
        if events["stop_recording"]:
          print("Exit program!")
          exit(1)
          
      events["start_recording"] = False
      print("Start record data! Press [E] to finish record current episode, or [R] to rerecord!")
      # collecte one episode
      timesteps, actions = self.collect(events)
      
      if len(actions) == 0:
        print(f"Data is empty, rerecord {count + 1}th episode!")
        continue
      
      if events["rerecord_episode"]:
        print(f"Rerecord {count + 1}th episode!")
        events["rerecord_episode"] = False
        continue
      
      # save as hdf5
      h5_dataset.save_to_hdf5(timesteps, actions)
      
      count += 1 
      