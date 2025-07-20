#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from functools import cached_property
from typing import Any
import threading

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

# ROS2 imports
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float64MultiArray
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("警告: 未找到ROS2，将使用模拟模式")

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_bxi_arm import BxiArmConfig

logger = logging.getLogger(__name__)


class RobotControlNode(Node):
    """ROS2节点，用于发布机械臂关节位置和接收电机数据"""
    
    def __init__(self, config: BxiArmConfig):
        super().__init__(config.ros_node_name)
        
        # 创建发布器，使用配置的话题名称
        self.qpos_publisher = self.create_publisher(
            Float64MultiArray, 
            config.ros_topic_name, 
            10
        )
        
        # 创建订阅器，订阅/robot_publish话题
        self.robot_publish_subscriber = self.create_subscription(
            Float64MultiArray,
            '/robot_publish',
            self.robot_publish_callback,
            10
        )
        
        # 存储当前关节位置
        self.current_qpos = [0.0] * config.num_joints
        
        # 存储电机数据：8个电机 x 3个值(位置、速度、力矩) = 24个值
        self.motor_data = [0.0] * 24
        self.motor_positions = [0.0] * 8  # 8个电机位置
        self.motor_velocities = [0.0] * 8  # 8个电机速度
        self.motor_torques = [0.0] * 8    # 8个电机力矩
        
        self.get_logger().info('BxiArm ROS2控制节点已启动')
    
    def publish_qpos(self, qpos_data: list):
        """发布关节位置数据"""
        try:
            # 确保数据长度正确
            if len(qpos_data) != 8:
                self.get_logger().warning(f"关节数据长度不正确，期望8个值，得到{len(qpos_data)}个")
                return
            
            # 创建消息
            msg = Float64MultiArray()
            msg.data = qpos_data
            
            # 发布消息
            self.qpos_publisher.publish(msg)
            
            # 更新当前位置
            self.current_qpos = qpos_data.copy()
            
            self.get_logger().debug(f"发布qpos: {qpos_data}")
            
        except Exception as e:
            self.get_logger().error(f"发布关节位置数据时出错: {e}")
    
    def robot_publish_callback(self, msg):
        """处理/robot_publish话题的回调函数
        
        Args:
            msg: Float64MultiArray消息，包含24个值：
                 8个电机 × 3个值(位置、速度、力矩)
        """
        try:
            if len(msg.data) != 24:
                self.get_logger().warning(f"电机数据长度不正确，期望24个值，得到{len(msg.data)}个")
                return
            
            # 存储完整的电机数据
            self.motor_data = list(msg.data)
            
            # 解析数据：每3个值为一个电机的位置、速度、力矩
            for i in range(8):
                base_index = i * 3
                self.motor_positions[i] = msg.data[base_index]      # 位置
                self.motor_velocities[i] = msg.data[base_index + 1] # 速度
                self.motor_torques[i] = msg.data[base_index + 2]    # 力矩
            
            self.get_logger().debug(f"接收到电机数据 - 位置: {self.motor_positions[:4]}...")
            
        except Exception as e:
            self.get_logger().error(f"处理电机数据时出错: {e}")


class BxiArm(Robot):
    """
    Bxi Arm designed by BXI
    通过ROS2发布关节位置数据来控制机械臂
    """

    config_class = BxiArmConfig
    name = "bxi_arm"

    def __init__(self, config: BxiArmConfig):
        super().__init__(config)
        self.config = config
        
        # ROS2相关
        self.ros_node = None
        self.ros_thread = None
        self.exit_event = threading.Event()
        
        # 关节名称（8个关节）
        self.joint_names = [
            "joint_1", "joint_2", "joint_3", "joint_4",
            "joint_5", "joint_6", "joint_7", "gripper"
        ]
        
        # 当前关节位置
        self.current_joint_positions = {f"{name}.pos": 0.0 for name in self.joint_names}
        
        # 电机观测数据
        self.motor_observations = {}
        self._init_motor_observations()
        
        # 初始化摄像头
        if self.config.cameras:
            self.cameras = make_cameras_from_configs(self.config.cameras)
        else:
            self.cameras = {}

    def _init_motor_observations(self):
        """初始化电机观测数据结构"""
        self.motor_observations = {}
        
        # 为每个电机添加位置、速度、力矩观测
        for joint_name in self.joint_names:
            self.motor_observations[f"{joint_name}.pos"] = 0.0
            self.motor_observations[f"{joint_name}.vel"] = 0.0  
            self.motor_observations[f"{joint_name}.torque"] = 0.0

    @property
    def _motors_ft(self) -> dict[str, type]:
        """电机特征定义"""
        motor_features = {}
        # 为每个关节添加位置、速度、力矩特征
        for joint_name in self.joint_names:
            motor_features[f"{joint_name}.pos"] = float
            motor_features[f"{joint_name}.vel"] = float
            motor_features[f"{joint_name}.torque"] = float
        return motor_features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """摄像头特征定义"""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) 
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """观测特征定义"""
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """动作特征定义，对应8个关节位置"""
        return {
            "qpos0": float, "qpos1": float, "qpos2": float, "qpos3": float,
            "qpos4": float, "qpos5": float, "qpos6": float, "qpos7": float
        }

    @property
    def is_connected(self) -> bool:
        """检查设备连接状态"""
        ros_connected = self.ros_node is not None and rclpy.ok()
        cameras_connected = all(cam.is_connected for cam in self.cameras.values())
        return ros_connected and cameras_connected

    def _init_ros2(self):
        """初始化ROS2节点"""
        if not ROS2_AVAILABLE:
            logger.warning("ROS2不可用，跳过ROS2初始化")
            return
        
        try:
            # 初始化ROS2
            if not rclpy.ok():
                rclpy.init()
            
            # 创建控制节点
            self.ros_node = RobotControlNode(self.config)
            
            # 在单独线程中运行ROS2
            def run_ros():
                try:
                    while rclpy.ok() and not self.exit_event.is_set():
                        rclpy.spin_once(self.ros_node, timeout_sec=0.1)
                except Exception as e:
                    logger.error(f"ROS2线程运行错误: {e}")
            
            self.ros_thread = threading.Thread(target=run_ros, daemon=True)
            self.ros_thread.start()
            
            logger.info("ROS2节点已初始化")
            
        except Exception as e:
            logger.error(f"初始化ROS2节点失败: {e}")
            raise

    def connect(self, calibrate: bool = True) -> None:
        """连接设备"""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # 初始化ROS2
        self._init_ros2()
        
        # 连接摄像头
        for cam in self.cameras.values():
            cam.connect()

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        """校准状态"""
        return True

    def calibrate(self) -> None:
        """校准机械臂"""
        pass

    def configure(self) -> None:
        """配置机械臂"""
        pass

    def setup_motors(self) -> None:
        """设置电机"""
        pass

    def get_observation(self) -> dict[str, Any]:
        """获取观测数据"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # 更新电机观测数据
        self._update_motor_observations()
        
        # 使用电机观测数据而不是当前关节位置
        obs_dict = self.motor_observations.copy()

        # 读取摄像头图像
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict
    
    def _update_motor_observations(self):
        """从ROS节点更新电机观测数据"""
        if self.ros_node and ROS2_AVAILABLE:
            # 从ROS节点获取最新的电机数据
            for i, joint_name in enumerate(self.joint_names):
                self.motor_observations[f"{joint_name}.pos"] = self.ros_node.motor_positions[i]
                self.motor_observations[f"{joint_name}.vel"] = self.ros_node.motor_velocities[i]
                self.motor_observations[f"{joint_name}.torque"] = self.ros_node.motor_torques[i]

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """发送动作指令到机械臂
        
        Args:
            action: 包含qpos0-qpos7的关节位置字典
            
        Returns:
            实际发送的动作
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        try:
            # 提取关节位置数据
            qpos_data = []
            for i in range(8):
                key = f"qpos{i}"
                if key in action:
                    qpos_data.append(float(action[key]))
                else:
                    logger.warning(f"缺少关节位置数据: {key}")
                    qpos_data.append(0.0)
            
            # 通过ROS2发布关节位置
            if self.ros_node and ROS2_AVAILABLE:
                self.ros_node.publish_qpos(qpos_data)
                
                # 更新当前关节位置
                for i, joint_name in enumerate(self.joint_names):
                    self.current_joint_positions[f"{joint_name}.pos"] = qpos_data[i]
            
            # 返回实际发送的动作
            return {f"qpos{i}": qpos_data[i] for i in range(8)}
            
        except Exception as e:
            logger.error(f"发送动作指令失败: {e}")
            # 返回当前位置作为默认值
            return {f"qpos{i}": 0.0 for i in range(8)}

    def disconnect(self):
        """断开连接"""
        logger.info(f"正在断开 {self} 连接...")
        
        # 设置退出事件
        self.exit_event.set()
        
        # 停止ROS2线程
        if self.ros_thread and self.ros_thread.is_alive():
            self.ros_thread.join(timeout=2.0)
            if self.ros_thread.is_alive():
                logger.warning("ROS2线程未能在2秒内正常退出")
        
        # 销毁ROS2节点
        if self.ros_node:
            try:
                self.ros_node.destroy_node()
            except Exception as e:
                logger.error(f"销毁ROS2节点时出错: {e}")
        
        # 关闭ROS2
        if ROS2_AVAILABLE and rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception as e:
                logger.error(f"关闭ROS2时出错: {e}")

        # 断开摄像头连接
        for cam_name, cam in self.cameras.items():
            try:
                if cam.is_connected:
                    cam.disconnect()
                    logger.info(f"摄像头 {cam_name} 已断开连接")
            except Exception as e:
                logger.error(f"断开摄像头 {cam_name} 时出错: {e}")

        logger.info(f"{self} 连接已断开")
