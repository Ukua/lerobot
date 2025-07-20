#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""
基于VMC动作捕捉的机械臂遥操作模块
支持右手数据到机械臂映射
"""

import logging
import signal
import sys
from typing import Any
import numpy as np
from math import radians, degrees
from scipy.spatial.transform import Rotation
import threading
import time

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from ..teleoperator import Teleoperator
from .configuration_rebocap import RebocapTeleopConfig

from pynput import keyboard
from queue import Queue

# VMC相关导入
try:
    from vmcp.osc import OSC
    from vmcp.osc.typing import Message
    from vmcp.osc.backend.osc4py3 import as_comthreads as backend
    from vmcp.events import (
        Event,
        RootTransformEvent,
        BoneTransformEvent,
        BlendShapeEvent,
        BlendShapeApplyEvent,
        DeviceTransformEvent,
        StateEvent,
        RelativeTimeEvent
    )
    from vmcp.typing import (
        CoordinateVector,
        Quaternion,
        Bone,
        DeviceType,
        BlendShapeKey as AbstractBlendShapeKey,
        ModelState,
        Timestamp
    )
    from vmcp.protocol import (
        root_transform,
        bone_transform,
        device_transform,
        blendshape,
        blendshape_apply,
        state,
    )
    from vmcp.facades import on_receive
    VMC_AVAILABLE = True
except ImportError:
    logging.warning("VMC packages not available. RebocapTeleop will be disabled.")
    VMC_AVAILABLE = False


def quaternion_to_euler(quat, degrees_output=False, seq='xyz'):
    """
    将四元数转换为欧拉角
    :param quat: 四元数，格式为[x, y, z, w]或[w, x, y, z]
    :param degrees_output: 是否输出角度制
    :param seq: 欧拉角顺序，默认'xyz'
    :return: 欧拉角 (roll, pitch, yaw)
    """
    # 确保输入是numpy数组
    quat = np.array(quat)
    # 如果是[w, x, y, z]格式，转换为[x, y, z, w]
    if quat.shape[-1] == 4 and abs(quat[0]) > 1 and abs(quat[3]) <= 1:
        quat = np.roll(quat, -1)
    r = Rotation.from_quat(quat)
    euler = r.as_euler(seq, degrees=degrees_output)
    
    return euler


def fix_angle_discontinuity(current_angles, previous_angles):
    """
    修正角度跳变，保持连续性
    例如：从3.14跳到-3.13时，将-3.13修正为3.15
    :param current_angles: 当前角度数组
    :param previous_angles: 上一帧角度数组
    :return: 修正后的角度数组
    """
    if previous_angles is None:
        return current_angles
    
    current_angles = np.array(current_angles)
    previous_angles = np.array(previous_angles)
    corrected_angles = current_angles.copy()
    
    for i in range(len(current_angles)):
        diff = current_angles[i] - previous_angles[i]
        
        # 如果角度差大于π，说明发生了从负到正的跳变
        if diff > np.pi:
            corrected_angles[i] = current_angles[i] - 2 * np.pi
        # 如果角度差小于-π，说明发生了从正到负的跳变
        elif diff < -np.pi:
            corrected_angles[i] = current_angles[i] + 2 * np.pi
    
    return corrected_angles


class RebocapTeleop(Teleoperator):
    """
    基于VMC动作捕捉的机械臂遥操作类
    支持通过VMC协议接收右手数据并映射到机械臂关节
    """

    config_class = RebocapTeleopConfig
    name = "rebocap"

    def __init__(self, config: RebocapTeleopConfig):
        super().__init__(config)
        self.config = config
        
        # 检查VMC可用性
        if not VMC_AVAILABLE:
            raise ImportError("VMC packages not available. Please install vmcp to use RebocapTeleop.")
        
        # VMC数据存储
        self.pos_upper_arm = [0, 0, 0]
        self.rot_upper_arm = [0, 0, 0, 1]  # Quaternion [x, y, z, w]
        self.pos_lower_arm = [0, 0, 0]
        self.rot_lower_arm = [0, 0, 0, 1]  # Quaternion [x, y, z, w]
        self.pos_hand = [0, 0, 0]
        self.rot_hand = [0, 0, 0, 1]  # Quaternion [x, y, z, w]
        
        # 存储上一帧的欧拉角，用于跳变检测
        self.previous_euler = None
        self.previous_euler2 = None
        
        # VMC控制标志
        self.vmc_listening = False
        self.vmc_thread = None
        self.vmc_connected = False
        
        # 初始化OSC
        try:
            self.osc = OSC(backend)
            self.receiver = None
            logging.info("VMC组件初始化成功")
        except Exception as e:
            logging.error(f"VMC组件初始化失败: {e}")
            raise DeviceAlreadyConnectedError(f"Failed to initialize VMC: {e}")
        
        # 注册信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)
        
        self.logs = {}

        #keyboard
        self.event_queue = Queue()
        self.current_pressed = {}
        self.gripper = 0.0  # 初始夹爪位置

    @property
    def action_features(self) -> dict:
        """返回动作特征定义，支持7自由度机械臂关节角度，最后一个为夹爪"""
        return {
            "dtype": "float32",
            "shape": (8,),
            "names": {
                "qpos0": 0, "qpos1": 1, "qpos2": 2,
                "qpos3": 3, "qpos4": 4, "qpos5": 5, "qpos6": 6,
                "qpos7": 7
            },
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.vmc_connected

    @property
    def is_calibrated(self) -> bool:
        return True
    
    # for keyboard
    def _on_press(self, key):
        if hasattr(key, "char"):
            self.event_queue.put((key.char, True))

    def _on_release(self, key):
        if hasattr(key, "char"):
            self.event_queue.put((key.char, False))
        if key == keyboard.Key.esc:
            logging.info("ESC pressed, disconnecting.")
            self.disconnect()

    def _drain_pressed_keys(self):
        while not self.event_queue.empty():
            key_char, is_pressed = self.event_queue.get_nowait()
            self.current_pressed[key_char] = is_pressed


    def _signal_handler(self, sig, frame):
        """优雅地处理Ctrl+C信号"""
        logging.info("收到中断信号，正在停止VMC接收...")
        self.vmc_listening = False
        if self.osc:
            self.osc.close()

    def _vmc_received(self, event: Event):
        """VMC事件接收回调函数"""
        if isinstance(event, BoneTransformEvent):
            # 过滤右手相关骨骼
            if event.joint == Bone.RIGHT_UPPER_ARM:
                self.pos_upper_arm = [event.position.x, event.position.y, event.position.z]
                self.rot_upper_arm = [event.rotation.x, event.rotation.y, event.rotation.z, event.rotation.w]
            elif event.joint == Bone.RIGHT_LOWER_ARM:
                self.pos_lower_arm = [event.position.x, event.position.y, event.position.z]
                self.rot_lower_arm = [event.rotation.x, event.rotation.y, event.rotation.z, event.rotation.w]
            elif event.joint == Bone.RIGHT_HAND:
                self.pos_hand = [event.position.x, event.position.y, event.position.z]
                self.rot_hand = [event.rotation.x, event.rotation.y, event.rotation.z, event.rotation.w]

    def _start_vmc_receiver(self):
        """启动VMC接收器"""
        try:
            self.osc.open()
            # 创建接收器
            self.receiver = self.osc.create_receiver(
                self.config.vmc_address, 
                self.config.vmc_port, 
                "receiver1"
            ).open()
            on_receive(self.receiver, RootTransformEvent, self._vmc_received)
            on_receive(self.receiver, BoneTransformEvent, self._vmc_received)
            on_receive(self.receiver, DeviceTransformEvent, self._vmc_received)
            on_receive(self.receiver, BlendShapeEvent, self._vmc_received)
            on_receive(self.receiver, BlendShapeApplyEvent, self._vmc_received)
            on_receive(self.receiver, StateEvent, self._vmc_received)
            on_receive(self.receiver, RelativeTimeEvent, self._vmc_received)
            
            # 在单独线程中运行OSC
            self.vmc_listening = True
            self.vmc_thread = threading.Thread(target=self._run_vmc_osc, daemon=True)
            self.vmc_thread.start()
            
            self.vmc_connected = True
            logging.info(f"VMC接收器启动成功，端口: {self.config.vmc_port}")
            return True
            
        except Exception as e:
            logging.error(f"启动VMC接收器失败: {e}")
            self.vmc_connected = False
            return False

    def _run_vmc_osc(self):
        """在单独线程中运行OSC"""
        while self.vmc_listening:
            try:
                self.osc.run()
                time.sleep(0.001)  # 1ms延迟
            except Exception as e:
                if self.vmc_listening:  # 只有在还在监听时才记录错误
                    logging.error(f"OSC运行错误: {e}")
                break

    def _calculate_qpos(self):
        """基于VMC数据计算机械臂关节位置"""
        qpos = np.zeros(9)  # 初始化控制器输入
        
       # 获取当前欧拉角 - 上臂
        current_euler = quaternion_to_euler(self.rot_upper_arm, seq='XYX')
        corrected_euler = current_euler
        
        # 更新控制器 - 上臂
        qpos[0] = corrected_euler[0] + radians(90)
        qpos[1] = corrected_euler[1]
        qpos[2] = corrected_euler[2] + radians(90)  - radians(180)
        
        # 更新上一帧的角度
        self.previous_euler = corrected_euler.copy()
        

        # 获取小臂绕Z轴的旋转分量
        lower_arm_euler = quaternion_to_euler(self.rot_lower_arm, seq='ZXZ')
        
        lower_arm_z_rotation = lower_arm_euler[0] + radians(90)  # 取Z轴旋转并取反
        
        # 获取手部旋转的欧拉角
        hand_euler = quaternion_to_euler(self.rot_hand, seq='XYZ')
        
        # 将小臂的Z轴旋转只应用到手部的X轴旋转上
        corrected_euler = hand_euler.copy()
        corrected_euler[0] = hand_euler[0] + lower_arm_z_rotation  # 只修改X轴旋转
        
        # 更新控制器
        # qpos[4] = (corrected_euler[0] + radians(90))
        qpos[4] = corrected_euler[0]
        qpos[5] = -corrected_euler[1]
        qpos[6] = corrected_euler[2]
        
        # 更新上一帧的角度
        self.previous_euler2 = corrected_euler.copy()
        
        # 手肘夹角
        q = np.array(self.rot_lower_arm)
        q = q / np.linalg.norm(q)
        
        x, y, z, w = q
        
        # 计算旋转后的X轴
        x_new_x = 1 - 2*y**2 - 2*z**2
        x_new_y = 2*x*y + 2*z*w
        x_new_z = 2*x*z - 2*y*w
        
        x_new = np.array([x_new_x, x_new_y, x_new_z])
        x_orig = np.array([1, 0, 0])
        
        # 计算点积和夹角
        dot_product = np.dot(x_orig, x_new)
        elbow_angle = np.arccos(np.clip(dot_product, -1.0, 1.0))  # 添加clip防止数值错误
        qpos[3] = -elbow_angle
        return qpos

    def connect(self) -> None:
        """连接到VMC设备"""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                "RebocapTeleop is already connected. Do not run `connect()` twice."
            )
        
        # 启动VMC接收器
        if not self._start_vmc_receiver():
            raise DeviceNotConnectedError("Failed to connect to VMC receiver")
        
        logging.info("RebocapTeleop connected successfully")

        self.listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
        self.listener.start()
        logging.info("Keyboard listener started")

    def calibrate(self) -> None:
        """校准设备（VMC不需要特殊校准）"""
        pass

    def configure(self):
        """配置设备"""
        pass

    def get_action(self) -> dict[str, Any]:
        """获取基于VMC数据的动作"""
        before_read_t = time.perf_counter()
        
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "RebocapTeleop is not connected. You need to run `connect()` before `get_action()`."
            )

        try:
            # 计算VMC关节角度
            qpos = self._calculate_qpos()

            for key, val in self.current_pressed.items():
                if key == keyboard.Key.up:
                    gripper -= 1
                elif key == keyboard.Key.down:
                    gripper += 1
            # 低通滤波
            self.gripper = 0.9 * self.gripper + 0.1 * gripper  # 使用当前qpos[7]作为输入
            # 限制夹爪位置在0到1之间
            self.gripper = max(0.0, min(5.0, self.gripper))
            self.current_pressed.clear()

            action_dict = {
                "qpos0": float(qpos[0]),
                "qpos1": float(qpos[1]), 
                "qpos2": float(qpos[2]),
                "qpos3": float(qpos[3]),
                "qpos4": float(qpos[4]),
                "qpos5": float(qpos[5]),
                "qpos6": float(qpos[6]),
                "qpos7": self.gripper,
            }

            self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t
            return action_dict
            
        except Exception as e:
            logging.error(f"获取VMC动作数据时出错: {e}")
            # 返回零位置作为安全措施
            return {
                "qpos0": 0.0,
                "qpos1": 0.0, 
                "qpos2": 0.0,
                "qpos3": 0.0,
                "qpos4": 0.0,
                "qpos5": 0.0,
                "qpos6": 0.0,
                "qpos7": 0.0,
            }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """发送反馈（VMC不需要反馈）"""
        pass

    def disconnect(self) -> None:
        """断开VMC连接"""
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "RebocapTeleop is not connected. You need to run `connect()` before `disconnect()`."
            )
        
        # 停止VMC接收
        self.vmc_listening = False
        if self.vmc_thread and self.vmc_thread.is_alive():
            self.vmc_thread.join(timeout=1.0)
        
        if self.osc:
            self.osc.close()
        
        self.vmc_connected = False
        logging.info("RebocapTeleop disconnected")
