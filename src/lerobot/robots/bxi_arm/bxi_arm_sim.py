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

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode

import time
import mujoco
import mujoco.viewer
import threading

from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
import numpy as np
import cv2
import keyboard
import random

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_bxi_arm import BxiArmSimConfig

logger = logging.getLogger(__name__)


class MuJoCoCamera:
    """MuJoCo仿真相机类"""
    
    def __init__(self, model, data, camera_name, width=640, height=480):
        self.model = model
        self.data = data
        self.camera_name = camera_name
        self.width = width
        self.height = height
        self._connected = False
        
        # 获取相机ID
        try:
            self.camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        except:
            raise ValueError(f"Cannot find camera '{camera_name}' in MuJoCo model")
            
        # 初始化渲染器
        self.renderer = mujoco.Renderer(model, height=self.height, width=self.width)
        
    def connect(self):
        """连接相机"""
        self._connected = True
        logger.info(f"MuJoCo相机 {self.camera_name} 已连接")
    
    def disconnect(self):
        """断开相机连接"""
        self._connected = False
        logger.info(f"MuJoCo相机 {self.camera_name} 已断开连接")
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    def async_read(self):
        """异步读取相机图像"""
        if not self._connected:
            raise DeviceNotConnectedError(f"MuJoCo相机 {self.camera_name} 未连接")
        
        # 使用指定相机渲染图像
        self.renderer.update_scene(self.data, camera=self.camera_id)
        rgb_array = self.renderer.render()
        
        # 转换为BGR格式 (OpenCV格式)
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        
        return bgr_array
    
    def read(self):
        """同步读取相机图像"""
        return self.async_read()


class BxiArmSim(Robot):
    """
    Bxi Arm designed by BXI
    """

    config_class = BxiArmSimConfig
    name = "bxi_arm_sim"

    def __init__(self, config: BxiArmSimConfig):
        super().__init__(config)
        self.config = config
        # norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        # self.bus = FeetechMotorsBus(
        #     port=self.config.port,
        #     motors={
        #         "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
        #         "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
        #         "elbow_flex": Motor(3, "sts3215", norm_mode_body),
        #         "wrist_flex": Motor(4, "sts3215", norm_mode_body),
        #         "wrist_roll": Motor(5, "sts3215", norm_mode_body),
        #         "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
        #     },
        #     calibration=self.calibration,
        # )
        self.model = mujoco.MjModel.from_xml_path('./mjcf/mjmodel.xml')
        # spec = mujoco.MjSpec.from_file('./mjcf/mjmodel.xml')
        # self.model = mujoco.MjModel.from_spec(spec)


        self.data = mujoco.MjData(self.model)
        self.exit_event = threading.Event()
        self.sim_thread = SimulationThread(self.model, self.data, self.exit_event)
        
        # 创建MuJoCo仿真相机
        self.cameras = {}
        for cam_name, cam_config in config.cameras.items():
            # 从相机配置中获取对应的MuJoCo相机名称
            # 假设相机配置中的名字与MJCF文件中的相机名字对应
            mujoco_cam_name = cam_name  # 可以根据需要调整映射关系
            try:
                self.cameras[cam_name] = MuJoCoCamera(
                    self.model, 
                    self.data, 
                    mujoco_cam_name,
                    width=cam_config.width,
                    height=cam_config.height
                )
                logger.info(f"成功创建MuJoCo相机: {cam_name} -> {mujoco_cam_name}")
            except ValueError as e:
                logger.warning(f"无法创建相机 {cam_name}: {e}")

        self.sim_thread.pos = [0.2, 0.2, 0.3, 0]  # 初始位置
        self.rot = [0.0, 0.0, 1.0]  # 初始姿态（欧拉角）
        self.bxi_chain = Chain.from_urdf_file("./mjcf/simplified.urdf")
        self.sim_thread.last_ik = [0.0] * 8
        #     cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        # }

    @property
    def _motors_ft(self) -> dict[str, type]:
        joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.model.njnt)]
        return {f"{name}.pos": float for name in joint_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}
        # state_dim = self.model.njnt
        # return {"state": (state_dim,), **self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        # return self._motors_ft
        # return {"delta_x.pos": float, "delta_y.pos": float, "delta_z.pos": float,"gripper.pos": float}
        return {"x.pos": float, "y.pos": float, "z.pos": float, "g.pos": float}

    @property
    def is_connected(self) -> bool:
        # return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())
        return True

    def connect(self, calibrate: bool = True) -> None:
        # """
        # We assume that at connection time, arm is in a rest position,
        # and torque can be safely disabled to run calibration.
        # """
        # if self.is_connected:
        #     raise DeviceAlreadyConnectedError(f"{self} already connected")

        # self.bus.connect()
        # if not self.is_calibrated and calibrate:
        #     self.calibrate()
        self.sim_thread.daemon = True  # Set the thread as a daemon thread
        self.sim_thread.start()

        for cam in self.cameras.values():
            cam.connect()

        # self.configure()
        # logger.info(f"{self} connected.")
        pass

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def setup_motors(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        # if not self.is_connected:
        #     raise DeviceNotConnectedError(f"{self} is not connected.")

        # # Read arm position
        # start = time.perf_counter()
        # obs_dict = self.bus.sync_read("Present_Position")
        # obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        # dt_ms = (time.perf_counter() - start) * 1e3
        # logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # # Capture images from cameras
        

        # joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) 
        #               for i in range(self.model.njnt)]
        # obs_dict = {f"{motor}.pos": self.data.qpos[i] for i, motor in enumerate(joint_names)}
        obs_dict = self.sim_thread.get_joint_positions()

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            the action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # # Cap goal position when too far away from present position.
        # # /!\ Slower fps expected due to reading from the follower.
        # if self.config.max_relative_target is not None:
        #     present_pos = self.bus.sync_read("Present_Position")
        #     goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
        #     goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # # Send goal position to the arm
        # self.bus.sync_write("Goal_Position", goal_pos)

        # step_start = time.time()

        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        # mujoco.mj_step(self.model, self.data)

        # Example modification of a viewer option: toggle contact points every two seconds.
        # 这一段主要是展示了一个在viewer中添加接触点显示的示例
        # with viewer.lock():
        #     viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        # self.viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        # 这一段是确保了仿真步进的统一
        # time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
        # if time_until_next_step > 0:
        #     time.sleep(time_until_next_step)
        

        # return {f"{motor}.pos": val for motor, val in goal_pos.items()}
        # joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) 
        #               for i in range(self.model.njnt)]
        # return {f"{motor}.pos": self.data.qpos[i] for i, motor in enumerate(joint_names)}
        # print(f"Sending action: {action}")

        # 计算下一步位置
        
        # action_dict = {
        #     "delta_x": delta_x,
        #     "delta_y": delta_y,
        #     "delta_z": delta_z,
        # }

        
        # if self.config.use_gripper:
        #     action_dict["gripper"] = gripper_action
        
        # delta_pos = [action['delta_x.pos']*0.01, action['delta_y.pos']*0.01, action['delta_z.pos']*0.01, action['gripper.pos']*0.001]
        # self.sim_thread.pos =  [self.sim_thread.pos[i] + delta_pos[i] for i in range(4)]
        self.sim_thread.pos = [action['x.pos'], action['y.pos'], action['z.pos'], action['g.pos']/10.0]
        # 确保目标位置在安全范围内
        # if self.sim_thread.pos[3] < 0.0:
        #     self.sim_thread.pos[3] = 0.0
        #     delta_pos[3]=0
        # elif self.sim_thread.pos[3] > 0.05:
        #     self.sim_thread.pos[3] = 0.05
        #     delta_pos[3]=0


        x, y, z,_ = self.sim_thread.pos
        # roll, pitch, yaw = self.rot

        # # 计算旋转矩阵
        # # 绕X轴的旋转 (Roll)
        # Rx = np.array([
        #     [1, 0, 0],
        #     [0, np.cos(roll), -np.sin(roll)],
        #     [0, np.sin(roll), np.cos(roll)]
        # ])

        # # 绕Y轴的旋转 (Pitch)
        # Ry = np.array([
        #     [np.cos(pitch), 0, np.sin(pitch)],
        #     [0, 1, 0],
        #     [-np.sin(pitch), 0, np.cos(pitch)]
        # ])

        # # 绕Z轴的旋转 (Yaw)
        # Rz = np.array([
        #     [np.cos(yaw), -np.sin(yaw), 0],
        #     [np.sin(yaw), np.cos(yaw), 0],
        #     [0, 0, 1]
        # ])

        # # RPY顺序通常是Z-Y-X，即先偏航，再俯仰，最后滚动
        # # 也可以根据具体定义调整旋转顺序
        # R = Rz @ Ry @ Rx

        # # 构建齐次变换矩阵
        # T = np.eye(4)
        # T[:3, :3] = R
        # T[:3, 3] = [x, y, z]

        

        # print(f"目标位置: {self.sim_thread.pos}, 目标姿态: {self.rot}")
        # ik = self.bxi_chain.inverse_kinematics(self.sim_thread.pos)
        # ik = self.bxi_chain.inverse_kinematics(target_position=self.sim_thread.pos,
        #     initial_position=self.sim_thread.last_ik,
        # )
        ik = self.bxi_chain.inverse_kinematics(target_position=self.sim_thread.pos[:3],
            target_orientation=self.rot,
            orientation_mode="Y",
            initial_position=self.sim_thread.last_ik,
            optimizer="fmin_slsqp",
            # optimizer="least_squares"
        )
        # ik=np.insert(ik,0, 0.0)
        # ik=np.insert(ik,4, 0.0)
        # print(f"Inverse Kinematics 结果: {ik}")
        # ik = self.bxi_chain.inverse_kinematics(
        #     target_orientation=self.rot,
        #     orientation_mode="Y",
        #     initial_position=ik,
        #     optimizer="fmin_slsqp",
        #     # optimizer="least_squares"
        # )
        self.sim_thread.last_ik = ik
        #ik反向
        # ik = ik[::-1]

        # 可视化目标点
        # target_site_id = self.model.site("target_site").id
        # j1_id = self.model.joint("joint_link1").id
        # j2_id = self.model.joint("joint_link2").id
        # j3_id = self.model.joint("joint_link3").id
        # j4_id = self.model.joint("joint_link4").id
        # j5_id = self.model.joint("joint_link5").id
        # j6_id = self.model.joint("joint_link6").id
        # j7_id = self.model.joint("joint_link7").id

        # self.data.site_xpos[target_site_id] = [x, y, z]
        self.sim_thread.data.mocap_pos[0] = [x, y, z]


        # 发送动作到仿真环境
        # self.sim_thread.send_action(action)
        # self.data.qpos[j1_id] = ik[0]
        # self.data.qpos[j2_id] = ik[1]
        # self.data.qpos[j3_id] = ik[2]
        # self.data.qpos[j4_id] = ik[3]
        # self.data.qpos[j5_id] = ik[4]
        # self.data.qpos[j6_id] = ik[5]
        # self.data.qpos[j7_id] = ik[6]
        # print(f"发送动作: {ik}")
        
        ik=np.append(ik, self.sim_thread.pos[3])  # 添加末端执行器位置

        # print(f"发送动作: {ik}")
        self.sim_thread.send_action(ik[1:10])

        # print(f"发送动作: {action}")
        # print(f"实际发送动作: {ik[1:10]}")

        # return self.sim_thread.get_joint_positions()
        # return {
        #     "delta_x.pos": delta_pos[0]* 100.0,
        #     "delta_y.pos": delta_pos[1]* 100.0,
        #     "delta_z.pos": delta_pos[2]* 100.0,
        #     "gripper.pos": delta_pos[3] * 1000.0,
        # }
        return {
            "x.pos": self.sim_thread.pos[0],
            "y.pos": self.sim_thread.pos[1],
            "z.pos": self.sim_thread.pos[2],
            "g.pos": self.sim_thread.pos[3]*10.0,
        }

    def disconnect(self):
        logger.info(f"正在断开 {self} 连接...")
        
        # 设置退出事件，通知所有线程停止
        self.exit_event.set()
        
        # 停止仿真线程
        if hasattr(self, 'sim_thread') and self.sim_thread.is_alive():
            self.sim_thread.stop()
            # 等待线程退出，设置超时避免无限等待
            self.sim_thread.join(timeout=5.0)
           
            if self.sim_thread.is_alive():
                logger.warning("仿真线程未能在5秒内正常退出")
            

        # 断开摄像头连接
        for cam_name, cam in self.cameras.items():
            try:
                if cam.is_connected:
                    cam.disconnect()
                    logger.info(f"摄像头 {cam_name} 已断开连接")
            except Exception as e:
                logger.error(f"断开摄像头 {cam_name} 时出错: {e}")

        logger.info(f"{self} 连接已断开")
    
class SimulationThread(threading.Thread):
    def __init__(self, model, data, exit_event):
        super().__init__()
        self.model = model
        self.data = data
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.running = threading.Event()
        self.exit_event = exit_event
        self.qpos = [0.0] * model.nq  # 初始化关节位置

        self.reset()

        #按键重置
        # keyboard.on_press_key('num 0', self.reset)

        # # 生成地板和方块
        # self.model.add('geom', name='floor', type='plane', size=[0, 0, 1], rgba=[0.5, 0.5, 0.5, 1])
        # # self.model.worldbody.add('geom', name='box', type='box', size=[0.1, 0.1, 0.1], pos=[0, 0, 0.05], rgba=[1, 0, 0, 1])
        #  # 添加一个带有可移动方块的 body
        # movable_body = self.model.add('body', name='movable_box_body', pos=[0.5, 0, 0.5])
        # movable_body.add('joint', name='box_joint', type='free', axis=[1, 0, 0], range=[-0.5, 0.5])
        # movable_body.add('geom', name='box', type='box', size=[0.1, 0.1, 0.1], rgba=[0, 0, 1, 1])
    def reset(self):
        """
        重置仿真环境
        """
        logger.info("重置仿真环境")
        # 重置关节位置
        mujoco.mj_resetData(self.model, self.data)
        self.qpos = [0.0] * self.model.nq  # 重置关节位置
        # 重置site位置
        self.pos = [0.2, 0.2, 0.3, 0]  # 初始位置
        self.last_ik = [0.0] * 8

        # # 获取cube的body ID
        # cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        
        # # 设置位置
        # self.data.xpos[cube_body_id] = [0.3+random.random(), random.random(), 0.05]\
        # self.data.mocap_pos[1] = [0.3+random.random(), random.random(), 0.05]
        
        # 获取cube的关节ID（freejoint）
        cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube")
        
        # freejoint的qpos包含7个值：[x, y, z, qw, qx, qy, qz]
        qpos_start = self.model.jnt_qposadr[cube_joint_id]
        
        # 设置位置
        # self.data.qpos[qpos_start:qpos_start+3] = [0.2+random.random()*0.1, random.random()*0.2, 0.05]
        self.data.qpos[qpos_start:qpos_start+3] = [0.2+random.random()*0.1, 0.1+random.random()*0.1, 0.05]

        # self.last_sync_time = time.time()
        # self.last_step_time = time.time()

        
    def run(self):
        logger.info("仿真线程开始运行")
        self.running.set()
        self.last_sync_time = time.time()
        self.last_step_time = time.time()
        while self.running.is_set() and not self.exit_event.is_set():
            try:
                # 计算圆周运动的位置
                # t = time.time()
                # radius = 0.3
                # angular_speed = 1.0
                # self.data.qpos[0] = radius * np.cos(t * angular_speed)
                # self.data.qpos[1] = radius * np.sin(t * angular_speed)
                # self.data.qpos[2] = 0.1
                # step_start = time.time()
                # 步进仿真
                
                while  self.last_step_time + 0.002 < time.time():
                    # for i in range(8):
                    #     self.data.qpos[i] = self.qpos[i]

                    # 阻抗控制，默认kp=100, kd=1
                    # kp = 5.0
                    # kd = 0.5
                    kps = [10,500,500,500,100,100,100,100]
                    kds = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
                    for i in range(7):
                        error = self.qpos[i] - self.data.qpos[i]
                        self.data.ctrl[i] = self.qpos[i]
                        # self.data.ctrl[i] = kps[i] * error - kds[i] * self.data.qvel[i]
                        # self.data.qvel[i] = kp * error - kd * self.data.qvel[i]
                    # kp = 100.0
                    # kd = 1.0
                    kp = 50000
                    kd = 100
                    for i in range(7,8):
                        error = self.qpos[i] - self.data.qpos[i]
                        self.data.ctrl[i] = kp * error - kd * self.data.qvel[i]
                        # self.data.qvel[i] = kp * error - kd * self.data.qvel[i]

                    if keyboard.is_pressed('num 0'):
                        self.reset()

                    mujoco.mj_step(self.model, self.data)
                    self.last_step_time = self.last_step_time+0.002
                
                
                # time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                # if time_until_next_step > 0:
                    # time.sleep(time_until_next_step)
                
                if self.last_sync_time + 1/60.0 < time.time():
                    with self.viewer.lock():
                        self.viewer.sync()
                    self.last_sync_time = time.time()
                # time.sleep(0.002)
                
            except Exception as e:
                logger.error(f"仿真线程错误")
                self.exit_event.set()
                break
                
        logger.info("仿真线程结束运行")
        # 线程退出前关闭viewer
        try:
            if self.viewer is not None:
                self.viewer.close()
        except Exception as e:
            logger.error(f"关闭viewer时出错: {e}")

    def stop(self):
        logger.info("正在停止仿真线程...")
        self.viewer.close()
        self.running.clear()
        # 这里不直接关闭viewer，交由run()退出时关闭

    def send_action(self, action: list[9]):
        """
        发送动作到仿真环境
        """
        if not self.running.is_set():
            raise DeviceNotConnectedError("仿真线程未运行，无法发送动作。")
        
        # 将动作应用到仿真数据中
        for i in range(8):
            self.qpos[i] = action[i]
        # print(action)

        
        # 这里可以添加更多的动作处理逻辑
        # 例如更新关节速度、力等
        # self.data.qvel[i] = joint_velocity[i]

    
    def get_joint_positions(self) -> dict[str, float]:
        """
        获取关节位置
        """
        joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) 
                      for i in range(self.model.njnt)]
        obs = {f"{motor}.pos": self.data.qpos[i] for i, motor in enumerate(joint_names)}
        # print(f"获取关节位置: {obs}")
        obs["joint_gripper_l.pos"] = obs["joint_gripper_l.pos"]*100
        obs["joint_gripper_r.pos"] = obs["joint_gripper_r.pos"]*100
        return obs
    
    def get_observation(self) -> dict[str, Any]:
        """
        获取仿真环境的观测数据
        """
        obs_dict = {}
        joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) 
                      for i in range(self.model.njnt)]
        obs_dict.update({f"{motor}.pos": self.data.qpos[i] for i, motor in enumerate(joint_names)})

        # 可以添加其他观测数据，例如传感器数据等
        return obs_dict
