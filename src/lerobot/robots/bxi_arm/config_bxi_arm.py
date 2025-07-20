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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("bxi_arm")
@dataclass
class BxiArmConfig(RobotConfig):
    # Port to connect to the arm
    # port: str

    # true for simulation, false for real robot
    simulate: bool = False

    # show the robot in sim,no matter if the robot is simulated or not
    visible: bool = True

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    # ROS2 topic name for publishing joint positions
    ros_topic_name: str = "robot_qpos"

    # ROS2 node name
    ros_node_name: str = "bxi_arm_control_node"

    # Number of joints (including gripper)
    num_joints: int = 8

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

@RobotConfig.register_subclass("bxi_arm_sim")
@dataclass
class BxiArmSimConfig(RobotConfig):
    # Port to connect to the arm
    # port: str

    # true for simulation, false for real robot
    simulate: bool = True

    # show the robot in sim,no matter if the robot is simulated or not
    visible: bool = True

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
