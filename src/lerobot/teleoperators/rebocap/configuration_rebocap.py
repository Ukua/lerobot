#!/usr/bin/env python

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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("rebocap")
@dataclass
class RebocapTeleopConfig(TeleoperatorConfig):
    """
    Configuration for RebocapTeleop using VMC motion capture
    """
    # VMC接收端口
    vmc_port: int = 39539
    # VMC接收地址
    vmc_address: str = "0.0.0.0"
    # 是否启用夹爪控制
    use_gripper: bool = True
    # 角度跳变检测阈值
    angle_discontinuity_threshold: float = 3.14159
