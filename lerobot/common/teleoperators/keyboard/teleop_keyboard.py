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

import logging
import os
import sys
import time
from queue import Queue
from typing import Any

import pygame
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .configuration_keyboard import KeyboardEndEffectorTeleopConfig, KeyboardTeleopConfig

import keyboard

# PYNPUT_AVAILABLE = True
# try:
#     if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
#         logging.info("No DISPLAY set. Skipping pynput import.")
#         raise ImportError("pynput blocked intentionally due to no display.")

#     from pynput import keyboard
# except ImportError:
#     keyboard = None
#     PYNPUT_AVAILABLE = False
# except Exception as e:
#     keyboard = None
#     PYNPUT_AVAILABLE = False
#     logging.info(f"Could not import pynput: {e}")


class KeyboardTeleop(Teleoperator):
    """
    Teleop class to use keyboard inputs for control.
    """

    config_class = KeyboardTeleopConfig
    name = "keyboard"

    def __init__(self, config: KeyboardTeleopConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.event_queue = Queue()
        self.current_pressed = {}
        self.listener = None
        self.logs = {}

    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (len(self.arm),),
            "names": {"motors": list(self.arm.motors)},
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return True
        # return PYNPUT_AVAILABLE and isinstance(self.listener, keyboard.Listener) and self.listener.is_alive()

    @property
    def is_calibrated(self) -> bool:
        pass

    def connect(self) -> None:
        pass
        # if self.is_connected:
        #     raise DeviceAlreadyConnectedError(
        #         "Keyboard is already connected. Do not run `robot.connect()` twice."
        #     )

        # if PYNPUT_AVAILABLE:
        #     logging.info("pynput is available - enabling local keyboard listener.")
        #     self.listener = keyboard.Listener(
        #         on_press=self._on_press,
        #         on_release=self._on_release,
        #     )
        #     self.listener.start()
        # else:
        #     logging.info("pynput not available - skipping local keyboard listener.")
        #     self.listener = None

    def calibrate(self) -> None:
        pass

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

    def configure(self):
        pass

    def get_action(self) -> dict[str, Any]:
        before_read_t = time.perf_counter()
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "KeyboardTeleop is not connected. You need to run `connect()` before `get_action()`."
            )

        self._drain_pressed_keys()

        # Generate action based on current key states
        action = {key for key, val in self.current_pressed.items() if val}
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        return dict.fromkeys(action, None)

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "KeyboardTeleop is not connected. You need to run `robot.connect()` before `disconnect()`."
            )
        if self.listener is not None:
            self.listener.stop()


class KeyboardEndEffectorTeleop(KeyboardTeleop):
    """
    Teleop class to use keyboard inputs for end effector control.
    Designed to be used with the `So100FollowerEndEffector` robot.
    """

    config_class = KeyboardEndEffectorTeleopConfig
    name = "keyboard_ee"

    def __init__(self, config: KeyboardEndEffectorTeleopConfig):
        super().__init__(config)
        self.config = config
        self.misc_keys_queue = Queue()
        self.joystick = None
        pygame.init()
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            logging.info(f"Gamepad '{self.joystick.get_name()}' connected.")
        else:
            logging.warning("No gamepad connected.")

    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (4,),
                "names": {"delta_x.pos": 0, "delta_y.pos": 1, "delta_z.pos": 2, "gripper.pos": 3},
            }
        else:
            return {
                "dtype": "float32",
                "shape": (3,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2},
            }

    def _on_press(self, key):
        if hasattr(key, "char"):
            key = key.char
        self.event_queue.put((key, True))

    def _on_release(self, key):
        if hasattr(key, "char"):
            key = key.char
        self.event_queue.put((key, False))

    def disconnect(self) -> None:
        super().disconnect()
        if self.joystick:
            self.joystick.quit()
        pygame.quit()

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "KeyboardTeleop is not connected. You need to run `connect()` before `get_action()`."
            )

        self._drain_pressed_keys()
        delta_x = 0.0
        delta_y = 0.0
        delta_z = 0.0
        gripper_action = 0.0  # default gripper action is to stay

        # Gamepad control
        if self.joystick:
            pygame.event.get()  # Pump events
            # Axes: 0: left stick horizontal, 1: left stick vertical, 2: right stick horizontal, 3: right stick vertical
            # Note: Pygame axes may be inverted depending on the controller.
            # This mapping assumes a standard controller layout (e.g., Xbox).
            # Left stick for x/y movement
            axis_y = -self.joystick.get_axis(1)  # Left stick vertical
            axis_x = -self.joystick.get_axis(0)  # Left stick horizontal
            # Right stick for z movement
            axis_z = self.joystick.get_axis(3) # Right stick vertical

            
            #左右扳机：夹爪开合
            # Left trigger for closing the gripper, right trigger for opening
            gripper_action = -self.joystick.get_axis(4)+ self.joystick.get_axis(5)



            # Deadzone to prevent drift
            dead_zone = 0.1
            if abs(axis_x) > dead_zone:
                delta_x = -axis_x  # Invert for intuitive control
            if abs(axis_y) > dead_zone:
                delta_y = axis_y
            if abs(axis_z) > dead_zone:
                delta_z = -axis_z

            # # Buttons for gripper: 0: A, 1: B, 2: X, 3: Y
            # if self.joystick.get_button(0):  # 'A' button to close
            #     gripper_action = 0
            # elif self.joystick.get_button(1):  # 'B' button to open
            #     gripper_action = 2

        # Keyboard control (overwrites gamepad)
        if keyboard.is_pressed("num 8"):
            delta_x = 1.0
        elif keyboard.is_pressed("num 2"):
            delta_x = -1.0
        if keyboard.is_pressed("num 4"):
            delta_y = 1.0
        elif keyboard.is_pressed("num 6"):
            delta_y = -1.0
        if keyboard.is_pressed("num 7"):
            delta_z = 1.0
        elif keyboard.is_pressed("num 9"):
            delta_z = -1.0
        if keyboard.is_pressed("num 1"):
            gripper_action = 1
        elif keyboard.is_pressed("num 3"):
            gripper_action = -1


        # Generate action based on current key states
        # for key, val in self.current_pressed.items():
        #     if key == keyboard.Key.up:
        #         delta_x = int(val)
        #     elif key == keyboard.Key.down:
        #         delta_x = -int(val)
        #     elif key == keyboard.Key.left:
        #         delta_y = int(val)
        #     elif key == keyboard.Key.right:
        #         delta_y = -int(val)
        #     elif key == keyboard.Key.shift:
        #         delta_z = -int(val)
        #     elif key == keyboard.Key.shift_r:
        #         delta_z = int(val)
        #     elif key == keyboard.Key.ctrl_r:
        #         # Gripper actions are expected to be between 0 (close), 1 (stay), 2 (open)
        #         gripper_action = int(val) + 1
        #     elif key == keyboard.Key.ctrl_l:
        #         gripper_action = int(val) - 1
        #     elif val:
        #         # If the key is pressed, add it to the misc_keys_queue
        #         # this will record key presses that are not part of the delta_x, delta_y, delta_z
        #         # this is useful for retrieving other events like interventions for RL, episode success, etc.
        #         self.misc_keys_queue.put(key)

        self.current_pressed.clear()

        action_dict = {
            "delta_x.pos": delta_x,
            "delta_y.pos": delta_y,
            "delta_z.pos": delta_z,
            "gripper.pos": gripper_action,
        }

        if self.config.use_gripper:
            action_dict["gripper.pos"] = gripper_action

        # print(f"Current action: {action_dict}")
        return action_dict
