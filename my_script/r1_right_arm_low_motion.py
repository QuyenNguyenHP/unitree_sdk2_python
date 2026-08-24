#!/usr/bin/env python3
"""Repeat recorded R1 right-arm poses in development mode via rt/lowcmd.

Created: 2026-08-24
Mode: Development/debug low-level mode; robot must be securely suspended
Topics: reads rt/lowstate; writes rt/lowcmd at 500 Hz
Purpose: move once to Pose 1, then repeat Pose 2 <-> Pose 3 using synchronized
minimum-jerk shoulder-pitch and elbow motion.
"""

import argparse
import math
import signal
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


CONTROL_PERIOD = 0.002  # Unitree's R1 low-level example publishes at 500 Hz.
STATE_TIMEOUT = 1.0
MODE_PR = 0
MINIMUM_JERK_PEAK_SLOPE = 1.875


@dataclass(frozen=True)
class Joint:
    name: str
    index: int
    kp: float
    kd: float


RIGHT_ARM = (
    Joint("Right shoulder pitch", 22, 100.0, 2.0),
    Joint("Right shoulder roll", 23, 100.0, 2.0),
    Joint("Right shoulder yaw", 24, 100.0, 2.0),
    Joint("Right elbow", 25, 100.0, 2.0),
    Joint("Right wrist roll", 26, 50.0, 2.0),
)

# Recorded values are ordered as shoulder pitch, shoulder roll, shoulder yaw,
# elbow and wrist roll. Pose 2/3 intentionally retain Pose 1 for the three
# joints that should not move.
POSE_1_VALUES = (-0.092380, 0.015358, 0.145309, -0.068757, 0.064654)
POSE_2_VALUES = (0.375244, POSE_1_VALUES[1], POSE_1_VALUES[2], -0.464781, POSE_1_VALUES[4])
POSE_3_VALUES = (-0.618961, POSE_1_VALUES[1], POSE_1_VALUES[2], 0.633055, POSE_1_VALUES[4])


def make_pose(values: Tuple[float, ...]) -> Dict[int, float]:
    return {joint.index: value for joint, value in zip(RIGHT_ARM, values)}


POSE_1 = make_pose(POSE_1_VALUES)
POSE_2 = make_pose(POSE_2_VALUES)
POSE_3 = make_pose(POSE_3_VALUES)
REPEATING_SEQUENCE = (("Pose 2", POSE_2), ("Pose 3", POSE_3))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeat three recorded R1 right-arm poses using rt/lowcmd."
    )
    parser.add_argument("network_interface", help="Robot network interface, e.g. enp2s0")
    parser.add_argument(
        "--cycles", type=int, default=0,
        help="Number of cycles; 0 repeats until Ctrl+C (default: 0)",
    )
    parser.add_argument(
        "--speed-deg-s", type=float, default=60.0,
        help="Maximum joint target speed (default: 60 degrees/second)",
    )
    parser.add_argument(
        "--hold", type=float, default=1.0,
        help="Seconds to hold each reached pose (default: 1.0)",
    )
    args = parser.parse_args()
    if args.cycles < 0:
        parser.error("--cycles must be zero or greater")
    if args.speed_deg_s <= 0.0:
        parser.error("--speed-deg-s must be greater than zero")
    if args.hold < 0.0:
        parser.error("--hold must be zero or greater")
    return args


class LowMotionController:
    def __init__(self, speed: float) -> None:
        self.speed = speed
        self.stop_event = threading.Event()
        self.state_ready = threading.Event()
        self.lock = threading.Lock()
        self.last_state_time = 0.0
        self.mode_machine = 0
        self.measured: Dict[int, float] = {}
        self.commanded: Dict[int, float] = {}
        self.publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.command = unitree_hg_msg_dds__LowCmd_()
        self.crc = CRC()

    def initialize(self) -> None:
        self.publisher.Init()
        self.subscriber.Init(self._handle_state, 10)
        if not self.state_ready.wait(5.0):
            raise RuntimeError("No rt/lowstate sample received within 5 seconds")

        with self.lock:
            self.commanded = dict(self.measured)
            self.command.mode_pr = MODE_PR
            self.command.mode_machine = self.mode_machine
            for joint in RIGHT_ARM:
                motor = self.command.motor_cmd[joint.index]
                motor.mode = 1
                motor.tau = 0.0
                motor.q = self.commanded[joint.index]
                motor.dq = 0.0
                motor.kp = joint.kp
                motor.kd = joint.kd

    def _handle_state(self, msg: LowState_) -> None:
        with self.lock:
            self.mode_machine = int(msg.mode_machine)
            self.last_state_time = time.monotonic()
            self.measured = {
                joint.index: float(msg.motor_state[joint.index].q)
                for joint in RIGHT_ARM
            }
            self.state_ready.set()

    def _write(self) -> None:
        with self.lock:
            self.command.mode_machine = self.mode_machine
            for joint in RIGHT_ARM:
                self.command.motor_cmd[joint.index].q = self.commanded[joint.index]
            self.command.crc = self.crc.Crc(self.command)
        self.publisher.Write(self.command)

    def move_to(self, target: Dict[int, float]) -> bool:
        # Use one minimum-jerk interpolation ratio for every joint. The
        # quintic curve has zero velocity and acceleration at both ends, so
        # shoulder pitch and elbow start/stop together without a hard step in
        # velocity or acceleration.
        with self.lock:
            start = dict(self.commanded)
        maximum_distance = max(
            abs(target[joint.index] - start[joint.index]) for joint in RIGHT_ARM
        )
        # The peak slope of 10t^3 - 15t^4 + 6t^5 is 1.875. Include it in the
        # duration calculation so --speed-deg-s remains a true peak limit.
        duration = MINIMUM_JERK_PEAK_SLOPE * maximum_distance / self.speed
        if duration == 0.0:
            self._write()
            return True

        start_time = time.monotonic()
        next_tick = time.monotonic()
        while not self.stop_event.is_set():
            elapsed_ratio = min(1.0, (time.monotonic() - start_time) / duration)
            ratio = (
                10.0 * elapsed_ratio**3
                - 15.0 * elapsed_ratio**4
                + 6.0 * elapsed_ratio**5
            )
            with self.lock:
                if time.monotonic() - self.last_state_time > STATE_TIMEOUT:
                    raise RuntimeError("rt/lowstate timed out")
                for joint in RIGHT_ARM:
                    self.commanded[joint.index] = start[joint.index] + ratio * (
                        target[joint.index] - start[joint.index]
                    )
            self._write()
            if elapsed_ratio >= 1.0:
                return True
            next_tick += CONTROL_PERIOD
            self.stop_event.wait(max(0.0, next_tick - time.monotonic()))
        return False

    def hold(self, duration: float) -> bool:
        deadline = time.monotonic() + duration
        next_tick = time.monotonic()
        while not self.stop_event.is_set() and time.monotonic() < deadline:
            with self.lock:
                if time.monotonic() - self.last_state_time > STATE_TIMEOUT:
                    raise RuntimeError("rt/lowstate timed out")
            self._write()
            next_tick += CONTROL_PERIOD
            self.stop_event.wait(max(0.0, next_tick - time.monotonic()))
        return not self.stop_event.is_set()

    def disable_and_close(self) -> None:
        # Reduce stiffness smoothly before disabling these five motors.
        steps = 100
        for step in range(steps):
            ratio = 1.0 - (step + 1) / steps
            for joint in RIGHT_ARM:
                motor = self.command.motor_cmd[joint.index]
                motor.kp = joint.kp * ratio
                motor.kd = joint.kd * ratio
            self._write()
            time.sleep(CONTROL_PERIOD)
        for joint in RIGHT_ARM:
            self.command.motor_cmd[joint.index].mode = 0
        for _ in range(10):
            self._write()
            time.sleep(CONTROL_PERIOD)
        self.subscriber.Close()
        self.publisher.Close()


def release_active_motion_service() -> None:
    switcher = MotionSwitcherClient()
    switcher.SetTimeout(5.0)
    switcher.Init()
    for _ in range(10):
        code, mode = switcher.CheckMode()
        if code != 0:
            raise RuntimeError(f"MotionSwitcher CheckMode failed with code {code}")
        if not mode or not mode.get("name"):
            return
        print(f"Releasing active motion service: {mode['name']}")
        code, _ = switcher.ReleaseMode()
        if code != 0:
            raise RuntimeError(f"MotionSwitcher ReleaseMode failed with code {code}")
        time.sleep(1.0)
    raise RuntimeError("Active motion service did not release")


def print_poses() -> None:
    for label, pose in (("Pose 1", POSE_1), ("Pose 2", POSE_2), ("Pose 3", POSE_3)):
        values = ", ".join(f"{pose[j.index]:+.6f}" for j in RIGHT_ARM)
        print(f"{label}: [{values}]")


def main() -> int:
    args = parse_args()
    print("DANGER: rt/lowcmd does not provide standing or balance control.")
    print("Suspend the R1 securely and keep the emergency stop accessible.")
    print("Enter development/debug mode before continuing.")
    print_poses()
    if input("Type LOW MOTION to continue: ").strip() != "LOW MOTION":
        print("Cancelled; no command was sent.")
        return 1

    ChannelFactoryInitialize(0, args.network_interface)
    try:
        release_active_motion_service()
    except RuntimeError as exc:
        print(f"Cannot obtain low-level control: {exc}")
        return 2

    controller = LowMotionController(math.radians(args.speed_deg_s))
    try:
        controller.initialize()
    except RuntimeError as exc:
        controller.subscriber.Close()
        controller.publisher.Close()
        print(f"Cannot start: {exc}")
        return 3

    def request_stop(_signum: int, _frame: Optional[object]) -> None:
        controller.stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    completed = 0
    exit_code = 0
    try:
        # Pose 1 is the starting pose and is reached only once.
        if controller.move_to(POSE_1):
            print("Reached Pose 1 (starting pose)")
            controller.hold(args.hold)

        # After startup, alternate Pose 2 and Pose 3 continuously.
        while not controller.stop_event.is_set() and (
            args.cycles == 0 or completed < args.cycles
        ):
            for label, pose in REPEATING_SEQUENCE:
                if not controller.move_to(pose):
                    break
                print(f"Reached {label}")
                if not controller.hold(args.hold):
                    break
            else:
                completed += 1
                print(f"Completed Pose 2 -> Pose 3 cycle {completed}")
                continue
            break
    except RuntimeError as exc:
        print(f"Safety stop: {exc}")
        exit_code = 4
    finally:
        print("Reducing right-arm stiffness and disabling its motors...")
        controller.disable_and_close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
