#!/usr/bin/env python3
"""Run the recorded R1 right-arm polishing motion in locomotion mode.

Created: 2026-08-24
Mode: Locomotion (FSM 811); built-in standing/balance control remains active
Topics: reads rt/lowstate; writes rt/arm_sdk at 100 Hz
Purpose: move once to Pose 1, then repeat Pose 2 <-> Pose 3 as a smooth,
synchronized polishing action. Defaults: 120 degrees/second and 0.05 s hold.
"""

import argparse
import json
import math
import signal
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.r1.loco.r1_loco_api import ROBOT_API_ID_LOCO_GET_FSM_ID
from unitree_sdk2py.r1.loco.r1_loco_client import LocoClient
from unitree_sdk2py.utils.crc import CRC


LOCOMOTION_FSM_ID = 811
CONTROL_PERIOD = 0.01  # 100 Hz for rt/arm_sdk.
STATE_TIMEOUT = 1.0
WEIGHT_RAMP_DURATION = 1.0
MINIMUM_JERK_PEAK_SLOPE = 1.875


@dataclass(frozen=True)
class Joint:
    name: str
    index: int
    kp: float
    kd: float


RIGHT_ARM = (
    Joint("Right shoulder pitch", 22, 50.0, 2.0),
    Joint("Right shoulder roll", 23, 50.0, 2.0),
    Joint("Right shoulder yaw", 24, 40.0, 2.0),
    Joint("Right elbow", 25, 40.0, 2.0),
    Joint("Right wrist roll", 26, 30.0, 2.0),
)

# mode_pr claims all these upper-body joints. They are held at their measured
# startup positions while only the right arm performs the polishing motion.
HELD_JOINTS = (
    Joint("Waist yaw", 13, 50.0, 3.0),
    Joint("Left shoulder pitch", 15, 50.0, 2.0),
    Joint("Left shoulder roll", 16, 50.0, 2.0),
    Joint("Left shoulder yaw", 17, 40.0, 2.0),
    Joint("Left elbow", 18, 40.0, 2.0),
    Joint("Left wrist roll", 19, 30.0, 2.0),
    Joint("Head pitch", 29, 15.0, 1.0),
    Joint("Head yaw", 30, 15.0, 1.0),
)
SDK_JOINTS = RIGHT_ARM + HELD_JOINTS
Pose = Dict[int, float]

POSE_1_VALUES = (-0.092380, 0.015358, 0.145309, -0.068757, 0.064654)
POSE_2_VALUES = (0.375244, POSE_1_VALUES[1], POSE_1_VALUES[2], -0.464781, POSE_1_VALUES[4])
POSE_3_VALUES = (-0.618961, POSE_1_VALUES[1], POSE_1_VALUES[2], 0.633055, POSE_1_VALUES[4])


def make_pose(values: Tuple[float, ...]) -> Pose:
    return {joint.index: value for joint, value in zip(RIGHT_ARM, values)}


POSE_1 = make_pose(POSE_1_VALUES)
POSE_2 = make_pose(POSE_2_VALUES)
POSE_3 = make_pose(POSE_3_VALUES)
REPEATING_SEQUENCE = (("Pose 2", POSE_2), ("Pose 3", POSE_3))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the R1 right-arm polishing action through rt/arm_sdk."
    )
    parser.add_argument("network_interface", help="Robot network interface, e.g. enp2s0")
    parser.add_argument(
        "--cycles", type=int, default=0,
        help="Pose 2 -> Pose 3 cycles; 0 repeats until Ctrl+C (default: 0)",
    )
    parser.add_argument(
        "--speed-deg-s", type=float, default=120.0,
        help="Peak joint speed (default: 120 degrees/second)",
    )
    parser.add_argument(
        "--hold", type=float, default=0.05,
        help="Seconds to hold each pose (default: 0.05)",
    )
    args = parser.parse_args()
    if args.cycles < 0:
        parser.error("--cycles must be zero or greater")
    if args.speed_deg_s <= 0.0:
        parser.error("--speed-deg-s must be greater than zero")
    if args.hold < 0.0:
        parser.error("--hold must be zero or greater")
    return args


def get_fsm_id(client: LocoClient) -> Tuple[int, Optional[int]]:
    code, data = client._Call(ROBOT_API_ID_LOCO_GET_FSM_ID, "{}")
    if code != 0:
        return code, None
    try:
        return code, int(json.loads(data)["data"])
    except (TypeError, ValueError, KeyError, json.JSONDecodeError):
        return code, None


class PolishingController:
    def __init__(self, speed: float) -> None:
        self.speed = speed
        self.stop_event = threading.Event()
        self.state_ready = threading.Event()
        self.lock = threading.Lock()
        self.last_state_time = 0.0
        self.measured: Pose = {}
        self.commanded: Pose = {}
        self.weight = 0.0
        self.command = unitree_hg_msg_dds__LowCmd_()
        self.publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.crc = CRC()

    def initialize(self) -> None:
        self.publisher.Init()
        self.subscriber.Init(self._handle_state, 10)
        if not self.state_ready.wait(5.0):
            raise RuntimeError("No rt/lowstate sample received within 5 seconds")
        with self.lock:
            self.commanded = dict(self.measured)
            for joint in SDK_JOINTS:
                motor = self.command.motor_cmd[joint.index]
                motor.tau = 0.0
                motor.q = self.commanded[joint.index]
                motor.dq = 0.0
                motor.kp = joint.kp
                motor.kd = joint.kd

    def _handle_state(self, msg: LowState_) -> None:
        with self.lock:
            self.last_state_time = time.monotonic()
            self.measured = {
                joint.index: float(msg.motor_state[joint.index].q)
                for joint in SDK_JOINTS
            }
            self.state_ready.set()

    def _write(self) -> None:
        with self.lock:
            for joint in SDK_JOINTS:
                self.command.motor_cmd[joint.index].q = self.commanded[joint.index]
            self.command.mode_pr = round(self.weight * 100.0)
            self.command.crc = self.crc.Crc(self.command)
        self.publisher.Write(self.command)

    def ramp_weight(self, target: float) -> bool:
        steps = max(1, round(WEIGHT_RAMP_DURATION / CONTROL_PERIOD))
        start = self.weight
        next_tick = time.monotonic()
        for step in range(steps):
            if self.stop_event.is_set() and target > 0.0:
                return False
            ratio = (step + 1) / steps
            self.weight = start + (target - start) * ratio
            self._write()
            next_tick += CONTROL_PERIOD
            time.sleep(max(0.0, next_tick - time.monotonic()))
        return True

    def move_to(self, target: Pose) -> bool:
        with self.lock:
            start = {joint.index: self.commanded[joint.index] for joint in RIGHT_ARM}
        maximum_distance = max(
            abs(target[joint.index] - start[joint.index]) for joint in RIGHT_ARM
        )
        duration = MINIMUM_JERK_PEAK_SLOPE * maximum_distance / self.speed
        if duration == 0.0:
            self._write()
            return True

        start_time = time.monotonic()
        next_tick = start_time
        while not self.stop_event.is_set():
            now = time.monotonic()
            elapsed_ratio = min(1.0, (now - start_time) / duration)
            ratio = 10.0 * elapsed_ratio**3 - 15.0 * elapsed_ratio**4 + 6.0 * elapsed_ratio**5
            with self.lock:
                if now - self.last_state_time > STATE_TIMEOUT:
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

    def close(self) -> None:
        self.ramp_weight(0.0)
        self.subscriber.Close()
        self.publisher.Close()


def main() -> int:
    args = parse_args()
    print("WARNING: This action takes control of the R1 upper body through rt/arm_sdk.")
    print("The robot must stand still in locomotion mode (FSM 811).")
    print(f"Speed={args.speed_deg_s:g} deg/s, hold={args.hold:g} s")
    if input("Type POLISH to continue: ").strip() != "POLISH":
        print("Cancelled; no arm command was sent.")
        return 1

    ChannelFactoryInitialize(0, args.network_interface)
    client = LocoClient()
    client.SetTimeout(3.0)
    client.Init()
    code, fsm_id = get_fsm_id(client)
    if code != 0 or fsm_id != LOCOMOTION_FSM_ID:
        print(
            f"Cannot start: expected FSM {LOCOMOTION_FSM_ID}, "
            f"received code={code}, fsm={fsm_id}."
        )
        return 2

    controller = PolishingController(math.radians(args.speed_deg_s))
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
        if controller.ramp_weight(1.0) and controller.move_to(POSE_1):
            print("Reached Pose 1 (starting pose)")
            controller.hold(args.hold)

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
                print(f"Completed polishing cycle {completed}")
                continue
            break
    except RuntimeError as exc:
        print(f"Safety stop: {exc}")
        exit_code = 4
    finally:
        print("Releasing rt/arm_sdk control...")
        controller.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
