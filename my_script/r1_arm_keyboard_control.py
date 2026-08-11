#!/usr/bin/env python3
"""Control individual Unitree R1 arm joints from the keyboard via rt/arm_sdk."""

import argparse
import json
import math
import os
import select
import signal
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.r1.loco.r1_loco_api import ROBOT_API_ID_LOCO_GET_FSM_ID
from unitree_sdk2py.r1.loco.r1_loco_client import LocoClient
from unitree_sdk2py.utils.crc import CRC


LOCOMOTION_FSM_ID = 811
CONTROL_PERIOD = 0.01
STATE_TIMEOUT = 1.0
WEIGHT_RAMP_DURATION = 1.0


@dataclass(frozen=True)
class JointSpec:
    key: str
    name: str
    index: int
    kp: float
    kd: float


# These are the ten arm joints exposed by Unitree R1 ArmSdk, in its official
# ordering. R1 has five controlled joints per arm.
ARM_JOINTS = (
    JointSpec("1", "Left shoulder pitch", 15, 50.0, 2.0),
    JointSpec("2", "Left shoulder roll", 16, 50.0, 2.0),
    JointSpec("3", "Left shoulder yaw", 17, 40.0, 2.0),
    JointSpec("4", "Left elbow", 18, 40.0, 2.0),
    JointSpec("5", "Left wrist roll", 19, 30.0, 2.0),
    JointSpec("6", "Right shoulder pitch", 22, 50.0, 2.0),
    JointSpec("7", "Right shoulder roll", 23, 50.0, 2.0),
    JointSpec("8", "Right shoulder yaw", 24, 40.0, 2.0),
    JointSpec("9", "Right elbow", 25, 40.0, 2.0),
    JointSpec("0", "Right wrist roll", 26, 30.0, 2.0),
)

# mode_pr weight also claims these three upper-body joints. They must be held
# at measured startup positions even though this script only edits the arms.
PASSIVE_UPPER_BODY_JOINTS = (
    JointSpec("", "Waist yaw", 13, 50.0, 3.0),
    JointSpec("", "Head pitch", 29, 15.0, 1.0),
    JointSpec("", "Head yaw", 30, 15.0, 1.0),
)
ALL_SDK_JOINTS = ARM_JOINTS + PASSIVE_UPPER_BODY_JOINTS


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def move_toward(current: float, target: float, maximum_delta: float) -> float:
    return current + clamp(target - current, -maximum_delta, maximum_delta)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Control individual R1 arm joints through rt/arm_sdk."
    )
    parser.add_argument(
        "network_interface",
        help="Network interface connected to the robot, for example enp2s0",
    )
    parser.add_argument(
        "--step-deg",
        type=float,
        default=3.0,
        help="Target change for each Up/Down press (default: 3 degrees)",
    )
    parser.add_argument(
        "--travel-deg",
        type=float,
        default=30.0,
        help="Maximum deviation from each startup angle (default: 30 degrees)",
    )
    parser.add_argument(
        "--max-speed-deg-s",
        type=float,
        default=30.0,
        help="Maximum joint target slew rate (default: 30 degrees/second)",
    )
    args = parser.parse_args()

    if args.step_deg <= 0.0:
        parser.error("--step-deg must be greater than zero")
    if not 0.0 < args.travel_deg <= 60.0:
        parser.error("--travel-deg must be greater than zero and at most 60")
    if args.max_speed_deg_s <= 0.0:
        parser.error("--max-speed-deg-s must be greater than zero")
    if not sys.stdin.isatty():
        parser.error("keyboard control requires an interactive terminal")
    return args


def get_fsm_id(client: LocoClient) -> Tuple[int, Optional[int]]:
    code, data = client._Call(ROBOT_API_ID_LOCO_GET_FSM_ID, "{}")
    if code != 0:
        return code, None
    try:
        return code, int(json.loads(data)["data"])
    except (TypeError, ValueError, KeyError, json.JSONDecodeError):
        return code, None


class ArmController:
    def __init__(self, travel: float, maximum_speed: float) -> None:
        self.travel = travel
        self.maximum_speed = maximum_speed

        self._lock = threading.Lock()
        self._state_ready = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._active = False
        self._weight = 0.0
        self._last_state_time = 0.0
        self._fault: Optional[str] = None
        self._initial: Dict[int, float] = {}
        self._target: Dict[int, float] = {}
        self._command: Dict[int, float] = {}
        self._measured: Dict[int, float] = {}

        self._crc = CRC()
        self._cmd = unitree_hg_msg_dds__LowCmd_()
        self._publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self._subscriber = ChannelSubscriber("rt/lowstate", LowState_)

    @property
    def fault(self) -> Optional[str]:
        with self._lock:
            return self._fault

    def initialize(self) -> None:
        self._publisher.Init()
        self._subscriber.Init(self._handle_low_state, 10)

    def _handle_low_state(self, msg: LowState_) -> None:
        now = time.monotonic()
        with self._lock:
            self._last_state_time = now
            for spec in ALL_SDK_JOINTS:
                self._measured[spec.index] = float(msg.motor_state[spec.index].q)

            if not self._state_ready.is_set():
                for spec in ALL_SDK_JOINTS:
                    position = self._measured[spec.index]
                    self._initial[spec.index] = position
                    self._target[spec.index] = position
                    self._command[spec.index] = position
                    motor_cmd = self._cmd.motor_cmd[spec.index]
                    motor_cmd.tau = 0.0
                    motor_cmd.q = position
                    motor_cmd.dq = 0.0
                    motor_cmd.kp = spec.kp
                    motor_cmd.kd = spec.kd
                self._state_ready.set()

    def start(self, timeout: float = 5.0) -> bool:
        if not self._state_ready.wait(timeout):
            return False
        self._thread = threading.Thread(
            target=self._control_loop, name="r1-arm-control", daemon=True
        )
        self._active = True
        self._thread.start()
        return True

    def adjust(self, spec: JointSpec, delta: float) -> None:
        with self._lock:
            initial = self._initial[spec.index]
            self._target[spec.index] = clamp(
                self._target[spec.index] + delta,
                initial - self.travel,
                initial + self.travel,
            )

    def restore_startup_pose(self) -> None:
        with self._lock:
            for spec in ARM_JOINTS:
                self._target[spec.index] = self._initial[spec.index]

    def status(self, spec: JointSpec) -> Tuple[float, float, float]:
        with self._lock:
            return (
                self._target[spec.index],
                self._measured[spec.index],
                self._initial[spec.index],
            )

    def _prepare_command(self, weight: float) -> None:
        for spec in ALL_SDK_JOINTS:
            self._cmd.motor_cmd[spec.index].q = self._command[spec.index]
        # R1 ArmSdk stores its 0..1 blend weight as 0..100 in mode_pr.
        self._cmd.mode_pr = round(clamp(weight, 0.0, 1.0) * 100.0)
        self._cmd.crc = self._crc.Crc(self._cmd)

    def _control_loop(self) -> None:
        next_tick = time.monotonic()
        maximum_delta = self.maximum_speed * CONTROL_PERIOD
        weight_step = CONTROL_PERIOD / WEIGHT_RAMP_DURATION

        while not self._stop.is_set():
            now = time.monotonic()
            with self._lock:
                if now - self._last_state_time > STATE_TIMEOUT:
                    self._fault = "rt/lowstate timed out"
                    self._stop.set()
                    break
                self._weight = min(1.0, self._weight + weight_step)
                for spec in ARM_JOINTS:
                    self._command[spec.index] = move_toward(
                        self._command[spec.index],
                        self._target[spec.index],
                        maximum_delta,
                    )
                self._prepare_command(self._weight)
            self._publisher.Write(self._cmd)

            next_tick += CONTROL_PERIOD
            self._stop.wait(max(0.0, next_tick - time.monotonic()))

    def release_and_close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

        if self._active:
            steps = max(1, round(WEIGHT_RAMP_DURATION / CONTROL_PERIOD))
            with self._lock:
                starting_weight = self._weight
            for index in range(steps):
                weight = starting_weight * (1.0 - (index + 1) / steps)
                with self._lock:
                    self._weight = weight
                    self._prepare_command(weight)
                self._publisher.Write(self._cmd)
                time.sleep(CONTROL_PERIOD)
            self._active = False

        self._subscriber.Close()
        self._publisher.Close()


def read_key(fd: int) -> str:
    first = os.read(fd, 1)
    if first != b"\x1b":
        return first.decode(errors="ignore").lower()

    readable, _, _ = select.select([fd], [], [], 0.04)
    if not readable or os.read(fd, 1) != b"[":
        return "escape"
    readable, _, _ = select.select([fd], [], [], 0.04)
    if not readable:
        return "escape"
    return {b"A": "up", b"B": "down", b"C": "right", b"D": "left"}.get(
        os.read(fd, 1), "unknown"
    )


def print_controls() -> None:
    print("\nSelect joint:")
    for spec in ARM_JOINTS:
        print(f"  {spec.key}  {spec.name}")
    print("\nCommands:")
    print("  Up/Down     increase/decrease selected joint target")
    print("  Left/Right  select previous/next joint")
    print("  C           return every arm joint to its startup angle")
    print("  Q or ESC    release arm_sdk and quit")


def print_status(spec: JointSpec, values: Tuple[float, float, float]) -> None:
    target, measured, initial = values
    print(
        f"\r[{spec.key}] {spec.name:<22} "
        f"target={math.degrees(target):+7.1f} deg  "
        f"measured={math.degrees(measured):+7.1f} deg  "
        f"delta={math.degrees(target - initial):+6.1f} deg       ",
        end="",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    step = math.radians(args.step_deg)

    print("WARNING: This program takes control of both R1 arms through rt/arm_sdk.")
    print("Test with the robot standing still in locomotion mode and clear all obstacles.")
    confirmation = input("Type ARMS to continue: ").strip()
    if confirmation != "ARMS":
        print("Cancelled; no arm command was sent.")
        return 1

    ChannelFactoryInitialize(0, args.network_interface)
    loco_client = LocoClient()
    loco_client.SetTimeout(3.0)
    loco_client.Init()
    code, fsm_id = get_fsm_id(loco_client)
    if code != 0:
        print(f"Cannot verify locomotion mode: FSM query failed with code {code}.")
        return 2
    if fsm_id != LOCOMOTION_FSM_ID:
        mode = "unknown" if fsm_id is None else str(fsm_id)
        print(f"Robot FSM is {mode}, not locomotion ({LOCOMOTION_FSM_ID}).")
        return 2

    controller = ArmController(
        travel=math.radians(args.travel_deg),
        maximum_speed=math.radians(args.max_speed_deg_s),
    )
    controller.initialize()
    if not controller.start():
        controller.release_and_close()
        print("Timed out waiting for rt/lowstate; no arm control started.")
        return 3

    should_stop = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal should_stop
        should_stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    print_controls()
    selected = 0
    selected_by_key = {spec.key: index for index, spec in enumerate(ARM_JOINTS)}
    print_status(ARM_JOINTS[selected], controller.status(ARM_JOINTS[selected]))

    fd = sys.stdin.fileno()
    old_terminal_settings = termios.tcgetattr(fd)
    exit_code = 0
    next_status_time = time.monotonic()
    try:
        tty.setcbreak(fd)
        while not should_stop:
            if controller.fault is not None:
                print(f"\nSafety stop: {controller.fault}.")
                exit_code = 4
                break

            readable, _, _ = select.select([fd], [], [], 0.1)
            if not readable:
                if time.monotonic() >= next_status_time:
                    spec = ARM_JOINTS[selected]
                    print_status(spec, controller.status(spec))
                    next_status_time = time.monotonic() + 0.25
                continue

            key = read_key(fd)
            if key in {"q", "escape"}:
                break
            if key in selected_by_key:
                selected = selected_by_key[key]
            elif key == "left":
                selected = (selected - 1) % len(ARM_JOINTS)
            elif key == "right":
                selected = (selected + 1) % len(ARM_JOINTS)
            elif key == "up":
                controller.adjust(ARM_JOINTS[selected], step)
            elif key == "down":
                controller.adjust(ARM_JOINTS[selected], -step)
            elif key == "c":
                controller.restore_startup_pose()
            else:
                continue

            spec = ARM_JOINTS[selected]
            print_status(spec, controller.status(spec))
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_terminal_settings)
        print("\nReleasing rt/arm_sdk arm control...")
        controller.release_and_close()

    print("Arm control released; locomotion FSM was not changed.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
