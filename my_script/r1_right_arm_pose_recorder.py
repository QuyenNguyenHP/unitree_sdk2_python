#!/usr/bin/env python3
"""Record three R1 right-arm poses and replay them through rt/arm_sdk.

Created: 2026-08-24
Mode: Locomotion (FSM 811)
Topics: reads rt/lowstate; writes rt/arm_sdk only after PLAY confirmation
Purpose: interactively record three five-joint right-arm poses and replay them
with synchronized, speed-limited movement.
"""

import argparse
import json
import math
import signal
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.r1.loco.r1_loco_api import ROBOT_API_ID_LOCO_GET_FSM_ID
from unitree_sdk2py.r1.loco.r1_loco_client import LocoClient
from unitree_sdk2py.utils.crc import CRC

LOCOMOTION_FSM_ID = 811
POSE_COUNT = 3
CONTROL_PERIOD = 0.01
STATE_TIMEOUT = 1.0
WEIGHT_RAMP_DURATION = 1.0


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

# arm_sdk's blend weight claims all these joints. Hold the others at their
# measured startup positions so they cannot jump when playback starts.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record three R1 right-arm poses, then repeat them.")
    parser.add_argument("network_interface", help="Robot network interface, e.g. enp2s0")
    parser.add_argument("--cycles", type=int, default=0,
                        help="Playback cycles; 0 repeats until Ctrl+C (default: 0)")
    parser.add_argument("--speed-deg-s", type=float, default=20.0,
                        help="Maximum joint speed during playback (default: 20)")
    parser.add_argument("--hold", type=float, default=1.0,
                        help="Seconds to hold each pose (default: 1.0)")
    args = parser.parse_args()
    if args.cycles < 0:
        parser.error("--cycles must be zero or greater")
    if args.speed_deg_s <= 0.0:
        parser.error("--speed-deg-s must be greater than zero")
    if args.hold < 0.0:
        parser.error("--hold must be zero or greater")
    return args


class StateReader:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._positions: Pose = {}
        self._sample_number = 0
        self._last_update = 0.0
        self._subscriber = ChannelSubscriber("rt/lowstate", LowState_)

    def start(self) -> None:
        self._subscriber.Init(self._handle, 10)

    def _handle(self, msg: LowState_) -> None:
        with self._condition:
            self._positions = {j.index: float(msg.motor_state[j.index].q) for j in SDK_JOINTS}
            self._sample_number += 1
            self._last_update = time.monotonic()
            self._condition.notify_all()

    def snapshot(self, timeout: float = 5.0) -> Pose:
        with self._condition:
            previous = self._sample_number
            if not self._condition.wait_for(lambda: self._sample_number > previous, timeout=timeout):
                raise TimeoutError("No new rt/lowstate sample received")
            return dict(self._positions)

    def latest(self) -> Tuple[Pose, float]:
        with self._condition:
            return dict(self._positions), self._last_update

    def close(self) -> None:
        self._subscriber.Close()


def print_pose(number: int, pose: Pose) -> None:
    print(f"\nPose {number}:")
    print("Idx  Joint                       q (rad)       q (deg)")
    print("---  --------------------------  ------------  ------------")
    for joint in RIGHT_ARM:
        value = pose[joint.index]
        print(f"{joint.index:>3}  {joint.name:<26}  {value:>+12.6f}  {math.degrees(value):>+12.2f}")
    values = ", ".join(f"{pose[j.index]:.6f}" for j in RIGHT_ARM)
    print(f"Python: [{values}]")


def get_fsm_id(client: LocoClient) -> Tuple[int, Optional[int]]:
    code, data = client._Call(ROBOT_API_ID_LOCO_GET_FSM_ID, "{}")
    if code != 0:
        return code, None
    try:
        return code, int(json.loads(data)["data"])
    except (TypeError, ValueError, KeyError, json.JSONDecodeError):
        return code, None


class PosePlayer:
    def __init__(self, reader: StateReader, speed: float) -> None:
        self.reader = reader
        self.speed = speed
        self.stop_event = threading.Event()
        self.crc = CRC()
        self.command = unitree_hg_msg_dds__LowCmd_()
        self.publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.current: Pose = {}
        self.weight = 0.0

    def initialize(self) -> None:
        positions, updated = self.reader.latest()
        if not positions or time.monotonic() - updated > STATE_TIMEOUT:
            raise RuntimeError("rt/lowstate is not current")
        self.current = positions
        for joint in SDK_JOINTS:
            motor = self.command.motor_cmd[joint.index]
            motor.mode = 1
            motor.tau = 0.0
            motor.q = positions[joint.index]
            motor.dq = 0.0
            motor.kp = joint.kp
            motor.kd = joint.kd
        self.publisher.Init()

    def _write(self) -> None:
        for joint in SDK_JOINTS:
            self.command.motor_cmd[joint.index].q = self.current[joint.index]
        self.command.mode_pr = round(self.weight * 100.0)
        self.command.crc = self.crc.Crc(self.command)
        self.publisher.Write(self.command)

    def ramp_weight(self, target: float) -> bool:
        steps = max(1, round(WEIGHT_RAMP_DURATION / CONTROL_PERIOD))
        start = self.weight
        for step in range(steps):
            if self.stop_event.is_set() and target > 0.0:
                return False
            self.weight = start + (target - start) * (step + 1) / steps
            self._write()
            time.sleep(CONTROL_PERIOD)
        return True

    def move_to(self, pose: Pose) -> bool:
        maximum_delta = self.speed * CONTROL_PERIOD
        while not self.stop_event.is_set():
            _, updated = self.reader.latest()
            if time.monotonic() - updated > STATE_TIMEOUT:
                raise RuntimeError("rt/lowstate timed out during playback")
            complete = True
            for joint in RIGHT_ARM:
                difference = pose[joint.index] - self.current[joint.index]
                if abs(difference) > maximum_delta:
                    difference = math.copysign(maximum_delta, difference)
                    complete = False
                self.current[joint.index] += difference
            self._write()
            if complete:
                return True
            time.sleep(CONTROL_PERIOD)
        return False

    def close(self) -> None:
        self.ramp_weight(0.0)
        self.publisher.Close()


def main() -> int:
    args = parse_args()
    ChannelFactoryInitialize(0, args.network_interface)
    reader = StateReader()
    reader.start()
    try:
        reader.snapshot()
        print("Record exactly three poses of the five right-arm joints.")
        print("Move the arm safely, then press Enter to capture each pose.")
        poses: List[Pose] = []
        for number in range(1, POSE_COUNT + 1):
            input(f"\nPosition the right arm for pose {number}, then press Enter: ")
            pose = reader.snapshot()
            poses.append(pose)
            print_pose(number, pose)

        print("\nAll three poses recorded.")
        print("Playback requires locomotion mode (FSM 811) and takes arm_sdk control.")
        if input("Type PLAY to start repeating the poses: ").strip() != "PLAY":
            print("Playback cancelled; no command was sent.")
            return 0

        client = LocoClient()
        client.SetTimeout(3.0)
        client.Init()
        code, fsm_id = get_fsm_id(client)
        if code != 0 or fsm_id != LOCOMOTION_FSM_ID:
            print(f"Cannot play: expected FSM {LOCOMOTION_FSM_ID}, received code={code}, fsm={fsm_id}.")
            return 2

        player = PosePlayer(reader, math.radians(args.speed_deg_s))
        player.initialize()

        def request_stop(_signum: int, _frame: object) -> None:
            player.stop_event.set()

        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        completed_cycles = 0
        exit_code = 0
        try:
            player.ramp_weight(1.0)
            while not player.stop_event.is_set() and (args.cycles == 0 or completed_cycles < args.cycles):
                for number, pose in enumerate(poses, 1):
                    if not player.move_to(pose):
                        break
                    print(f"Reached pose {number}")
                    if player.stop_event.wait(args.hold):
                        break
                else:
                    completed_cycles += 1
                    print(f"Completed cycle {completed_cycles}")
                    continue
                break
        except RuntimeError as exc:
            print(f"Safety stop: {exc}")
            exit_code = 3
        finally:
            print("Releasing rt/arm_sdk control...")
            player.close()
        return exit_code
    except (TimeoutError, RuntimeError) as exc:
        print(f"Cannot continue: {exc}")
        return 1
    finally:
        reader.close()


if __name__ == "__main__":
    raise SystemExit(main())
