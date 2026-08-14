#!/usr/bin/env python3
"""Capture one read-only snapshot of all Unitree R1 joint positions."""

import argparse
import math
import threading
from typing import Dict

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

# Physical R1 motors and their indices in LowState.motor_state.
R1_MOTORS = {
    0: "Left hip pitch",
    1: "Left hip roll",
    2: "Left hip yaw",
    3: "Left knee",
    4: "Left ankle pitch",
    5: "Left ankle roll",
    6: "Right hip pitch",
    7: "Right hip roll",
    8: "Right hip yaw",
    9: "Right knee",
    10: "Right ankle pitch",
    11: "Right ankle roll",
    12: "Waist roll",
    13: "Waist yaw",
    15: "Left shoulder pitch",
    16: "Left shoulder roll",
    17: "Left shoulder yaw",
    18: "Left elbow",
    19: "Left wrist roll",
    22: "Right shoulder pitch",
    23: "Right shoulder roll",
    24: "Right shoulder yaw",
    25: "Right elbow",
    26: "Right wrist roll",
    29: "Head pitch",
    30: "Head yaw",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture one read-only R1 joint-position snapshot from rt/lowstate "
            "without changing the robot mode."
        )
    )
    parser.add_argument(
        "network_interface",
        help="Network interface connected to the robot, for example eth10",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for a low-state sample (default: 5)",
    )
    args = parser.parse_args()
    if args.timeout <= 0.0:
        parser.error("--timeout must be greater than zero")
    return args


def capture_motor_positions(timeout: float) -> Dict[int, float]:
    """Capture one LowState sample and return motor index to q (radians)."""
    ready = threading.Event()
    positions: Dict[int, float] = {}

    def handle_low_state(msg: LowState_) -> None:
        if ready.is_set():
            return
        for motor_index in R1_MOTORS:
            positions[motor_index] = float(msg.motor_state[motor_index].q)
        ready.set()

    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(handle_low_state, 10)
    try:
        if not ready.wait(timeout):
            raise TimeoutError(
                f"No rt/lowstate sample was received within {timeout:g} seconds"
            )
    finally:
        subscriber.Close()

    return dict(positions)


def print_positions(positions: Dict[int, float]) -> None:
    print("\nR1 joint-position snapshot:")
    print("Idx  Joint                       q (rad)       q (deg)")
    print("---  --------------------------  ------------  ------------")
    for motor_index, name in R1_MOTORS.items():
        q = positions[motor_index]
        print(
            f"{motor_index:>3}  {name:<26}  "
            f"{q:>+12.6f}  {math.degrees(q):>+12.2f}"
        )


def main() -> int:
    args = parse_args()
    ChannelFactoryInitialize(0, args.network_interface)

    print("Waiting for one read-only rt/lowstate sample...")

    try:
        positions = capture_motor_positions(args.timeout)
    except TimeoutError as exc:
        print(str(exc))
        return 2

    print_positions(positions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
