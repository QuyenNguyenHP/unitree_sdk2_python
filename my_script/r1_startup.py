#!/usr/bin/env python3
"""Interactively switch an R1 among zero torque, stance, and locomotion."""

from __future__ import annotations

import argparse
import json
import select
import sys
import termios
import time
import tty
from typing import Optional

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.r1.loco.r1_loco_api import ROBOT_API_ID_LOCO_GET_FSM_ID
from unitree_sdk2py.r1.loco.r1_loco_client import LocoClient


ZERO_TORQUE_FSM = 0
STANCE_FSM = 4
LOCOMOTION_FSM = 811
FSM_NAMES = {
    ZERO_TORQUE_FSM: "zero torque",
    1: "damping",
    STANCE_FSM: "stance",
    701: "lie to stand",
    702: "stand to lie",
    LOCOMOTION_FSM: "locomotion",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "network_interface",
        help="network interface connected to the robot, for example eth10",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="seconds allowed for each verified mode transition (default: 5)",
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    if not sys.stdin.isatty():
        parser.error("this script requires an interactive terminal")
    return args


def get_fsm_id(client: LocoClient) -> tuple[int, Optional[int]]:
    """Query API 7001, which the SDK registers without a public wrapper."""
    code, data = client._Call(ROBOT_API_ID_LOCO_GET_FSM_ID, "{}")
    if code != 0:
        return code, None
    try:
        return code, int(json.loads(data)["data"])
    except (TypeError, ValueError, KeyError, json.JSONDecodeError):
        return code, None


def wait_for_fsm(client: LocoClient, expected: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code, fsm_id = get_fsm_id(client)
        if code == 0 and fsm_id == expected:
            return True
        time.sleep(0.2)
    return False


def read_key() -> str:
    while True:
        readable, _, _ = select.select([sys.stdin], [], [], 0.2)
        if readable:
            return sys.stdin.read(1)


def request_and_verify(
    client: LocoClient, target: int, label: str, timeout: float
) -> bool:
    code = client.SetFsmId(target)
    if code != 0:
        print(f"Failed to request {label}: RPC code {code}.")
        return False
    if not wait_for_fsm(client, target, timeout):
        current_code, current = get_fsm_id(client)
        current_text = (
            f"{FSM_NAMES.get(current, 'unknown')} ({current})"
            if current_code == 0 and current is not None
            else f"unavailable, RPC code {current_code}"
        )
        print(f"Timed out waiting for {label}; current FSM is {current_text}.")
        return False
    print(f"Robot is now in {label} mode (FSM {target}).")
    return True


def main() -> int:
    args = parse_args()

    print("WARNING: Changing modes can make the real robot move or support its weight.")
    print("Place the robot safely, clear the area, and keep emergency damping ready.")

    ChannelFactoryInitialize(0, args.network_interface)
    client = LocoClient()
    client.SetTimeout(3.0)
    client.Init()

    code, fsm_id = get_fsm_id(client)
    if code != 0 or fsm_id is None:
        print(f"Cannot verify the current FSM (RPC code {code}); no command was sent.")
        return 1
    print(f"Current FSM: {FSM_NAMES.get(fsm_id, 'unknown')} ({fsm_id})")
    if fsm_id not in {ZERO_TORQUE_FSM, STANCE_FSM, LOCOMOTION_FSM}:
        print("This controller only supports FSM 0, 4, and 811; no command was sent.")
        return 2

    original_terminal = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while True:
            code, fsm_id = get_fsm_id(client)
            if code != 0 or fsm_id is None:
                print(f"\nCannot refresh current FSM (RPC code {code}); exiting.")
                return 4

            if fsm_id == ZERO_TORQUE_FSM:
                prompt = "ZERO TORQUE (0): Space -> STANCE | Q/Esc -> exit: "
            elif fsm_id == STANCE_FSM:
                prompt = "STANCE (4): Space -> LOCOMOTION | 0 -> ZERO TORQUE | Q/Esc -> exit: "
            elif fsm_id == LOCOMOTION_FSM:
                prompt = "LOCOMOTION (811): Space -> STANCE | Q/Esc -> exit: "
            else:
                print(f"\nFSM changed externally to unsupported mode {fsm_id}; exiting.")
                return 5

            print(prompt, end="", flush=True)
            key = read_key()
            print()
            if key.lower() == "q" or key == "\x1b":
                break

            target: Optional[int] = None
            if key == " ":
                if fsm_id == ZERO_TORQUE_FSM:
                    target = STANCE_FSM
                elif fsm_id == STANCE_FSM:
                    target = LOCOMOTION_FSM
                elif fsm_id == LOCOMOTION_FSM:
                    target = STANCE_FSM
            elif key == "0" and fsm_id == STANCE_FSM:
                target = ZERO_TORQUE_FSM

            if target is None:
                print("Key ignored in the current mode.")
                continue

            label = FSM_NAMES[target]
            if not request_and_verify(client, target, label, args.timeout):
                print("Mode transition failed; exiting without another command.")
                return 6
    except KeyboardInterrupt:
        print("\nStartup cancelled with Ctrl+C.")
        return 130
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original_terminal)

    print("Mode controller exited. The robot remains in its current mode.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
