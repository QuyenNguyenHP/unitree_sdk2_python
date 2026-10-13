#!/usr/bin/env python3
"""Walk forward in five measured segments, turning right around obstacles."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.r1.loco.r1_loco_api import ROBOT_API_ID_LOCO_GET_FSM_ID
from unitree_sdk2py.r1.loco.r1_loco_client import LocoClient

# Reuse the stereo pipeline maintained with the YOLO-World distance tool.
SCRIPT_DIR = Path(__file__).resolve().parent
STEREO_DIR = SCRIPT_DIR / "r1_yolo_world_stereo"
sys.path.insert(0, str(STEREO_DIR))

from r1_stereo_common import (
    StereoGeometry,
    StereoRtpCapture,
    compute_disparity,
    create_sgbm,
)


LOCOMOTION_FSM_ID = 811


# ---------------------------------------------------------------------------
# USER SETTINGS
# Edit these values to change the default behavior. Command-line options can
# still override most of them without modifying this file.
# ---------------------------------------------------------------------------
CALIBRATION_FILE = STEREO_DIR / "r1_stereo_calibration.npz"

FORWARD_STEPS = 5             # Number of successful forward movements.
CLEARANCE_M = 1.0             # Turn right when an obstacle is closer than this.
FORWARD_SPEED_M_S = 0.25      # Forward velocity for one movement.
TURN_SPEED_RAD_S = 0.5        # Right-turn angular velocity (magnitude).
COMMAND_DURATION_S = 1.0      # R1 firmware works reliably with one-second commands.
MAX_RIGHT_TURNS = 12          # Stop instead of turning forever.
SETTLE_SECONDS = 0.6          # Pause before measuring after each movement.
DEPTH_SAMPLES = 3             # All samples must provide a reliable distance.

LEFT_CAMERA_PORT = 5002
RIGHT_CAMERA_PORT = 5003
CAPTURE_BACKEND = "auto"      # "auto", "opencv", or "gstreamer-cli".
NUM_DISPARITIES = 128         # Must be a positive multiple of 16.
BLOCK_SIZE = 5                # Must be an odd integer >= 3.
MAX_PAIR_DELTA_MS = 80.0

# Central image region used as the forward travel corridor. Values are image
# fractions in the range 0.0 to 1.0.
CORRIDOR_LEFT = 0.30
CORRIDOR_RIGHT = 0.70
CORRIDOR_TOP = 0.25
CORRIDOR_BOTTOM = 0.70
MIN_DEPTH_M = 0.15
MAX_DEPTH_M = 10.0
MIN_VALID_DEPTH_PIXELS = 200
OBSTACLE_PERCENTILE = 10.0    # Low percentile rejects isolated depth noise.

START_LOCOMOTION_BY_DEFAULT = False
SKIP_WALK_CONFIRMATION_BY_DEFAULT = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("network_interface", help="interface connected to the R1")
    parser.add_argument(
        "--calibration",
        default=str(CALIBRATION_FILE),
    )
    parser.add_argument("--steps", type=int, default=FORWARD_STEPS)
    parser.add_argument("--clearance-m", type=float, default=CLEARANCE_M)
    parser.add_argument("--forward-speed", type=float, default=FORWARD_SPEED_M_S)
    parser.add_argument("--turn-speed", type=float, default=TURN_SPEED_RAD_S)
    parser.add_argument("--max-turns", type=int, default=MAX_RIGHT_TURNS)
    parser.add_argument("--settle-seconds", type=float, default=SETTLE_SECONDS)
    parser.add_argument("--samples", type=int, default=DEPTH_SAMPLES)
    parser.add_argument("--left-port", type=int, default=LEFT_CAMERA_PORT)
    parser.add_argument("--right-port", type=int, default=RIGHT_CAMERA_PORT)
    parser.add_argument(
        "--capture-backend",
        choices=("auto", "opencv", "gstreamer-cli"),
        default=CAPTURE_BACKEND,
    )
    parser.add_argument("--num-disparities", type=int, default=NUM_DISPARITIES)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--max-pair-delta-ms", type=float, default=MAX_PAIR_DELTA_MS)
    parser.add_argument(
        "--start-locomotion",
        action="store_true",
        default=START_LOCOMOTION_BY_DEFAULT,
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        default=SKIP_WALK_CONFIRMATION_BY_DEFAULT,
        help="skip the WALK confirmation",
    )
    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be at least 1")
    if not 0.3 <= args.clearance_m <= 5.0:
        parser.error("--clearance-m must be between 0.3 and 5.0")
    if not 0.0 < args.forward_speed <= 0.5:
        parser.error("--forward-speed must be greater than 0 and at most 0.5")
    if not 0.0 < args.turn_speed <= 1.0:
        parser.error("--turn-speed must be greater than 0 and at most 1.0")
    if args.max_turns < 1 or args.samples < 1:
        parser.error("--max-turns and --samples must be at least 1")
    if args.settle_seconds < 0.0:
        parser.error("--settle-seconds cannot be negative")
    if COMMAND_DURATION_S <= 0.0:
        parser.error("COMMAND_DURATION_S must be greater than zero")
    if not 0.0 <= CORRIDOR_LEFT < CORRIDOR_RIGHT <= 1.0:
        parser.error("CORRIDOR_LEFT/RIGHT must define a valid image region")
    if not 0.0 <= CORRIDOR_TOP < CORRIDOR_BOTTOM <= 1.0:
        parser.error("CORRIDOR_TOP/BOTTOM must define a valid image region")
    if not 0.0 < MIN_DEPTH_M < MAX_DEPTH_M:
        parser.error("MIN_DEPTH_M must be smaller than MAX_DEPTH_M")
    if MIN_VALID_DEPTH_PIXELS < 1:
        parser.error("MIN_VALID_DEPTH_PIXELS must be at least 1")
    if not 0.0 <= OBSTACLE_PERCENTILE <= 100.0:
        parser.error("OBSTACLE_PERCENTILE must be between 0 and 100")
    return args


def get_fsm_id(client: LocoClient) -> tuple[int, Optional[int]]:
    # The SDK registers this API but does not provide a public getter wrapper.
    code, data = client._Call(ROBOT_API_ID_LOCO_GET_FSM_ID, "{}")
    if code != 0:
        return code, None
    try:
        return code, int(json.loads(data)["data"])
    except (TypeError, ValueError, KeyError, json.JSONDecodeError):
        return code, None


def corridor_distance(points: np.ndarray, disparity: np.ndarray) -> Optional[float]:
    """Return a conservative distance estimate in the central travel corridor."""
    height, width = disparity.shape
    x1, x2 = int(width * CORRIDOR_LEFT), int(width * CORRIDOR_RIGHT)
    y1, y2 = int(height * CORRIDOR_TOP), int(height * CORRIDOR_BOTTOM)
    z = points[y1:y2, x1:x2, 2]
    d = disparity[y1:y2, x1:x2]
    valid = (
        np.isfinite(z)
        & (d > 0.5)
        & (z >= MIN_DEPTH_M)
        & (z <= MAX_DEPTH_M)
    )
    values = z[valid]
    if values.size < MIN_VALID_DEPTH_PIXELS:
        return None

    # A low percentile notices foreground obstacles without trusting isolated
    # minimum-depth speckles.
    return float(np.percentile(values, OBSTACLE_PERCENTILE))


def measure_clearance(
    capture: StereoRtpCapture,
    geometry: StereoGeometry,
    matcher,
    sample_count: int,
    max_pair_delta_ms: float,
) -> Optional[float]:
    measurements = []
    for _ in range(sample_count):
        left, right, _ = capture.read_pair(max_delta_ms=max_pair_delta_ms)
        left_rectified, right_rectified = geometry.rectify(left, right)
        disparity = compute_disparity(matcher, left_rectified, right_rectified)
        points = geometry.points_from_disparity(disparity)
        distance = corridor_distance(points, disparity)
        if distance is not None:
            measurements.append(distance)
    if len(measurements) != sample_count:
        return None
    return min(measurements)


def send_motion(client: LocoClient, vx: float, omega: float, label: str) -> None:
    code = client.SetVelocity(vx, 0.0, omega, COMMAND_DURATION_S)
    if code != 0:
        raise RuntimeError(f"{label} command failed with code {code}")
    time.sleep(COMMAND_DURATION_S)
    stop_code = client.SetVelocity(0.0, 0.0, 0.0, COMMAND_DURATION_S)
    if stop_code != 0:
        raise RuntimeError(f"stop after {label} failed with code {stop_code}")


def main() -> int:
    args = parse_args()
    print("WARNING: This program autonomously moves the real robot.")
    print("Use a flat area and keep the remote emergency stop ready.")
    print("Stereo depth is perception-only and is not a safety-rated sensor.")
    if not args.yes and input("Type WALK to continue: ").strip() != "WALK":
        print("Cancelled; no motion command was sent.")
        return 1

    geometry = StereoGeometry(args.calibration)
    matcher = create_sgbm(args.num_disparities, args.block_size)
    width, height = geometry.image_size
    capture = StereoRtpCapture(
        args.left_port,
        args.right_port,
        width=width,
        height=height,
        backend=args.capture_backend,
    )

    ChannelFactoryInitialize(0, args.network_interface)
    client = LocoClient()
    client.SetTimeout(3.0)
    client.Init()

    code, fsm_id = get_fsm_id(client)
    if code != 0 or fsm_id is None:
        print(f"Cannot verify robot FSM (RPC code {code}); no motion was sent.")
        return 2
    if fsm_id != LOCOMOTION_FSM_ID:
        if not args.start_locomotion:
            print(f"Robot FSM is {fsm_id}, not locomotion (811).")
            print("Enter locomotion first or add --start-locomotion.")
            return 3
        if client.SetFsmId(LOCOMOTION_FSM_ID) != 0:
            print("Could not enter locomotion mode; no walking command was sent.")
            return 4
        time.sleep(3.0)
        code, fsm_id = get_fsm_id(client)
        if code != 0 or fsm_id != LOCOMOTION_FSM_ID:
            print("Locomotion mode could not be verified; no walking command was sent.")
            return 5

    stop_requested = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    completed = 0
    turns = 0
    try:
        capture.start()
        print(f"Stereo baseline: {geometry.baseline_m:.4f} m")
        while completed < args.steps and not stop_requested:
            time.sleep(args.settle_seconds)
            try:
                distance = measure_clearance(
                    capture,
                    geometry,
                    matcher,
                    args.samples,
                    args.max_pair_delta_ms,
                )
            except TimeoutError as exc:
                print(f"Depth unavailable: {exc}. Stopping.")
                return 6

            if distance is None:
                print("Depth is not reliable in the forward corridor. Stopping.")
                return 7

            print(f"Check before step {completed + 1}: nearest obstacle {distance:.2f} m")
            if distance < args.clearance_m:
                if turns >= args.max_turns:
                    print(f"Still blocked after {turns} right turns. Stopping.")
                    return 8
                turns += 1
                print(f"Blocked (< {args.clearance_m:.2f} m); turning right ({turns}/{args.max_turns}).")
                send_motion(client, 0.0, -args.turn_speed, "right turn")
                continue

            print(f"Path clear; moving forward step {completed + 1}/{args.steps}.")
            send_motion(client, args.forward_speed, 0.0, "forward")
            completed += 1

        if stop_requested:
            print("Stop requested; route aborted.")
            return 130
        print(f"Completed {completed} forward steps.")
        return 0
    finally:
        try:
            client.SetVelocity(0.0, 0.0, 0.0, COMMAND_DURATION_S)
        except Exception as exc:
            print(f"Warning: final stop command failed: {exc}")
        capture.stop()


if __name__ == "__main__":
    raise SystemExit(main())
