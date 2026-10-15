#!/usr/bin/env python3
"""Continuously print the R1 stereo center-clearance distance."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
STEREO_DIR = SCRIPT_DIR / "r1_yolo_world_stereo"
if str(STEREO_DIR) not in sys.path:
    sys.path.insert(0, str(STEREO_DIR))

from r1_stereo_common import (  # noqa: E402
    StereoGeometry,
    StereoRtpCapture,
    compute_disparity,
    create_sgbm,
)


CALIBRATION_FILE = STEREO_DIR / "r1_stereo_calibration.npz"

# Center-clearance box in pixels for the default 544 x 448 stereo image.
CENTER_CLEARANCE_X = 300
CENTER_CLEARANCE_Y = 180
CENTER_CLEARANCE_WIDTH = 220
CENTER_CLEARANCE_HEIGHT = 260

LEFT_CAMERA_PORT = 5002
RIGHT_CAMERA_PORT = 5003
CAPTURE_BACKEND = "auto"
NUM_DISPARITIES = 64
BLOCK_SIZE = 3
MAX_PAIR_DELTA_MS = 80.0
MIN_DISTANCE_M = 0.15
MAX_DISTANCE_M = 15.0
CELL_SIZE_PX = 16
MIN_VALID_RATIO = 0.40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print center-clearance distance from the R1 stereo cameras."
    )
    parser.add_argument("--calibration", default=str(CALIBRATION_FILE))
    parser.add_argument("--interval", type=float, default=0.25)
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
    args = parser.parse_args()

    if args.interval <= 0:
        parser.error("--interval must be greater than zero")
    if args.max_pair_delta_ms <= 0:
        parser.error("--max-pair-delta-ms must be greater than zero")
    return args


def center_clearance_distance(
    points_3d: np.ndarray, disparity: np.ndarray
) -> float | None:
    """Return the nearest reliable local surface inside the center box."""
    height, width = disparity.shape[:2]
    x1 = max(0, min(width - 1, CENTER_CLEARANCE_X))
    y1 = max(0, min(height - 1, CENTER_CLEARANCE_Y))
    x2 = max(x1 + 1, min(width, CENTER_CLEARANCE_X + CENTER_CLEARANCE_WIDTH))
    y2 = max(y1 + 1, min(height, CENTER_CLEARANCE_Y + CENTER_CLEARANCE_HEIGHT))

    distances = points_3d[y1:y2, x1:x2, 2]
    roi_disparity = disparity[y1:y2, x1:x2]
    valid = (
        np.isfinite(distances)
        & (roi_disparity > 0.5)
        & (distances >= MIN_DISTANCE_M)
        & (distances <= MAX_DISTANCE_M)
    )
    samples = distances[valid]
    if samples.size < 100:
        return None

    cell_distances: list[float] = []
    roi_height, roi_width = distances.shape
    for cell_y in range(0, roi_height, CELL_SIZE_PX):
        for cell_x in range(0, roi_width, CELL_SIZE_PX):
            cell_valid = valid[
                cell_y : cell_y + CELL_SIZE_PX,
                cell_x : cell_x + CELL_SIZE_PX,
            ]
            if (
                cell_valid.size == 0
                or np.count_nonzero(cell_valid)
                < cell_valid.size * MIN_VALID_RATIO
            ):
                continue
            cell_values = distances[
                cell_y : cell_y + CELL_SIZE_PX,
                cell_x : cell_x + CELL_SIZE_PX,
            ][cell_valid]
            cell_distances.append(float(np.median(cell_values)))

    if cell_distances:
        return min(cell_distances)
    return float(np.percentile(samples, 10.0))


def main() -> int:
    args = parse_args()
    stop_requested = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    capture: StereoRtpCapture | None = None
    try:
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

        print(
            "Center clearance region: "
            f"x={CENTER_CLEARANCE_X}, y={CENTER_CLEARANCE_Y}, "
            f"width={CENTER_CLEARANCE_WIDTH}, height={CENTER_CLEARANCE_HEIGHT}px"
        )
        print(
            f"Stereo ports: left={args.left_port}, right={args.right_port}. "
            "Press Ctrl+C to stop."
        )
        capture.start()

        last_printed_at = 0.0
        while not stop_requested:
            left, right, pair_delta_ms = capture.read_pair(
                max_delta_ms=args.max_pair_delta_ms
            )
            left_rectified, right_rectified = geometry.rectify(left, right)
            disparity = compute_disparity(
                matcher, left_rectified, right_rectified
            )
            points_3d = geometry.points_from_disparity(disparity)
            distance = center_clearance_distance(points_3d, disparity)

            now = time.monotonic()
            if now - last_printed_at < args.interval:
                continue
            if distance is None:
                print(
                    f"CENTER CLEARANCE: unavailable | pair delta: "
                    f"{pair_delta_ms:.1f} ms",
                    flush=True,
                )
            else:
                print(
                    f"CENTER CLEARANCE: {distance:.3f} m | pair delta: "
                    f"{pair_delta_ms:.1f} ms",
                    flush=True,
                )
            last_printed_at = now
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if capture is not None:
            capture.stop()

    print("Center-clearance monitor stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
