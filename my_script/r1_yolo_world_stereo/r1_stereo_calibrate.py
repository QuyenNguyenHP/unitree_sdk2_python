#!/usr/bin/env python3
"""Calibrate the R1 left/right RTP cameras using a chessboard."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from r1_stereo_common import StereoRtpCapture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-port", type=int, default=5002)
    parser.add_argument("--right-port", type=int, default=5003)
    parser.add_argument("--width", type=int, default=544)
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument(
        "--capture-backend",
        choices=("auto", "opencv", "gstreamer-cli"),
        default="auto",
    )
    parser.add_argument("--columns", type=int, default=9, help="inner corners across")
    parser.add_argument("--rows", type=int, default=6, help="inner corners down")
    parser.add_argument(
        "--square-size-m",
        type=float,
        required=True,
        help="physical chessboard square size in metres",
    )
    parser.add_argument("--minimum-samples", type=int, default=20)
    parser.add_argument("--max-pair-delta-ms", type=float, default=80.0)
    parser.add_argument("--output", default="r1_stereo_calibration.npz")
    return parser.parse_args()


def find_corners(gray: np.ndarray, board_size: tuple[int, int]):
    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(
            gray,
            board_size,
            flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
        )
        return found, corners

    found, corners = cv2.findChessboardCorners(
        gray,
        board_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if found:
        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
    return found, corners


def solve_calibration(
    object_points,
    left_points,
    right_points,
    image_size: tuple[int, int],
    output: Path,
) -> None:
    left_rms, K1, D1, _, _ = cv2.calibrateCamera(
        object_points, left_points, image_size, None, None
    )
    right_rms, K2, D2, _, _ = cv2.calibrateCamera(
        object_points, right_points, image_size, None, None
    )
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-6,
    )
    stereo_rms, K1, D1, K2, D2, R, T, _, _ = cv2.stereoCalibrate(
        object_points,
        left_points,
        right_points,
        K1,
        D1,
        K2,
        D2,
        image_size,
        criteria=criteria,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )
    baseline_m = float(np.linalg.norm(T))
    if not 0.01 <= baseline_m <= 1.0:
        raise RuntimeError(
            f"Implausible baseline {baseline_m:.6f} m. Check --square-size-m."
        )

    np.savez(
        output,
        K1=K1,
        D1=D1,
        K2=K2,
        D2=D2,
        R=R,
        T=T,
        image_size=np.asarray(image_size, dtype=np.int32),
    )
    print(f"Saved calibration: {output}")
    print(f"Left RMS: {left_rms:.4f} px")
    print(f"Right RMS: {right_rms:.4f} px")
    print(f"Stereo RMS: {stereo_rms:.4f} px")
    print(f"Baseline: {baseline_m:.6f} m")
    if stereo_rms > 1.0:
        print("WARNING: stereo RMS is high; collect better and more varied samples.")


def main() -> int:
    args = parse_args()
    if args.square_size_m <= 0:
        raise SystemExit("--square-size-m must be positive")

    board_size = (args.columns, args.rows)
    object_template = np.zeros((args.columns * args.rows, 3), np.float32)
    object_template[:, :2] = np.mgrid[
        0 : args.columns, 0 : args.rows
    ].T.reshape(-1, 2)
    object_template *= args.square_size_m

    object_points = []
    left_points = []
    right_points = []
    image_size = None

    capture = StereoRtpCapture(
        args.left_port,
        args.right_port,
        width=args.width,
        height=args.height,
        backend=args.capture_backend,
    )
    capture.start()
    print("SPACE: capture pair | C: calibrate | Q/ESC: quit")
    print("Move/tilt the chessboard and cover the centre, edges and corners.")

    try:
        while True:
            try:
                left, right, delta_ms = capture.read_pair(
                    max_delta_ms=args.max_pair_delta_ms
                )
            except TimeoutError as exc:
                print(exc)
                continue

            if left.shape[:2] != right.shape[:2]:
                raise RuntimeError("Left and right frame sizes differ")
            image_size = (left.shape[1], left.shape[0])
            left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
            left_found, left_corners = find_corners(left_gray, board_size)
            right_found, right_corners = find_corners(right_gray, board_size)

            preview_left = left.copy()
            preview_right = right.copy()
            if left_found:
                cv2.drawChessboardCorners(
                    preview_left, board_size, left_corners, left_found
                )
            if right_found:
                cv2.drawChessboardCorners(
                    preview_right, board_size, right_corners, right_found
                )
            status = (
                f"samples={len(object_points)}/{args.minimum_samples} "
                f"pair_delta={delta_ms:.1f}ms "
                f"corners={'OK' if left_found and right_found else 'NOT FOUND'}"
            )
            cv2.putText(
                preview_left,
                status,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0) if left_found and right_found else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(
                "R1 stereo calibration - left | right",
                np.hstack((preview_left, preview_right)),
            )

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                if left_found and right_found:
                    object_points.append(object_template.copy())
                    left_points.append(left_corners.copy())
                    right_points.append(right_corners.copy())
                    print(f"Captured pair {len(object_points)}")
                else:
                    print("Chessboard must be visible in both images")
            elif key == ord("c"):
                if len(object_points) < args.minimum_samples:
                    print(
                        f"Need at least {args.minimum_samples} samples; "
                        f"currently have {len(object_points)}"
                    )
                    continue
                assert image_size is not None
                solve_calibration(
                    object_points,
                    left_points,
                    right_points,
                    image_size,
                    Path(args.output),
                )
                return 0
            elif key in (ord("q"), 27):
                return 0
    finally:
        capture.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
