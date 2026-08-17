#!/usr/bin/env python3
"""Detect objects in the R1 left camera image and report their depth.

This example must run on the R1 NX expansion dock.  It consumes the left RGB
RTP/H.264 stream on UDP port 5002 and the 16-bit depth map exposed by Unitree
as /dev/video-dep.  Depth values are interpreted as millimetres, as documented
by Unitree's R1 video service tutorial.
"""

import argparse
import os
import subprocess
import time
from typing import Optional, Tuple

import cv2
import numpy as np


DEFAULT_WIDTH = 544
DEFAULT_HEIGHT = 448
INVALID_DEPTH = np.iinfo(np.uint16).max


def gstreamer_pipeline(port: int) -> str:
    """Return an OpenCV GStreamer pipeline for a Unitree RTP/H.264 stream."""
    return (
        f"udpsrc port={port} "
        'caps="application/x-rtp,media=video,clock-rate=90000,'
        'encoding-name=H264,payload=96" ! '
        "rtpjitterbuffer latency=50 drop-on-latency=true ! "
        "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! "
        "video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false"
    )


class DepthReader:
    """Read Unitree's raw 16-bit depth frames from a V4L2 loopback device."""

    def __init__(self, device: str):
        self.device = device
        self.width, self.height = self._detect_resolution()
        self.frame_size = self.width * self.height * np.dtype(np.uint16).itemsize
        self.fd: Optional[int] = None

    def _detect_resolution(self) -> Tuple[int, int]:
        try:
            output = subprocess.check_output(
                ["v4l2-ctl", "-d", self.device, "--get-fmt-video"],
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in output.splitlines():
                if "Width/Height" in line:
                    dimensions = line.split(":", 1)[1].strip().split("/")
                    return int(dimensions[0]), int(dimensions[1])
        except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
            pass
        return DEFAULT_WIDTH, DEFAULT_HEIGHT

    def open(self) -> None:
        try:
            self.fd = os.open(self.device, os.O_RDWR | os.O_NONBLOCK)
        except OSError:
            self.fd = os.open(self.device, os.O_RDONLY | os.O_NONBLOCK)

    def read(self, timeout_s: float = 0.15) -> Optional[np.ndarray]:
        if self.fd is None:
            raise RuntimeError("depth device is not open")

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                data = os.read(self.fd, self.frame_size)
            except BlockingIOError:
                time.sleep(0.002)
                continue

            if len(data) == self.frame_size:
                return np.frombuffer(data, dtype=np.uint16).reshape(
                    self.height, self.width
                )
            time.sleep(0.002)
        return None

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def box_distance_m(
    depth: np.ndarray,
    box: Tuple[int, int, int, int],
    inner_scale: float,
    min_depth_mm: int,
    max_depth_mm: int,
) -> Optional[float]:
    """Return a robust median depth from the central part of a detection box."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half_width = max(1.0, (x2 - x1) * inner_scale / 2.0)
    half_height = max(1.0, (y2 - y1) * inner_scale / 2.0)

    ix1 = max(0, int(cx - half_width))
    iy1 = max(0, int(cy - half_height))
    ix2 = min(depth.shape[1], int(cx + half_width) + 1)
    iy2 = min(depth.shape[0], int(cy + half_height) + 1)

    samples = depth[iy1:iy2, ix1:ix2]
    valid = samples[
        (samples != 0)
        & (samples != INVALID_DEPTH)
        & (samples >= min_depth_mm)
        & (samples <= max_depth_mm)
    ]
    if valid.size < 10:
        return None
    return float(np.median(valid)) / 1000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model")
    parser.add_argument("--rgb-port", type=int, default=5002)
    parser.add_argument("--depth-device", default="/dev/video-dep")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument(
        "--inner-scale",
        type=float,
        default=0.4,
        help="central fraction of each box used for depth (0 < value <= 1)",
    )
    parser.add_argument("--min-depth-mm", type=int, default=100)
    parser.add_argument("--max-depth-mm", type=int, default=10000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 < args.inner_scale <= 1.0:
        raise SystemExit("--inner-scale must be in the range (0, 1]")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing detector dependency. Install it with: pip install ultralytics"
        ) from exc

    capture = cv2.VideoCapture(gstreamer_pipeline(args.rgb_port), cv2.CAP_GSTREAMER)
    if not capture.isOpened():
        raise SystemExit(
            "Could not open the RGB stream. Check that the R1 push service is "
            "running and that OpenCV was built with GStreamer support."
        )

    depth_reader = DepthReader(args.depth_device)
    try:
        depth_reader.open()
    except OSError as exc:
        capture.release()
        raise SystemExit(
            f"Could not open {args.depth_device}. Run: sudo unitree_depth_start"
        ) from exc

    model = YOLO(args.model)
    latest_depth: Optional[np.ndarray] = None

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("RGB frame unavailable; retrying...")
                time.sleep(0.05)
                continue

            new_depth = depth_reader.read()
            if new_depth is not None:
                latest_depth = new_depth
            if latest_depth is None:
                continue

            # The documented left RGB and depth products are both 544x448.  Do
            # not silently scale coordinates if the service returns a mismatch:
            # that usually indicates a configuration or registration problem.
            if frame.shape[:2] != latest_depth.shape:
                raise RuntimeError(
                    f"RGB/depth size mismatch: {frame.shape[1]}x{frame.shape[0]} vs "
                    f"{latest_depth.shape[1]}x{latest_depth.shape[0]}"
                )

            result = model.predict(frame, conf=args.confidence, verbose=False)[0]
            annotated = frame.copy()

            for detection in result.boxes:
                x1, y1, x2, y2 = (int(value) for value in detection.xyxy[0].tolist())
                class_id = int(detection.cls[0])
                confidence = float(detection.conf[0])
                label = result.names[class_id]
                distance = box_distance_m(
                    latest_depth,
                    (x1, y1, x2, y2),
                    args.inner_scale,
                    args.min_depth_mm,
                    args.max_depth_mm,
                )
                distance_text = "no depth" if distance is None else f"{distance:.2f} m"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"{label} {confidence:.2f} | {distance_text}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("R1 object distance", annotated)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        capture.release()
        depth_reader.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
