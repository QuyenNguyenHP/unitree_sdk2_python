"""Shared RTP capture and stereo geometry helpers for the Unitree R1."""

from __future__ import annotations

import threading
import time
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def rtp_h264_pipeline(port: int, latency_ms: int = 100) -> str:
    """Build an OpenCV/GStreamer pipeline for an R1 RTP/H.264 stream."""
    return (
        f"udpsrc port={port} "
        'caps="application/x-rtp,media=video,clock-rate=90000,'
        'encoding-name=H264,payload=96" ! '
        f"rtpjitterbuffer latency={latency_ms} drop-on-latency=true ! "
        "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


@dataclass(frozen=True)
class FrameSample:
    sequence: int
    received_at: float
    image: np.ndarray


class _CameraReader:
    def __init__(
        self,
        port: int,
        latency_ms: int,
        width: int,
        height: int,
        backend: str,
    ):
        self.port = port
        self.pipeline = rtp_h264_pipeline(port, latency_ms)
        self.latency_ms = latency_ms
        self.width = width
        self.height = height
        self.frame_size = width * height * 3
        self.backend = backend
        self.capture: Optional[cv2.VideoCapture] = None
        self.process: Optional[subprocess.Popen] = None
        self.sample: Optional[FrameSample] = None
        self.error: Optional[str] = None
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self.backend == "auto":
            build_info = cv2.getBuildInformation()
            self.backend = (
                "opencv" if "GStreamer:                   YES" in build_info
                else "gstreamer-cli"
            )

        if self.backend == "opencv":
            self.capture = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
            if not self.capture.isOpened():
                raise RuntimeError(
                    f"Cannot open UDP port {self.port} through OpenCV. Check "
                    "incoming packets and OpenCV GStreamer support."
                )
        elif self.backend == "gstreamer-cli":
            self._start_gstreamer_process()
        else:
            raise ValueError(f"Unknown capture backend: {self.backend}")

        print(f"UDP {self.port}: capture backend={self.backend}")
        self.thread = threading.Thread(
            target=self._run,
            name=f"r1-camera-{self.port}",
            daemon=True,
        )
        self.thread.start()

    def _start_gstreamer_process(self) -> None:
        command = [
            "gst-launch-1.0",
            "-q",
            "udpsrc",
            f"port={self.port}",
            "caps=application/x-rtp,media=video,clock-rate=90000,"
            "encoding-name=H264,payload=96",
            "!",
            "rtpjitterbuffer",
            f"latency={self.latency_ms}",
            "drop-on-latency=true",
            "!",
            "rtph264depay",
            "!",
            "h264parse",
            "!",
            "avdec_h264",
            "!",
            "videoconvert",
            "!",
            f"video/x-raw,format=BGR,width={self.width},height={self.height}",
            "!",
            "fdsink",
            "fd=1",
            "sync=false",
        ]
        try:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=None,
                bufsize=0,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("gst-launch-1.0 is not installed") from exc

        time.sleep(0.1)
        if self.process.poll() is not None:
            raise RuntimeError(
                f"GStreamer receiver for UDP port {self.port} exited immediately"
            )

    def _run(self) -> None:
        sequence = 0
        while not self.stop_event.is_set():
            if self.backend == "opencv":
                assert self.capture is not None
                ok, image = self.capture.read()
            else:
                image = self._read_gstreamer_frame()
                ok = image is not None
            if not ok:
                if not self.stop_event.is_set():
                    with self.condition:
                        self.error = f"Lost video on UDP port {self.port}"
                        self.condition.notify_all()
                time.sleep(0.01)
                continue
            sequence += 1
            sample = FrameSample(sequence, time.monotonic(), image)
            with self.condition:
                self.sample = sample
                self.error = None
                self.condition.notify_all()

    def _read_gstreamer_frame(self) -> Optional[np.ndarray]:
        assert self.process is not None and self.process.stdout is not None
        data = bytearray()
        while len(data) < self.frame_size and not self.stop_event.is_set():
            chunk = self.process.stdout.read(self.frame_size - len(data))
            if not chunk:
                return None
            data.extend(chunk)
        if len(data) != self.frame_size:
            return None
        return np.frombuffer(data, dtype=np.uint8).reshape(
            self.height, self.width, 3
        ).copy()

    def latest(self) -> Optional[FrameSample]:
        with self.condition:
            return self.sample

    def stop(self) -> None:
        self.stop_event.set()
        if self.capture is not None:
            self.capture.release()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=1.0)
        if self.thread is not None:
            self.thread.join(timeout=1.0)


class StereoRtpCapture:
    """Decode two R1 streams concurrently and return nearest arrival-time pairs.

    OpenCV does not expose the RTP timestamp at appsink, so ``pair_delta_ms`` is
    based on local decoder arrival time. This is suitable for initial R1 stereo
    experiments, but true moving-scene stereo should pair frames by source/RTP
    timestamps using a native GStreamer application.
    """

    def __init__(
        self,
        left_port: int = 5002,
        right_port: int = 5003,
        latency_ms: int = 100,
        width: int = 544,
        height: int = 448,
        backend: str = "auto",
    ):
        self.left = _CameraReader(left_port, latency_ms, width, height, backend)
        self.right = _CameraReader(right_port, latency_ms, width, height, backend)
        self.last_left_sequence = 0
        self.last_right_sequence = 0

    def start(self) -> None:
        self.left.start()
        try:
            self.right.start()
        except Exception:
            self.left.stop()
            raise

    def read_pair(
        self,
        timeout_s: float = 2.0,
        max_delta_ms: float = 80.0,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            left = self.left.latest()
            right = self.right.latest()
            if (
                left is not None
                and right is not None
                and left.sequence != self.last_left_sequence
                and right.sequence != self.last_right_sequence
            ):
                delta_ms = abs(left.received_at - right.received_at) * 1000.0
                if delta_ms <= max_delta_ms:
                    self.last_left_sequence = left.sequence
                    self.last_right_sequence = right.sequence
                    return left.image, right.image, delta_ms
            time.sleep(0.002)

        errors = [message for message in (self.left.error, self.right.error) if message]
        detail = f" ({'; '.join(errors)})" if errors else ""
        raise TimeoutError(f"Timed out waiting for a stereo pair{detail}")

    def stop(self) -> None:
        self.left.stop()
        self.right.stop()


class StereoGeometry:
    """Rectify stereo frames and convert disparity to metric 3D points."""

    REQUIRED_KEYS = ("K1", "D1", "K2", "D2", "R", "T", "image_size")

    def __init__(self, calibration_path: str | Path):
        calibration_path = Path(calibration_path)
        if not calibration_path.is_file():
            raise FileNotFoundError(f"Calibration file not found: {calibration_path}")

        with np.load(calibration_path, allow_pickle=False) as values:
            missing = [key for key in self.REQUIRED_KEYS if key not in values]
            if missing:
                raise ValueError(f"Calibration file is missing: {', '.join(missing)}")
            self.K1 = values["K1"].astype(np.float64)
            self.D1 = values["D1"].astype(np.float64)
            self.K2 = values["K2"].astype(np.float64)
            self.D2 = values["D2"].astype(np.float64)
            self.R = values["R"].astype(np.float64)
            self.T = values["T"].astype(np.float64).reshape(3, 1)
            image_size_array = values["image_size"].astype(int).reshape(2)

        self.image_size = (int(image_size_array[0]), int(image_size_array[1]))
        baseline_m = float(np.linalg.norm(self.T))
        if not np.isfinite(baseline_m) or not 0.01 <= baseline_m <= 1.0:
            raise ValueError(
                f"Invalid stereo baseline {baseline_m:.6f} m. Calibration T must "
                "be expressed in metres."
            )
        if self.K1.shape != (3, 3) or self.K2.shape != (3, 3):
            raise ValueError("K1 and K2 must both be 3x3 camera matrices")

        R1, R2, P1, P2, self.Q, _, _ = cv2.stereoRectify(
            self.K1,
            self.D1,
            self.K2,
            self.D2,
            self.image_size,
            self.R,
            self.T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
        )
        self.left_map = cv2.initUndistortRectifyMap(
            self.K1, self.D1, R1, P1, self.image_size, cv2.CV_32FC1
        )
        self.right_map = cv2.initUndistortRectifyMap(
            self.K2, self.D2, R2, P2, self.image_size, cv2.CV_32FC1
        )
        self.baseline_m = baseline_m

    def rectify(
        self, left: np.ndarray, right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        expected_shape = (self.image_size[1], self.image_size[0])
        if left.shape[:2] != expected_shape or right.shape[:2] != expected_shape:
            raise ValueError(
                f"Calibration is for {self.image_size[0]}x{self.image_size[1]}, "
                f"but frames are {left.shape[1]}x{left.shape[0]} and "
                f"{right.shape[1]}x{right.shape[0]}"
            )
        left_rectified = cv2.remap(left, *self.left_map, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right, *self.right_map, cv2.INTER_LINEAR)
        return left_rectified, right_rectified

    def points_from_disparity(self, disparity: np.ndarray) -> np.ndarray:
        return cv2.reprojectImageTo3D(disparity, self.Q)


def create_sgbm(num_disparities: int, block_size: int) -> cv2.StereoSGBM:
    if num_disparities <= 0 or num_disparities % 16 != 0:
        raise ValueError("num_disparities must be a positive multiple of 16")
    if block_size < 3 or block_size % 2 == 0:
        raise ValueError("block_size must be an odd integer >= 3")
    channels = 1
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * channels * block_size * block_size,
        P2=32 * channels * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def compute_disparity(
    matcher: cv2.StereoSGBM,
    left_rectified: np.ndarray,
    right_rectified: np.ndarray,
) -> np.ndarray:
    left_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
    return matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
