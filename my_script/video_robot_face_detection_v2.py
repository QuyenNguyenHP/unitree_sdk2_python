#!/usr/bin/env python3
"""Detect faces in an R1 RTP/H.264 video stream received over UDP."""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


DEFAULT_UDP_PORT = 5001
DEFAULT_YUNET_MODEL = "face_detection_yunet_2026may.onnx"
DEFAULT_WIDTH = 544
DEFAULT_HEIGHT = 448


def create_face_detector(model_path: str):
    """Create a YuNet detector, falling back to Haar Cascade."""
    script_directory = Path(__file__).resolve().parent
    yunet_candidates = [Path(model_path), script_directory / model_path]
    yunet_path = next((path for path in yunet_candidates if path.is_file()), None)

    if hasattr(cv2, "FaceDetectorYN") and yunet_path is not None:
        detector = cv2.FaceDetectorYN.create(
            str(yunet_path),
            "",
            (320, 320),
            score_threshold=0.75,
            nms_threshold=0.3,
            top_k=5000,
        )
        return "yunet", detector

    cascade_filename = "haarcascade_frontalface_default.xml"
    cascade_candidates = [
        Path(cv2.data.haarcascades) / cascade_filename,
        Path("/usr/share/opencv4/haarcascades") / cascade_filename,
        Path("/usr/share/opencv/haarcascades") / cascade_filename,
        script_directory / "data" / cascade_filename,
    ]
    for cascade_path in cascade_candidates:
        if not cascade_path.is_file():
            continue
        cascade = cv2.CascadeClassifier(str(cascade_path))
        if not cascade.empty():
            print(f"Haar Cascade: {cascade_path}")
            return "haar", cascade

    searched_paths = "\n  - ".join(str(path) for path in cascade_candidates)
    raise RuntimeError(
        "Khong tai duoc Haar Cascade va cung khong co model YuNet. "
        f"Da tim Haar Cascade tai:\n  - {searched_paths}"
    )


def detect_faces(
    detector_type: str, detector, frame: np.ndarray
) -> List[Tuple[int, int, int, int]]:
    """Return detected faces as (x, y, width, height)."""
    if detector_type == "yunet":
        height, width = frame.shape[:2]
        detector.setInputSize((width, height))
        _, faces = detector.detect(frame)
        results: List[Tuple[int, int, int, int]] = []

        if faces is not None:
            for face in faces:
                x, y, face_width, face_height = face[:4].astype(int)
                results.append((x, y, face_width, face_height))
        return results

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )
    return list(faces)


def gstreamer_pipeline(port: int, latency_ms: int) -> str:
    """Build an OpenCV/GStreamer receiver for an RTP/H.264 UDP stream."""
    return (
        f"udpsrc port={port} "
        'caps="application/x-rtp,media=video,clock-rate=90000,'
        'encoding-name=H264,payload=96" ! '
        f"rtpjitterbuffer latency={latency_ms} drop-on-latency=true ! "
        "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


class UdpVideoCapture:
    """Receive RTP/H.264 using OpenCV GStreamer or gst-launch as fallback."""

    def __init__(self, port: int, latency_ms: int, width: int, height: int):
        self.port = port
        self.width = width
        self.height = height
        self.frame_size = width * height * 3
        self.capture: Optional[cv2.VideoCapture] = None
        self.process: Optional[subprocess.Popen] = None

        if "GStreamer:                   YES" in cv2.getBuildInformation():
            self.capture = cv2.VideoCapture(
                gstreamer_pipeline(port, latency_ms), cv2.CAP_GSTREAMER
            )
            if self.capture.isOpened():
                print("Video backend: OpenCV + GStreamer")
                return
            self.capture.release()
            self.capture = None

        if shutil.which("gst-launch-1.0") is None:
            raise RuntimeError(
                "OpenCV khong co GStreamer va khong tim thay gst-launch-1.0. "
                "Hay cai GStreamer hoac dung OpenCV co GStreamer."
            )

        command = [
            "gst-launch-1.0",
            "-q",
            "udpsrc",
            f"port={port}",
            "caps=application/x-rtp,media=video,clock-rate=90000,"
            "encoding-name=H264,payload=96",
            "!",
            "rtpjitterbuffer",
            f"latency={latency_ms}",
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
            "videoscale",
            "!",
            f"video/x-raw,format=BGR,width={width},height={height}",
            "!",
            "fdsink",
            "fd=1",
            "sync=false",
        ]
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=0,
        )
        time.sleep(0.1)
        if self.process.poll() is not None:
            raise RuntimeError("GStreamer receiver da thoat ngay khi khoi dong.")
        print("Video backend: gst-launch-1.0")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.capture is not None:
            return self.capture.read()

        assert self.process is not None and self.process.stdout is not None
        data = bytearray()
        while len(data) < self.frame_size:
            chunk = self.process.stdout.read(self.frame_size - len(data))
            if not chunk:
                return False, None
            data.extend(chunk)
        frame = np.frombuffer(data, dtype=np.uint8).reshape(
            self.height, self.width, 3
        )
        return True, frame.copy()

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect faces in the R1 RTP/H.264 stream received over UDP."
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=DEFAULT_YUNET_MODEL,
        help="YuNet ONNX model (falls back to Haar Cascade if unavailable)",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_UDP_PORT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument(
        "--latency-ms",
        type=int,
        default=100,
        help="RTP jitter-buffer latency in milliseconds",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    if args.latency_ms < 0:
        raise SystemExit("--latency-ms must be zero or greater")
    if args.width <= 0 or args.height <= 0:
        raise SystemExit("--width and --height must be greater than zero")

    try:
        detector_type, detector = create_face_detector(args.model)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    if detector_type == "yunet":
        print(f"Dang dung YuNet model: {args.model}")
    else:
        print("Khong tim thay model YuNet, fallback ve Haar Cascade.")

    try:
        capture = UdpVideoCapture(
            args.port, args.latency_ms, args.width, args.height
        )
    except RuntimeError as exc:
        raise SystemExit(
            f"Khong mo duoc RTP/H.264 UDP port {args.port}. {exc}"
        ) from exc

    print(f"Dang nhan RTP/H.264 qua UDP port {args.port}. Nhan ESC hoac q de thoat.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print(f"Chua nhan duoc frame tu UDP port {args.port}; dang thu lai...")
                time.sleep(0.05)
                continue

            faces = detect_faces(detector_type, detector, frame)

            for x, y, width, height in faces:
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + width, y + height),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    "Face",
                    (x, max(y - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            cv2.putText(
                frame,
                f"Faces detected: {len(faces)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.imshow("R1 UDP Camera + Face Detect", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
