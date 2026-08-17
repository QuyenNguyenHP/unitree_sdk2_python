#!/usr/bin/env python3
"""YOLO-World object detection with metric stereo distance on Unitree R1."""

from __future__ import annotations

import argparse
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from r1_stereo_common import (
    StereoGeometry,
    StereoRtpCapture,
    compute_disparity,
    create_sgbm,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True, help="stereo .npz file")
    parser.add_argument("--left-port", type=int, default=5002)
    parser.add_argument("--right-port", type=int, default=5003)
    parser.add_argument("--width", type=int, default=544)
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument(
        "--capture-backend",
        choices=("auto", "opencv", "gstreamer-cli"),
        default="auto",
    )
    parser.add_argument("--model", default="yolov8s-worldv2.pt")
    parser.add_argument(
        "--classes",
        default="person,chair,bottle,cup,table,door,box",
        help="comma-separated open-vocabulary prompts",
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument(
        "--detect-every",
        type=int,
        default=1,
        help="run YOLO every N frames and reuse boxes between detections",
    )
    parser.add_argument("--device", default=None, help="e.g. cpu, 0, cuda:0")
    parser.add_argument("--num-disparities", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--inner-scale", type=float, default=0.4)
    parser.add_argument("--min-distance-m", type=float, default=0.15)
    parser.add_argument("--max-distance-m", type=float, default=15.0)
    parser.add_argument("--max-pair-delta-ms", type=float, default=80.0)
    parser.add_argument(
        "--distance-mode",
        choices=("z", "euclidean"),
        default="z",
        help="optical-axis Z or Euclidean camera-to-point distance",
    )
    return parser.parse_args()


def object_distance(
    points_3d: np.ndarray,
    disparity: np.ndarray,
    box: Tuple[int, int, int, int],
    inner_scale: float,
    min_distance_m: float,
    max_distance_m: float,
    mode: str,
) -> Optional[float]:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    half_width = max(2.0, (x2 - x1) * inner_scale / 2.0)
    half_height = max(2.0, (y2 - y1) * inner_scale / 2.0)
    ix1 = max(0, int(cx - half_width))
    iy1 = max(0, int(cy - half_height))
    ix2 = min(points_3d.shape[1], int(cx + half_width) + 1)
    iy2 = min(points_3d.shape[0], int(cy + half_height) + 1)

    roi_points = points_3d[iy1:iy2, ix1:ix2]
    roi_disparity = disparity[iy1:iy2, ix1:ix2]
    if mode == "euclidean":
        distances = np.linalg.norm(roi_points, axis=2)
    else:
        distances = roi_points[:, :, 2]

    valid = (
        np.isfinite(distances)
        & (roi_disparity > 0.5)
        & (distances >= min_distance_m)
        & (distances <= max_distance_m)
    )
    samples = distances[valid]
    if samples.size < 20:
        return None

    # Remove mixed foreground/background tails before taking the median.
    low, high = np.percentile(samples, (10, 90))
    trimmed = samples[(samples >= low) & (samples <= high)]
    if trimmed.size < 10:
        return None
    return float(np.median(trimmed))


def disparity_preview(
    disparity: np.ndarray, num_disparities: int, target_height: int
) -> np.ndarray:
    normalized = np.clip(disparity / float(num_disparities), 0.0, 1.0)
    image = (normalized * 255.0).astype(np.uint8)
    image = cv2.applyColorMap(image, cv2.COLORMAP_TURBO)
    image[disparity <= 0.5] = (20, 20, 20)
    if image.shape[0] != target_height:
        scale = target_height / image.shape[0]
        image = cv2.resize(image, None, fx=scale, fy=scale)
    return image


def main() -> int:
    args = parse_args()
    if not 0.0 < args.inner_scale <= 1.0:
        raise SystemExit("--inner-scale must be in (0, 1]")
    if args.detect_every < 1:
        raise SystemExit("--detect-every must be >= 1")
    prompts = [item.strip() for item in args.classes.split(",") if item.strip()]
    if not prompts:
        raise SystemExit("--classes must contain at least one prompt")

    try:
        from ultralytics import YOLOWorld
    except ImportError as exc:
        raise SystemExit("Install YOLO-World support: pip install ultralytics") from exc

    geometry = StereoGeometry(args.calibration)
    matcher = create_sgbm(args.num_disparities, args.block_size)
    model = YOLOWorld(args.model)
    model.set_classes(prompts)

    capture = StereoRtpCapture(
        args.left_port,
        args.right_port,
        width=args.width,
        height=args.height,
        backend=args.capture_backend,
    )
    capture.start()
    print(f"YOLO-World prompts: {prompts}")
    print(f"Stereo baseline: {geometry.baseline_m:.4f} m")
    print("Q/ESC: quit")

    previous_time = time.monotonic()
    fps = 0.0
    frame_index = 0
    cached_detections = []
    last_yolo_ms = 0.0
    try:
        while True:
            try:
                left, right, pair_delta_ms = capture.read_pair(
                    max_delta_ms=args.max_pair_delta_ms
                )
            except TimeoutError as exc:
                print(exc)
                continue

            processing_started = time.monotonic()
            left_rectified, right_rectified = geometry.rectify(left, right)
            disparity = compute_disparity(matcher, left_rectified, right_rectified)
            points_3d = geometry.points_from_disparity(disparity)
            stereo_ms = (time.monotonic() - processing_started) * 1000.0

            annotated = left_rectified.copy()
            if frame_index % args.detect_every == 0:
                predict_options = {
                    "conf": args.confidence,
                    "imgsz": args.image_size,
                    "verbose": False,
                }
                if args.device is not None:
                    predict_options["device"] = args.device
                yolo_started = time.monotonic()
                result = model.predict(left_rectified, **predict_options)[0]
                last_yolo_ms = (time.monotonic() - yolo_started) * 1000.0
                cached_detections = []
                for detection in result.boxes:
                    coords = detection.xyxy[0].detach().cpu().tolist()
                    x1, y1, x2, y2 = (int(value) for value in coords)
                    x1 = max(0, min(annotated.shape[1] - 1, x1))
                    y1 = max(0, min(annotated.shape[0] - 1, y1))
                    x2 = max(x1 + 1, min(annotated.shape[1], x2))
                    y2 = max(y1 + 1, min(annotated.shape[0], y2))
                    class_id = int(detection.cls[0].item())
                    confidence = float(detection.conf[0].item())
                    name = result.names[class_id]
                    cached_detections.append(
                        (x1, y1, x2, y2, name, confidence)
                    )

            for x1, y1, x2, y2, name, confidence in cached_detections:
                distance = object_distance(
                    points_3d,
                    disparity,
                    (x1, y1, x2, y2),
                    args.inner_scale,
                    args.min_distance_m,
                    args.max_distance_m,
                    args.distance_mode,
                )
                distance_text = "no depth" if distance is None else f"{distance:.2f} m"
                label = f"{name} {confidence:.2f} | {distance_text}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(20, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            frame_index += 1

            now = time.monotonic()
            instant_fps = 1.0 / max(now - previous_time, 1e-6)
            fps = instant_fps if fps == 0.0 else 0.9 * fps + 0.1 * instant_fps
            previous_time = now
            cv2.putText(
                annotated,
                f"FPS {fps:.1f} | stereo {stereo_ms:.0f} ms | "
                f"YOLO {last_yolo_ms:.0f} ms/{args.detect_every}f | "
                f"pair {pair_delta_ms:.1f} ms",
                (10, annotated.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            depth_view = disparity_preview(
                disparity, args.num_disparities, annotated.shape[0]
            )
            cv2.imshow(
                "R1 YOLO-World distance | disparity",
                np.hstack((annotated, depth_view)),
            )
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        capture.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
