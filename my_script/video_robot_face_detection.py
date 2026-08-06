import os
import sys
from typing import List, Tuple

import cv2
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient


DEFAULT_YUNET_MODEL = "face_detection_yunet_2023mar.onnx"


def create_face_detector(model_path: str):
    """
    Tao detector khuon mat.
    Uu tien dung YuNet neu:
    - OpenCV co FaceDetectorYN
    - file model .onnx ton tai

    Neu khong du dieu kien thi fallback ve Haar Cascade.
    """
    if hasattr(cv2, "FaceDetectorYN") and model_path and os.path.isfile(model_path):
        detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            (320, 320),
            score_threshold=0.75,
            nms_threshold=0.3,
            top_k=5000,
        )
        return "yunet", detector

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError("Khong tai duoc Haar Cascade va cung khong co model YuNet.")
    return "haar", cascade


def detect_faces(detector_type: str, detector, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Tra ve danh sach khung mat theo dang (x, y, w, h).
    """
    if detector_type == "yunet":
        h, w = frame.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)
        results: List[Tuple[int, int, int, int]] = []

        if faces is not None:
            for face in faces:
                x, y, fw, fh = face[:4].astype(int)
                results.append((x, y, fw, fh))

        return results

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )
    return list(faces)


def main():
    # Can truyen ten network interface ket noi voi robot R1.
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <networkInterface> [yunet_model.onnx]")
        print(f"Example: python3 {sys.argv[0]} enxa0cec86d95d6")
        print(f"Example: python3 {sys.argv[0]} enxa0cec86d95d6 ./face_detection_yunet_2023mar.onnx")
        sys.exit(1)

    network_interface = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_YUNET_MODEL

    print("Buoc 1: kiem tra R1 co tra anh ve duoc khong.")
    print("Buoc 2: detect khuon mat tren anh, chua nhan dien ten.")
    print(f"Su dung network interface: {network_interface}")

    # Khoi tao DDS voi dung card mang noi toi robot.
    ChannelFactoryInitialize(0, network_interface)

    # Tao client lay anh tu service videohub.
    client = VideoClient()
    client.SetTimeout(3.0)
    client.Init()

    try:
        detector_type, detector = create_face_detector(model_path)
    except RuntimeError as exc:
        print(str(exc))
        sys.exit(1)

    if detector_type == "yunet":
        print(f"Dang dung YuNet model: {model_path}")
    else:
        print("Khong tim thay model YuNet, fallback ve Haar Cascade.")

    print("Dang nhan frame tu R1. Nhan ESC de thoat.")

    while True:
        code, data = client.GetImageSample()
        if code != 0:
            print(f"Loi lay anh tu robot. code: {code}")
            break

        # Chuyen byte JPEG thanh anh OpenCV.
        image_data = np.frombuffer(bytes(data), dtype=np.uint8)
        frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if frame is None:
            print("Khong giai ma duoc frame JPEG.")
            continue

        faces = detect_faces(detector_type, detector, frame)

        # Ve khung quanh mat va hien so luong detect duoc.
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
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

        cv2.imshow("R1 Camera + Face Detect", frame)

        # ESC de thoat.
        if cv2.waitKey(20) == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
