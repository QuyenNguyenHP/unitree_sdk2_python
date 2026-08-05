import os
import sys
from typing import List, Tuple

import cv2
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient

try:
    import face_recognition
except ImportError:
    face_recognition = None


# Thư mục chứa ảnh mẫu để nhận diện.
# Cấu trúc nên là:
# faces/
#   Alice/
#     1.jpg
#     2.jpg
#   Bob/
#     1.jpg
KNOWN_FACES_DIR = "faces"


def load_known_faces(root_dir: str) -> Tuple[List[np.ndarray], List[str]]:
    """
    Đọc ảnh mẫu từ thư mục root_dir và tạo ra:
    - danh sách vector đặc trưng khuôn mặt
    - danh sách tên tương ứng
    """
    known_encodings: List[np.ndarray] = []
    known_names: List[str] = []

    if not os.path.isdir(root_dir):
        print(f"Không tìm thấy thư mục ảnh mẫu: {root_dir}")
        print("Hãy tạo thư mục theo dạng faces/TenNguoi/*.jpg")
        return known_encodings, known_names

    for person_name in os.listdir(root_dir):
        person_dir = os.path.join(root_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        for file_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, file_name)
            if not os.path.isfile(image_path):
                continue

            try:
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
            except Exception as exc:
                print(f"Bỏ qua ảnh lỗi {image_path}: {exc}")
                continue

            if not encodings:
                print(f"Không tìm thấy khuôn mặt trong ảnh mẫu: {image_path}")
                continue

            known_encodings.append(encodings[0])
            known_names.append(person_name)
            print(f"Đã nạp ảnh mẫu cho: {person_name} -> {image_path}")

    return known_encodings, known_names


def recognize_faces(
    frame: np.ndarray,
    known_encodings: List[np.ndarray],
    known_names: List[str],
    tolerance: float = 0.45,
) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """
    Nhận diện khuôn mặt trong 1 frame.
    Kết quả trả về gồm:
    - tên người
    - toạ độ khung mặt (top, right, bottom, left)
    """
    # Thư viện face_recognition dùng ảnh RGB, trong khi OpenCV đọc BGR.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Giảm kích thước frame để tăng tốc độ nhận diện.
    small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)

    face_locations = face_recognition.face_locations(small_frame, model="hog")
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    results: List[Tuple[str, Tuple[int, int, int, int]]] = []

    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        name = "Unknown"

        if known_encodings:
            matches = face_recognition.compare_faces(
                known_encodings, face_encoding, tolerance=tolerance
            )
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)

            if len(face_distances) > 0:
                best_match_index = int(np.argmin(face_distances))
                if matches[best_match_index]:
                    name = known_names[best_match_index]

        # Vì detect ở ảnh đã resize 0.5, cần nhân lại toạ độ.
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        results.append((name, (top, right, bottom, left)))

    return results


def main():
    if face_recognition is None:
        print("Thiếu thư viện face_recognition.")
        print("Cài bằng lệnh:")
        print("pip install face_recognition")
        sys.exit(1)

    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <networkInterface> [known_faces_dir]")
        print(f"Example: python3 {sys.argv[0]} enxa0cec86d95d6")
        print(f"Example: python3 {sys.argv[0]} enxa0cec86d95d6 ./faces")
        sys.exit(1)

    network_interface = sys.argv[1]
    known_faces_dir = sys.argv[2] if len(sys.argv) > 2 else KNOWN_FACES_DIR

    print("Dang nap du lieu khuon mat mau...")
    known_encodings, known_names = load_known_faces(known_faces_dir)
    print(f"Tong so mau da nap: {len(known_names)}")

    print("Khoi tao kenh DDS voi robot...")
    ChannelFactoryInitialize(0, network_interface)

    # Dung VideoClient de lay anh JPEG tu service videohub.
    client = VideoClient()
    client.SetTimeout(3.0)
    client.Init()

    print("Bat dau nhan frame tu camera R1. Nhan ESC de thoat.")

    while True:
        code, data = client.GetImageSample()
        if code != 0:
            print(f"Loi lay anh tu robot. code: {code}")
            break

        image_data = np.frombuffer(bytes(data), dtype=np.uint8)
        frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if frame is None:
            print("Khong giai ma duoc frame JPEG tu robot.")
            continue

        results = recognize_faces(frame, known_encodings, known_names)

        for name, (top, right, bottom, left) in results:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(
                frame,
                name,
                (left, max(top - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            print(f"Nhan dien duoc: {name}")

        cv2.imshow("R1 Face Recognition", frame)
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
